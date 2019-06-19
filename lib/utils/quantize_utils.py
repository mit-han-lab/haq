# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import torch
import torch.nn as nn
from progress.bar import Bar
from sklearn.cluster import KMeans


def k_means_cpu(weight, n_clusters, init='k-means++', max_iter=50):
    # flatten the weight for computing k-means
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)  # single feature
    if n_clusters > weight.size:
        n_clusters = weight.size

    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).cuda().view(1, -1), torch.from_numpy(labels).int().cuda()


def reconstruct_weight_from_k_means_result(centroids, labels):
    weight = torch.zeros_like(labels).float().cuda()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight


def quantize_model(model, quantize_index, quantize_bits, max_iter=50, mode='cpu', quantize_bias=False,
                   centroids_init='k-means++', is_pruned=False, free_high_bit=False):
    assert len(quantize_index) == len(quantize_bits), \
        'You should provide the same number of bit setting as layer list!'
    if free_high_bit:
        # quantize weight with high bit will not lead accuracy loss, so we can omit them to save time
        quantize_bits = [-1 if i > 6 else i for i in quantize_bits]
    quantize_layer_bit_dict = {n: b for n, b in zip(quantize_index, quantize_bits)}
    centroid_label_dict = {}

    bar = Bar('KMeans:', max=len(quantize_index))
    for i, layer in enumerate(model.modules()):
        if i not in quantize_index:
            continue
        this_cl_list = []
        n_bit = quantize_layer_bit_dict[i]
        if n_bit < 0:  # if -1, do not quantize
            continue
        if type(n_bit) == list:  # given both the bit of weight and bias
            assert len(n_bit) == 2
            assert hasattr(layer, 'weight')
            assert hasattr(layer, 'bias')
        else:
            n_bit = [n_bit, n_bit]  # using same setting for W and b
        # quantize weight
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            if is_pruned:
                nz_mask = w.ne(0)
                print('*** pruned density: {:.4f}'.format(torch.sum(nz_mask) / w.numel()))
                ori_shape = w.size()
                w = w[nz_mask]
            if mode == 'cpu':
                centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** n_bit[0], init=centroids_init, max_iter=max_iter)
            else:
                raise NotImplementedError
            if is_pruned:
                full_labels = labels.new(ori_shape).zero_() - 1  # use -1 for pruned elements
                full_labels[nz_mask] = labels
                labels = full_labels
            this_cl_list.append([centroids, labels])
            w_q = reconstruct_weight_from_k_means_result(centroids, labels)
            layer.weight.data = w_q.float()
        # quantize bias
        if hasattr(layer, 'bias') and quantize_bias:
            w = layer.bias.data
            if mode == 'cpu':
                centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** n_bit[1], init=centroids_init, max_iter=max_iter)
            else:
                raise NotImplementedError
            this_cl_list.append([centroids, labels])
            w_q = reconstruct_weight_from_k_means_result(centroids, labels)
            layer.bias.data = w_q.float()

        centroid_label_dict[i] = this_cl_list

        bar.suffix = ' id: {id:} | bit: {bit:}'.format(id=i, bit=n_bit[0])
        bar.next()
    bar.finish()
    return centroid_label_dict


def kmeans_update_model(model, quantizable_idx, centroid_label_dict, free_high_bit=False):
    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        new_weight_data = layer.weight.data.clone()
        new_weight_data.zero_()
        this_cl_list = centroid_label_dict[i]
        num_centroids = this_cl_list[0][0].numel()
        if num_centroids > 2**6 and free_high_bit:
            # quantize weight with high bit will not lead accuracy loss, so we can omit them to save time
            continue
        for j in range(num_centroids):
            mask_cl = (this_cl_list[0][1] == j).float()
            new_weight_data += (layer.weight.data * mask_cl).sum() / mask_cl.sum() * mask_cl
        layer.weight.data = new_weight_data


