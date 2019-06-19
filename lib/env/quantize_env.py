# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import time
import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from progress.bar import Bar

from lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
from lib.utils.data_utils import get_split_train_dataset
from lib.utils.quantize_utils import quantize_model, kmeans_update_model


class QuantizeEnv:
    def __init__(self, model, pretrained_model, data, data_root, compress_ratio, args, n_data_worker=16,
                 batch_size=256, float_bit=32, is_model_pruned=False):
        # default setting
        self.quantizable_layer_types = [nn.Conv2d, nn.Linear]

        # save options
        self.model = model
        self.model_for_measure = deepcopy(model)
        self.cur_ind = 0
        self.strategy = []  # quantization strategy

        self.finetune_lr = args.finetune_lr
        self.optimizer = optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=0.9, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.pretrained_model = pretrained_model
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.data_root = data_root
        self.compress_ratio = compress_ratio
        self.is_model_pruned = is_model_pruned
        self.val_size = args.val_size
        self.train_size = args.train_size
        self.finetune_gamma = args.finetune_gamma
        self.finetune_lr = args.finetune_lr
        self.finetune_flag = args.finetune_flag
        self.finetune_epoch = args.finetune_epoch

        # options from args
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.float_bit = float_bit * 1.
        self.last_action = self.max_bit

        self.is_inception = args.arch.startswith('inception')
        self.is_imagenet = ('imagenet' in data)
        self.use_top5 = args.use_top5

        # sanity check
        assert self.compress_ratio > self.min_bit * 1. / self.float_bit, \
            'Error! You can make achieve compress_ratio smaller than min_bit!'

        # init reward
        self.best_reward = -math.inf

        # prepare data
        self._init_data()

        # build indexs
        self._build_index()
        self._get_weight_size()
        self.n_quantizable_layer = len(self.quantizable_idx)

        self.model.load_state_dict(self.pretrained_model, strict=True)
        self.org_acc = self._validate(self.val_loader, self.model)
        # build embedding (static part), same as pruning
        self._build_state_embedding()

        # restore weight
        self.reset()
        print('=> original acc: {:.3f}% on split dataset(train: %7d, val: %7d )'.format(self.org_acc,
                                                                                        self.train_size, self.val_size))
        print('=> original #param: {:.4f}, model size: {:.4f} MB'.format(sum(self.wsize_list) * 1. / 1e6,
                                                                         sum(self.wsize_list) * self.float_bit / 8e6))

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma

    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        action = self._action_wall(action)  # percentage to preserve

        self.strategy.append(action)  # save action to strategy

        # all the actions are made
        if self._is_final_layer():
            self._final_action_wall()
            assert len(self.strategy) == len(self.quantizable_idx)
            w_size = self._cur_weight()
            w_size_ratio = self._cur_weight() / self._org_weight()

            centroid_label_dict = quantize_model(self.model, self.quantizable_idx, self.strategy,
                                                 mode='cpu', quantize_bias=False, centroids_init='k-means++',
                                                 is_pruned=self.is_model_pruned, max_iter=3)
            if self.finetune_flag:
                train_acc = self._kmeans_finetune(self.train_loader, self.model, self.quantizable_idx,
                                                  centroid_label_dict, epochs=self.finetune_epoch, verbose=False)
                acc = self._validate(self.val_loader, self.model)
            else:
                acc = self._validate(self.val_loader, self.model)

            # reward = self.reward(acc, w_size_ratio)
            reward = self.reward(acc)

            info_set = {'w_ratio': w_size_ratio, 'accuracy': acc, 'w_size': w_size}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    self.strategy, self.best_reward, acc, w_size_ratio))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            return obs, reward, done, info_set

        w_size = self._cur_weight()
        info_set = {'w_size': w_size}
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer
        self.layer_embedding[self.cur_ind][-1] = action
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        return obs, reward, done, info_set

    # for quantization
    def reward(self, acc, w_size_ratio=None):
        if w_size_ratio is not None:
            return (acc - self.org_acc + 1. / w_size_ratio) * 0.1
        return (acc - self.org_acc) * 0.1

    def reset(self):
        # restore env by loading the pretrained model
        self.model.load_state_dict(self.pretrained_model, strict=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.finetune_lr, momentum=0.9, weight_decay=4e-5)
        self.cur_ind = 0
        self.strategy = []  # quantization strategy
        obs = self.layer_embedding[0].copy()
        return obs

    def _is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1

    def _final_action_wall(self):
        target = self.compress_ratio * self._org_weight()
        min_weight = 0
        for i, n_bit in enumerate(self.strategy):
            min_weight += self.wsize_list[i] * self.min_bit
        while min_weight < self._cur_weight() and target < self._cur_weight():
            for i, n_bit in enumerate(reversed(self.strategy)):
                if n_bit > self.min_bit:
                    self.strategy[-(i+1)] -= 1
                if target >= self._cur_weight():
                    break
        print('=> Final action list: {}'.format(self.strategy))

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind
        # limit the action to certain range
        action = float(action)
        min_bit, max_bit = self.bound_list[self.cur_ind]
        lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        self.last_action = action
        return action  # not constrained here

    def _cur_weight(self):
        cur_weight = 0.
        # quantized
        for i, n_bit in enumerate(self.strategy):
            cur_weight += n_bit * self.wsize_list[i]
        return cur_weight

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_bitops - self._cur_bitops()
        return reduced

    def _org_weight(self):
        org_weight = 0.
        org_weight += sum(self.wsize_list) * self.float_bit
        return org_weight

    def _init_data(self):
        self.train_loader, self.val_loader, n_class = get_split_train_dataset(
            self.data_type, self.batch_size, self.n_data_worker, data_root=self.data_root,
            val_size=self.val_size, train_size=self.train_size, for_inception=self.is_inception)

    def _build_index(self):
        self.quantizable_idx = []
        self.layer_type_list = []
        self.bound_list = []
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.quantizable_layer_types:
                self.quantizable_idx.append(i)
                self.layer_type_list.append(type(m))
                self.bound_list.append((self.min_bit, self.max_bit))
        print('=> Final bound list: {}'.format(self.bound_list))

    def _get_weight_size(self):
        # get the param size for each layers to prune, size expressed in number of params
        self.wsize_list = []
        for i, m in enumerate(self.model.modules()):
            if i in self.quantizable_idx:
                if not self.is_model_pruned:
                    self.wsize_list.append(m.weight.data.numel())
                else:  # the model is pruned, only consider non-zeros items
                    self.wsize_list.append(torch.sum(m.weight.data.ne(0)))
        self.wsize_dict = {i: s for i, s in zip(self.quantizable_idx, self.wsize_list)}

    def _get_latency_list(self):
        # use simulator to get the latency
        raise NotImplementedError

    def _get_energy_list(self):
        # use simulator to get the energy
        raise NotImplementedError

    def _build_state_embedding(self):
        # measure model for cifar 32x32 input
        if self.is_imagenet:
            measure_model(self.model_for_measure, 224, 224)
        else:
            measure_model(self.model_for_measure, 32, 32)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())
        real_module_list = list(self.model.modules())
        for i, ind in enumerate(self.quantizable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d or type(m) == nn.Conv2d:
                this_state.append([int(m.in_channels == m.groups)])  # layer type, 1 for conv_dw
                this_state.append([m.in_channels])  # in channels
                this_state.append([m.out_channels])  # out channels
                this_state.append([m.stride[0]])  # stride
                this_state.append([m.kernel_size[0]])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size
            elif type(m) == nn.Linear or type(m) == nn.Linear:
                this_state.append([0.])  # layer type, 0 for fc
                this_state.append([m.in_features])  # in channels
                this_state.append([m.out_features])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size

            this_state.append([i])  # index
            this_state.append([4.])  # bits
            layer_embedding.append(np.hstack(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def _kmeans_finetune(self, train_loader, model, idx, centroid_label_dict, epochs=1, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_acc = 0.

        # switch to train mode
        model.train()
        end = time.time()
        t1 = time.time()
        bar = Bar('train:', max=len(train_loader))
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                input_var, target_var = inputs.cuda(), targets.cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient
                self.optimizer.zero_grad()
                loss.backward()

                # do SGD step
                self.optimizer.step()

                kmeans_update_model(model, self.quantizable_idx, centroid_label_dict, free_high_bit=True)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()

            if self.use_top5:
                if top5.avg > best_acc:
                    best_acc = top5.avg
            else:
                if top1.avg > best_acc:
                    best_acc = top1.avg
            self.adjust_learning_rate()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        return best_acc

    def _validate(self, val_loader, model, verbose=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        t1 = time.time()
        with torch.no_grad():
            # switch to evaluate mode
            model.eval()

            end = time.time()
            bar = Bar('valid:', max=len(val_loader))
            for i, (inputs, targets) in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                input_var, target_var = inputs.cuda(), targets.cuda()

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        if self.use_top5:
            return top5.avg
        else:
            return top1.avg

