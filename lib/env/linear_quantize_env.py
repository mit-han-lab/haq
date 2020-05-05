# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import os
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
from lib.utils.quantize_utils import QConv2d, QLinear, calibrate


class LinearQuantizeEnv:
    def __init__(self, model, pretrained_model, data, data_root, compress_ratio, args, n_data_worker=16,
                 batch_size=256, float_bit=8, is_model_pruned=False):
        # default setting
        self.quantizable_layer_types = [QConv2d, QLinear]

        # save options
        self.model = model
        self.model_for_measure = deepcopy(model)
        self.model_name = args.arch
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
        self.last_weight_action = self.max_bit
        self.last_activation_action = self.max_bit
        self.action_radio_button = True

        self.is_inception = args.arch.startswith('inception')
        self.is_imagenet = ('imagenet' in data)
        self.use_top5 = args.use_top5

        # init reward
        self.best_reward = -math.inf

        # prepare data
        self._init_data()

        # build indexs
        self._build_index()
        self.n_quantizable_layer = len(self.quantizable_idx)

        self.model.load_state_dict(self.pretrained_model, strict=True)
        # self.org_acc = self._validate(self.val_loader, self.model)
        self.org_acc = self._validate(self.train_loader, self.model)
        # build embedding (static part), same as pruning
        self._build_state_embedding()

        # mode
        self.cost_mode = 'cloud_latency'
        self.simulator_batch = 16
        self.cost_lookuptable = self._get_lookuptable()

        # sanity check
        assert self.compress_ratio > self._min_cost() / self._org_cost(), \
            'Error! You can make achieve compress_ratio smaller than min_bit!'

        # restore weight
        self.reset()
        print('=> original acc: {:.3f}% on split dataset(train: %7d, val: %7d )'.format(self.org_acc,
                                                                                        self.train_size, self.val_size))
        print('=> original cost: {:.4f}'.format(self._org_cost()))

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma

    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        action = self._action_wall(action)  # percentage to preserve

        if self.action_radio_button:
            self.last_weight_action = action
        else:
            self.last_activation_action = action
            self.strategy.append([self.last_weight_action, self.last_activation_action])  # save action to strategy

        # all the actions are made
        if self._is_final_layer() and (not self.action_radio_button):
            self._final_action_wall()
            assert len(self.strategy) == len(self.quantizable_idx)
            cost = self._cur_cost()
            cost_ratio = cost / self._org_cost()

            self._set_mixed_precision(quantizable_idx=self.quantizable_idx, strategy=self.strategy)
            self.model = calibrate(self.model, self.train_loader)
            if self.finetune_flag:
                acc = self._finetune(self.train_loader, self.model, epochs=self.finetune_epoch, verbose=False)
                # train_acc = self._finetune(self.train_loader, self.model, epochs=self.finetune_epoch, verbose=False)
                # acc = self._validate(self.val_loader, self.model)
            else:
                acc = self._validate(self.val_loader, self.model)

            # reward = self.reward(acc, w_size_ratio)
            reward = self.reward(acc)

            info_set = {'cost_ratio': cost_ratio, 'accuracy': acc, 'cost': cost}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, cost_ratio: {:.3f}'.format(
                    self.strategy, self.best_reward, acc, cost_ratio))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            self.action_radio_button = not self.action_radio_button
            return obs, reward, done, info_set

        cost = self._cur_cost()
        info_set = {'cost': cost}
        reward = 0
        done = False

        if self.action_radio_button:
            self.layer_embedding[self.cur_ind][-1] = 0.0
        else:
            self.cur_ind += 1  # the index of next layer
            self.layer_embedding[self.cur_ind][-1] = 1.0
        self.layer_embedding[self.cur_ind][-2] = float(action) / float(self.max_bit)
        self.layer_embedding[self.cur_ind][-1] = float(self.action_radio_button)
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        self.action_radio_button = not self.action_radio_button
        return obs, reward, done, info_set

    # for quantization
    def reward(self, acc, cost_ratio=None):
        if cost_ratio is not None:
            return (acc - self.org_acc + 1. / cost_ratio) * 0.1
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
        target = self.compress_ratio * self._org_cost()
        min_cost = 0
        for i, n_bit in enumerate(self.strategy):
            min_cost += self.cost_lookuptable[i][int(self.min_bit-1)][int(self.min_bit-1)]

        print('before action_wall: ', self.strategy, min_cost, self._cur_cost())
        while min_cost < self._cur_cost() and target < self._cur_cost():
            # print('current: ', self.strategy, min_cost, self._cur_cost())
            for i, n_bit in enumerate(reversed(self.strategy)):
                if n_bit[1] > self.min_bit:
                    self.strategy[-(i+1)][1] -= 1
                self._keep_first_last_layer()
                if target >= self._cur_cost():
                    break
                if n_bit[0] > self.min_bit:
                    self.strategy[-(i+1)][0] -= 1
                self._keep_first_last_layer()
                if target >= self._cur_cost():
                    break
        print('after action_wall: ', self.strategy, min_cost, self._cur_cost())

    def _keep_first_last_layer(self):
        self.strategy[0][0] = 8
        # self.strategy[0][1] = 8
        # input image is already 8 bit
        self.strategy[0][1] = -1
        self.strategy[-1][0] = 8
        self.strategy[-1][1] = 8

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind
        # limit the action to certain range
        action = float(action)
        min_bit, max_bit = self.bound_list[self.cur_ind]
        lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        return action  # not constrained here

    def _set_mixed_precision(self, quantizable_idx, strategy):
        assert len(quantizable_idx) == len(strategy), \
            'You should provide the same number of bit setting as layer list for weight quantization!'
        quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}
        for i, layer in enumerate(self.model.modules()):
            if i not in quantizable_idx:
                continue
            else:
                layer.w_bit = quantize_layer_bit_dict[i][0]
                layer.a_bit = quantize_layer_bit_dict[i][1]

    def _cur_cost(self):
        cur_cost = 0.
        # quantized
        for i, n_bit in enumerate(self.strategy):
            cur_cost += self.cost_lookuptable[i, n_bit[0]-1, n_bit[1]-1]
        return cur_cost

    def _org_cost(self):
        org_cost = 0
        for i in range(self.cost_lookuptable.shape[0]):
            org_cost += self.cost_lookuptable[i, int(self.float_bit-1), int(self.float_bit-1)]
        return org_cost

    def _min_cost(self):
        min_cost = 0
        for i in range(self.cost_lookuptable.shape[0]):
            if i == 0 or i == (self.cost_lookuptable.shape[0] - 1):
                min_cost += self.cost_lookuptable[i, -1, -1]
            else:
                min_cost += self.cost_lookuptable[i, int(self.min_bit - 1), int(self.min_bit - 1)]
        return min_cost

    def _init_data(self):
        self.train_loader, self.val_loader, n_class = get_split_train_dataset(
            self.data_type, self.batch_size, self.n_data_worker, data_root=self.data_root,
            val_size=self.val_size, train_size=self.train_size, for_inception=self.is_inception)

    def _build_index(self):
        self.quantizable_idx = []
        self.bound_list = []
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.quantizable_layer_types:
                self.quantizable_idx.append(i)
                self.bound_list.append((self.min_bit, self.max_bit))
        print('=> Final bound list: {}'.format(self.bound_list))

    def _build_state_embedding(self):
        # measure model for cifar 32x32 input
        if self.is_imagenet:
            measure_model(self.model_for_measure, 224, 224)
        else:
            measure_model(self.model_for_measure, 32, 32)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())
        for i, ind in enumerate(self.quantizable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d or type(m) == QConv2d:
                this_state.append([int(m.in_channels == m.groups)])  # layer type, 1 for conv_dw
                this_state.append([m.in_channels])  # in channels
                this_state.append([m.out_channels])  # out channels
                this_state.append([m.stride[0]])  # stride
                this_state.append([m.kernel_size[0]])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size
            elif type(m) == nn.Linear or type(m) == QLinear:
                this_state.append([0.])  # layer type, 0 for fc
                this_state.append([m.in_features])  # in channels
                this_state.append([m.out_features])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size

            this_state.append([i])  # index
            this_state.append([1.])  # bits, 1 is the max bit
            this_state.append([1.])  # action radio button, 1 is the weight action
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

    def _get_lookuptable(self):

        lookup_table_folder = 'lib/simulator/lookup_tables/'
        os.makedirs(lookup_table_folder, exist_ok=True)
        if self.cost_mode == 'cloud_latency':
            fname = lookup_table_folder + self.model_name + '_' + self.data_type \
                    + '_batch' + str(self.simulator_batch) + '_latency_table.npy'
        else:
            # add your own cost lookuptable here
            raise NotImplementedError

        if os.path.isfile(fname):
            print('load latency table : ', fname)
            latency_list = np.load(fname)
            print(latency_list)
        else:
            # you can put your own simulator/lookuptable here
            raise NotImplementedError
        return latency_list.copy()

    def _finetune(self, train_loader, model, epochs=1, verbose=True):
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

