from ..training import fit, evaluate, save_model

from collections import OrderedDict
from heapq import nsmallest
from operator import itemgetter
from torch import nn, optim
from torch.optim import lr_scheduler

import copy
import math
import numpy as np
import os
import time
import torch

# Refernece:
# P. Molchanov, S. Tyree, T. Karras, T. Aila, and J. Kautz,
# “Pruning Convolutional Neural Networks for Resource Efficient Inference,”
# in International Conference on Learning Representations (ICLR), 2017.

def iterative_pruning(Pruning, model, one_epoch_remove, finetuning,
                      train_dl, valid_dl, epoch, lr,
                      save_name, save_mode):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    evaluate(model, valid_dl)

    filters = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            filters = filters + layer.out_channels

    print("* Ranking Filters")
    # get candidates to prune
    Pruning.reset()
    print("\t1. Search")
    Pruning.search(train_dl)
    print("\t2. Normalization")
    Pruning.normalize_ranks_per_layer()
    print("\t3. Select filters and make plan")
    prune_targets = Pruning.get_prunning_plan(one_epoch_remove)

    print("* Pruning Filters: {:d} -> {:d}".format(filters, filters - one_epoch_remove))
    model = model.cpu()
    for layer_index, filter_index in prune_targets:
        # print("\t", layer_index, filter_index)
        model = prune_vgg16_conv_layer(model, layer_index, filter_index)

    if finetuning == True:
        print("* Fine tuning")
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=lr)
        # accuracy
        scheduler = None
        # scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=25, cooldown=25, min_lr=1e-6)

        model.to(dev)
        history = fit(model, train_dl, valid_dl, loss_fn, opt, epoch, scheduler, save_name, save_mode)

        return history
    elif finetuning == False:
        save_name_pt = save_name + "_No_Retraining" + ".pt"
        save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
        save_model(model, path=save_path)

def prune_vgg16_conv_layer(model, layer_index, filter_index):
    #
    # If model is changed, you revise this part.
    # This pruning works only for my VGG16.
    #
    layer_list, layer_structure_count = vgg16_layer_list(model)

    conv = layer_list[layer_index]
    next_conv = None
    offset = 1

    while layer_index + offset <  len(layer_list):
        res = layer_list[layer_index + offset]
        if isinstance(res, nn.Conv2d):
            next_conv = res
            next_conv_index = layer_index + offset
            break
        offset += 1
    
    new_conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels-1,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding)
    
    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    new_weights[:filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights)
    new_conv.weight.data = new_conv.weight.cuda()

    old_bias = conv.bias.data.cpu().numpy()
    new_bias = new_conv.bias.data.cpu().numpy()
    new_bias[: filter_index] = old_bias[: filter_index]
    new_bias[filter_index :] = old_bias[filter_index + 1 :]
    new_conv.bias.data = torch.from_numpy(new_bias)
    new_conv.bias.data = new_conv.bias.cuda()

    #
    # If model is changed, you revise this part.
    # This pruning works only for my VGG16.
    #
    block = 0
    layer_in_block = 0
    count = 0
    for i in layer_structure_count:
        if layer_index < (layer_structure_count[i] + count):
            block = i
            layer_in_block = layer_index - count
            break
        else:
            count += layer_structure_count[i]
    # print("\t\t({:d}, {:d})".format(block, layer_in_block))
    
    if block == 0:
        model.block_1[layer_in_block] = new_conv
    elif block == 1:
        model.block_2[layer_in_block] = new_conv
    elif block == 2:
        model.block_3[layer_in_block] = new_conv
    elif block == 3:
        model.block_4[layer_in_block] = new_conv
    elif block == 4:
        model.block_5[layer_in_block] = new_conv

    if not (next_conv is None):
        next_new_conv = nn.Conv2d(in_channels=next_conv.in_channels-1,
                                  out_channels=next_conv.out_channels,
                                  kernel_size=next_conv.kernel_size,
                                  stride=next_conv.stride,
                                  padding=next_conv.padding)
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()
        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[: , filter_index :, :, :] = old_weights[:, filter_index + 1 :, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights)
        next_new_conv.weight.data = next_new_conv.weight.cuda()

        next_new_conv.bias.data = next_conv.bias.data
        next_new_conv.bias.data = next_new_conv.bias.cuda()

        #
        # If model is changed, you revise this part.
        # This pruning works only for my VGG16.
        #
        block = 0
        layer_in_block = 0
        count = 0
        for i in layer_structure_count:
            if next_conv_index < (layer_structure_count[i] + count):
                block = i
                layer_in_block = next_conv_index - count
                break
            else:
                count += layer_structure_count[i]
        # print("\t\t\t+ ({:d}, {:d})".format(block, layer_in_block))
        
        if block == 0:
            model.block_1[layer_in_block] = next_new_conv
        elif block == 1:
            model.block_2[layer_in_block] = next_new_conv
        elif block == 2:
            model.block_3[layer_in_block] = next_new_conv
        elif block == 3:
            model.block_4[layer_in_block] = next_new_conv
        elif block == 4:
            model.block_5[layer_in_block] = next_new_conv
        
        del next_new_conv
        del next_conv
    
    else:
        old_linear = model.classifier[0]
        params_per_input_channel = old_linear.in_features // conv.out_channels
        new_linear = nn.Linear(in_features=old_linear.in_features - params_per_input_channel,
                               out_features=old_linear.out_features)
        
        old_weights = old_linear.weight.data.cpu().numpy()
        new_weights = new_linear.weight.data.cpu().numpy() 

        new_weights[:, : filter_index * params_per_input_channel] = old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] = old_weights[:, (filter_index + 1) * params_per_input_channel :]
        new_linear.weight.data = torch.from_numpy(new_weights)
        new_linear.weight.data = new_linear.weight.cuda()

        new_linear.bias.data = old_linear.bias.data
        new_linear.bias.data = new_linear.bias.cuda()

        model.classifier[0] = new_linear

        del new_linear
        del old_linear
    
    del new_conv
    del conv

    return model

def vgg16_layer_list(model):
    layer_list = []
    for _, layer in list(model.block_1._modules.items()):
        layer_list.append(layer)
    for _, layer in list(model.block_2._modules.items()):
        layer_list.append(layer)
    for _, layer in list(model.block_3._modules.items()):
        layer_list.append(layer)
    for _, layer in list(model.block_4._modules.items()):
        layer_list.append(layer)
    for _, layer in list(model.block_5._modules.items()):
        layer_list.append(layer)

    layer = OrderedDict(
        {
            0: len(model.block_1),
            1: len(model.block_2),
            2: len(model.block_3),
            3: len(model.block_4),
            4: len(model.block_5)
        }
    )

    return layer_list, layer

class Pruning():
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        layer_count = 0

        #
        # If model is changed, you revise this part.
        # This pruning works only for my VGG16.
        #
        x, _, _ = self.recursive_register_hook(self.model, x, activation_index, layer_count)

        return self.model.classifier(torch.flatten(x, 1))
    
    #
    # If model is changed, you revise this part.
    # This pruning works only for my VGG16.
    #
    def recursive_register_hook(self, module, x, activation_index, layer_count):
        for _layer, (_name, _module) in enumerate(module._modules.items()):
            if _name == "classifier":
                break
            elif isinstance(_module, nn.Sequential):
                x, activation_index, layer_count = self.recursive_register_hook(_module, x, activation_index, layer_count)
            else:
                x = _module(x)
                if isinstance(_module, nn.Conv2d):
                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                    # self.activation_to_layer[activation_index] = _layer
                    self.activation_to_layer[activation_index] = layer_count
                    # print(activation_index, "\t", layer_count, "(", _layer, ")", _name, _module)
                    activation_index += 1
                layer_count += 1

        return x, activation_index, layer_count
    
    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def search(self, train_dl):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()

        for xb, yb in train_dl:
            xb = xb.cuda()
            yb = yb.cuda()

            self.model.zero_grad()

            output = self.forward(xb)

            loss = loss_fn(output, yb)
            loss.backward()
    
    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()
    
    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        filters_to_prune = sorted(filters_to_prune, key=lambda filters_to_prune: filters_to_prune[1], reverse=True)

        # print(self.activation_to_layer)
        # print(len(filters_to_prune))
        # for (l, f, _) in filters_to_prune:
        #     print(l, "\t", f)

        '''
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
        
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune
        '''

        f_l = []
        for (i, f, _) in filters_to_prune:
            f_l.append((i, f))

        return f_l