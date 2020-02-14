from ..training import fit, evaluate, save_model
from ..models import *

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

class Pruning:
    def __init__(self, model, train_dl, valid_dl):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.train_dl = train_dl
        self.valid_dl = valid_dl

        self.model = model.to(self.dev)
        self.pruner = Pruner(self.model)
    
    def iterative_pruning(self, one_epoch_remove, finetuning=False,
                          epoch=None, lr=None, save_name=None, save_mode=None):
        evaluate(self.model, self.valid_dl)

        filters = 0
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                filters = filters + layer.out_channels
        
        print("* Ranking Filters")
        # get candidates to prune
        self.pruner.reset()
        print("\t1. Search")
        self.pruner.search(self.train_dl)
        print("\t2. Normalization")
        self.pruner.normalize_ranks_per_layer()
        print("\t3. Select filters and make plan")
        prune_targets = self.pruner.get_prunning_plan(one_epoch_remove)

        self.model = self.model.cpu()
        for layer_index, filter_index in prune_targets:
            if isinstance(self.model, VGG):
                self.prune_vgg16_conv_layer(layer_index, filter_index)
            elif isinstance(self.model, MobileNet):
                self.prune_mobilenet_conv_layer(layer_index, filter_index)
            elif isinstance(self.model, LeNet5):
                self.prune_lenet5_conv_layer(layer_index, filter_index)
        
        new_filters = 0
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                new_filters = new_filters + layer.out_channels
        
        print("* Pruning Filters: {:d} -> {:d}".format(filters, new_filters))
        
        if finetuning == True:
            print("* Fine tuning")
            loss_fn = nn.CrossEntropyLoss()
            opt = optim.Adam(self.model.parameters(), lr=lr)
            scheduler = None
            # scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=25, cooldown=25, min_lr=1e-6)

            self.model = self.model.to(self.dev)
            history = fit(self.model, self.train_dl, self.valid_dl, loss_fn, opt, epoch, scheduler, save_name, save_mode)
            return history
        elif finetuning == False:
            save_name_pt = save_name + "_No_Retraining" + ".pt"
            save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
            save_model(self.model, path=save_path)
    
    def prune_lenet5_conv_layer(self, layer_index, filter_index):
        layer_list, layer_structure_count = self.lenet5_layer_list()

        conv = layer_list[layer_index][1]
        if layer_index + 2 < len(layer_list):
            next_conv = layer_list[layer_index + 3][1]
        else:
            next_conv = None
            next_linear = self.model.classifier[0]
        
        # conv
        new_conv = nn.Conv2d(in_channels=conv.in_channels,
                            out_channels=conv.out_channels-1,
                            kernel_size=conv.kernel_size,
                            stride=conv.stride,
                            padding=conv.padding)
        
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()
        new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
        new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
        new_conv.weight.data = torch.from_numpy(new_weights)
        new_conv.weight.data = new_conv.weight.to(self.dev)

        old_bias = conv.bias.data.cpu().numpy()
        new_bias = new_conv.bias.data.cpu().numpy()
        new_bias[: filter_index] = old_bias[: filter_index]
        new_bias[filter_index :] = old_bias[filter_index + 1 :]
        new_conv.bias.data = torch.from_numpy(new_bias)
        new_conv.bias.data = new_conv.bias.to(self.dev)

        if next_conv != None:
            new_next_conv = nn.Conv2d(in_channels=next_conv.in_channels-1,
                                    out_channels=next_conv.out_channels,
                                    kernel_size=next_conv.kernel_size,
                                    stride=next_conv.stride,
                                    padding=next_conv.padding)
            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = new_next_conv.weight.data.cpu().numpy()
            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[: , filter_index :, :, :] = old_weights[:, filter_index + 1 :, :, :]
            new_next_conv.weight.data = torch.from_numpy(new_weights)
            new_next_conv.weight.data = new_next_conv.weight.to(self.dev)

            new_next_conv.bias.data = next_conv.bias.data
            new_next_conv.bias.data = new_next_conv.bias.to(self.dev)
        else:
            params_per_input_channel = next_linear.in_features // conv.out_channels
            new_next_linear = nn.Linear(in_features=next_linear.in_features - params_per_input_channel,
                                        out_features=next_linear.out_features)
            
            old_weights = next_linear.weight.data.cpu().numpy()
            new_weights = new_next_linear.weight.data.cpu().numpy() 

            new_weights[:, : filter_index * params_per_input_channel] = old_weights[:, : filter_index * params_per_input_channel]
            new_weights[:, filter_index * params_per_input_channel :] = old_weights[:, (filter_index + 1) * params_per_input_channel :]
            new_next_linear.weight.data = torch.from_numpy(new_weights)
            new_next_linear.weight.data = new_next_linear.weight.to(self.dev)

            new_next_linear.bias.data = next_linear.bias.data
            new_next_linear.bias.data = new_next_linear.bias.to(self.dev)

        # exchange       
        if layer_index == 0:
            self.model.features[0] = new_conv
            self.model.features[3] = new_next_conv
        elif layer_index == 3:
            self.model.features[3] = new_conv
            self.model.features[6] = new_next_conv
        elif layer_index == 6:
            self.model.features[6] = new_conv
            self.model.classifier[0] = new_next_linear
    
    def lenet5_layer_list(self):
        layer_list = []
        for name, layer in list(self.model.features._modules.items()):
            layer_list.append((name, layer))

        layer = OrderedDict(
            {
                0: len(self.model.features),
            }
        )

        return layer_list, layer

    def prune_mobilenet_conv_layer(self, layer_index, filter_index):
        layer_list, layer_structure_count = self.mobilenet_layer_list()

        conv = layer_list[layer_index][1]
        next_bn = layer_list[layer_index + 1][1]
        if layer_index + 3 < len(layer_list):
            next_conv_dw = layer_list[layer_index + 3][1]
            next_conv_dw_bn = layer_list[layer_index + 4][1]
            next_conv_pw = layer_list[layer_index + 6][1]
            next_conv_pw_bn = layer_list[layer_index + 7][1]
        else:
            next_conv_dw = None
            next_conv_dw_bn = None
            next_conv_pw = None
            next_conv_pw_bn = None
            next_linear = self.model.classifier[0]
        
        # conv
        new_conv = nn.Conv2d(in_channels=conv.in_channels,
                            out_channels=conv.out_channels-1,
                            kernel_size=conv.kernel_size,
                            stride=conv.stride,
                            padding=conv.padding,
                            bias=False)
        
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()
        new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
        new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
        new_conv.weight.data = torch.from_numpy(new_weights)
        new_conv.weight.data = new_conv.weight.to(self.dev)

        # next_bn
        new_next_bn = nn.BatchNorm2d(num_features=new_conv.out_channels)

        if next_conv_dw != None:
            # next_conv_dw
            new_next_conv_dw = nn.Conv2d(in_channels=new_conv.out_channels,
                                         out_channels=new_conv.out_channels,
                                         kernel_size=next_conv_dw.kernel_size,
                                         stride=next_conv_dw.stride,
                                         padding=next_conv_dw.padding,
                                         groups=new_conv.out_channels,
                                         bias=False)
        
            old_weights = next_conv_dw.weight.data.cpu().numpy()
            new_weights = new_next_conv_dw.weight.data.cpu().numpy()
            new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
            new_weights[filter_index :, :, :, :] = old_weights[filter_index + 1 :, :, :, :]
            new_next_conv_dw.weight.data = torch.from_numpy(new_weights)
            new_next_conv_dw.weight.data = new_next_conv_dw.weight.to(self.dev)

            # next_conv_dw_bn
            new_next_conv_dw_bn = nn.BatchNorm2d(num_features=new_next_conv_dw.out_channels)

            # next_conv_pw
            new_next_conv_pw = nn.Conv2d(in_channels=new_next_conv_dw.out_channels,
                                         out_channels=next_conv_pw.out_channels,
                                         kernel_size=next_conv_pw.kernel_size,
                                         stride=next_conv_pw.stride,
                                         padding=next_conv_pw.padding,
                                         bias=False)
        
            old_weights = next_conv_pw.weight.data.cpu().numpy()
            new_weights = new_next_conv_pw.weight.data.cpu().numpy()
            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[:, filter_index :, :, :] = old_weights[:, filter_index + 1 :, :, :]
            new_next_conv_pw.weight.data = torch.from_numpy(new_weights)
            new_next_conv_pw.weight.data = new_next_conv_pw.weight.to(self.dev)

            # next_conv_pw_bn
            new_next_conv_pw_bn = nn.BatchNorm2d(num_features=new_next_conv_pw.out_channels)
        else:
            params_per_input_channel = next_linear.in_features // conv.out_channels
            new_linear = nn.Linear(in_features=next_linear.in_features - params_per_input_channel,
                                   out_features=next_linear.out_features)
            
            old_weights = next_linear.weight.data.cpu().numpy()
            new_weights = new_linear.weight.data.cpu().numpy() 

            new_weights[:, : filter_index * params_per_input_channel] = old_weights[:, : filter_index * params_per_input_channel]
            new_weights[:, filter_index * params_per_input_channel :] = old_weights[:, (filter_index + 1) * params_per_input_channel :]
            new_linear.weight.data = torch.from_numpy(new_weights)
            new_linear.weight.data = new_linear.weight.to(self.dev)

            new_linear.bias.data = next_linear.bias.data
            new_linear.bias.data = new_linear.bias.to(self.dev)

        # exchange
        block = 0
        count = 0
        for i in layer_structure_count:
            if layer_index < (layer_structure_count[i] + count):
                block = i
                break
            else:
                count += layer_structure_count[i]
        
        if block == 0:
            self.model.conv_1[0] = new_conv
            self.model.conv_1[1] = new_next_bn
            self.model.conv_DW_2[0] = new_next_conv_dw
            self.model.conv_DW_2[1] = new_next_conv_dw_bn
            self.model.conv_DW_2[3] = new_next_conv_pw
            self.model.conv_DW_2[4] = new_next_conv_pw_bn
        elif block == 1:
            self.model.conv_DW_2[3] = new_conv
            self.model.conv_DW_2[4] = new_next_bn
            self.model.conv_DW_3[0] = new_next_conv_dw
            self.model.conv_DW_3[1] = new_next_conv_dw_bn
            self.model.conv_DW_3[3] = new_next_conv_pw
            self.model.conv_DW_3[4] = new_next_conv_pw_bn
        elif block == 2:
            self.model.conv_DW_3[3] = new_conv
            self.model.conv_DW_3[4] = new_next_bn
            self.model.conv_DW_4[0] = new_next_conv_dw
            self.model.conv_DW_4[1] = new_next_conv_dw_bn
            self.model.conv_DW_4[3] = new_next_conv_pw
            self.model.conv_DW_4[4] = new_next_conv_pw_bn
        elif block == 3:
            self.model.conv_DW_4[3] = new_conv
            self.model.conv_DW_4[4] = new_next_bn
            self.model.conv_DW_5[0] = new_next_conv_dw
            self.model.conv_DW_5[1] = new_next_conv_dw_bn
            self.model.conv_DW_5[3] = new_next_conv_pw
            self.model.conv_DW_5[4] = new_next_conv_pw_bn
        elif block == 4:
            self.model.conv_DW_5[3] = new_conv
            self.model.conv_DW_5[4] = new_next_bn
            self.model.conv_DW_6[0] = new_next_conv_dw
            self.model.conv_DW_6[1] = new_next_conv_dw_bn
            self.model.conv_DW_6[3] = new_next_conv_pw
            self.model.conv_DW_6[4] = new_next_conv_pw_bn
        elif block == 5:
            self.model.conv_DW_6[3] = new_conv
            self.model.conv_DW_6[4] = new_next_bn
            self.model.conv_DW_7[0] = new_next_conv_dw
            self.model.conv_DW_7[1] = new_next_conv_dw_bn
            self.model.conv_DW_7[3] = new_next_conv_pw
            self.model.conv_DW_7[4] = new_next_conv_pw_bn
        elif block == 6:
            self.model.conv_DW_7[3] = new_conv
            self.model.conv_DW_7[4] = new_next_bn
            self.model.conv_DW_8_1[0] = new_next_conv_dw
            self.model.conv_DW_8_1[1] = new_next_conv_dw_bn
            self.model.conv_DW_8_1[3] = new_next_conv_pw
            self.model.conv_DW_8_1[4] = new_next_conv_pw_bn
        elif block == 7:
            self.model.conv_DW_8_1[3] = new_conv
            self.model.conv_DW_8_1[4] = new_next_bn
            self.model.conv_DW_8_2[0] = new_next_conv_dw
            self.model.conv_DW_8_2[1] = new_next_conv_dw_bn
            self.model.conv_DW_8_2[3] = new_next_conv_pw
            self.model.conv_DW_8_2[4] = new_next_conv_pw_bn
        elif block == 8:
            self.model.conv_DW_8_2[3] = new_conv
            self.model.conv_DW_8_2[4] = new_next_bn
            self.model.conv_DW_8_3[0] = new_next_conv_dw
            self.model.conv_DW_8_3[1] = new_next_conv_dw_bn
            self.model.conv_DW_8_3[3] = new_next_conv_pw
            self.model.conv_DW_8_3[4] = new_next_conv_pw_bn
        elif block == 9:
            self.model.conv_DW_8_3[3] = new_conv
            self.model.conv_DW_8_3[4] = new_next_bn
            self.model.conv_DW_8_4[0] = new_next_conv_dw
            self.model.conv_DW_8_4[1] = new_next_conv_dw_bn
            self.model.conv_DW_8_4[3] = new_next_conv_pw
            self.model.conv_DW_8_4[4] = new_next_conv_pw_bn
        elif block == 10:
            self.model.conv_DW_8_4[3] = new_conv
            self.model.conv_DW_8_4[4] = new_next_bn
            self.model.conv_DW_8_5[0] = new_next_conv_dw
            self.model.conv_DW_8_5[1] = new_next_conv_dw_bn
            self.model.conv_DW_8_5[3] = new_next_conv_pw
            self.model.conv_DW_8_5[4] = new_next_conv_pw_bn
        elif block == 11:
            self.model.conv_DW_8_5[3] = new_conv
            self.model.conv_DW_8_5[4] = new_next_bn
            self.model.conv_DW_9[0] = new_next_conv_dw
            self.model.conv_DW_9[1] = new_next_conv_dw_bn
            self.model.conv_DW_9[3] = new_next_conv_pw
            self.model.conv_DW_9[4] = new_next_conv_pw_bn
        elif block == 12:
            self.model.conv_DW_9[3] = new_conv
            self.model.conv_DW_9[4] = new_next_bn
            self.model.conv_DW_10[0] = new_next_conv_dw
            self.model.conv_DW_10[1] = new_next_conv_dw_bn
            self.model.conv_DW_10[3] = new_next_conv_pw
            self.model.conv_DW_10[4] = new_next_conv_pw_bn
        elif block == 13:
            self.model.conv_DW_10[3] = new_conv
            self.model.conv_DW_10[4] = new_next_bn
            self.model.classifier[0] = new_linear

    def mobilenet_layer_list(self):
        layer_list = []
        for name, layer in list(self.model.conv_1._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_2._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_3._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_4._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_5._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_6._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_7._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_8_1._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_8_2._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_8_3._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_8_4._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_8_5._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_9._modules.items()):
            layer_list.append((name, layer))
        for name, layer in list(self.model.conv_DW_10._modules.items()):
            layer_list.append((name, layer))

        layer = OrderedDict(
            {
                0: len(self.model.conv_1),
                1: len(self.model.conv_DW_2),
                2: len(self.model.conv_DW_3),
                3: len(self.model.conv_DW_4),
                4: len(self.model.conv_DW_5),
                5: len(self.model.conv_DW_6),
                6: len(self.model.conv_DW_7),
                7: len(self.model.conv_DW_8_1),
                8: len(self.model.conv_DW_8_2),
                9: len(self.model.conv_DW_8_3),
                10: len(self.model.conv_DW_8_4),
                11: len(self.model.conv_DW_8_5),
                12: len(self.model.conv_DW_9),
                13: len(self.model.conv_DW_10),
            }
        )

        return layer_list, layer

    def prune_vgg16_conv_layer(self, layer_index, filter_index):
        #
        # If model is changed, you revise this part.
        # This pruning works only for my VGG16.
        #
        layer_list, layer_structure_count = self.vgg16_layer_list()

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
        new_conv.weight.data = new_conv.weight.to(self.dev)

        old_bias = conv.bias.data.cpu().numpy()
        new_bias = new_conv.bias.data.cpu().numpy()
        new_bias[: filter_index] = old_bias[: filter_index]
        new_bias[filter_index :] = old_bias[filter_index + 1 :]
        new_conv.bias.data = torch.from_numpy(new_bias)
        new_conv.bias.data = new_conv.bias.to(self.dev)

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
        
        if block == 0:
            self.model.block_1[layer_in_block] = new_conv
        elif block == 1:
            self.model.block_2[layer_in_block] = new_conv
        elif block == 2:
            self.model.block_3[layer_in_block] = new_conv
        elif block == 3:
            self.model.block_4[layer_in_block] = new_conv
        elif block == 4:
            self.model.block_5[layer_in_block] = new_conv

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
            next_new_conv.weight.data = next_new_conv.weight.to(self.dev)

            next_new_conv.bias.data = next_conv.bias.data
            next_new_conv.bias.data = next_new_conv.bias.to(self.dev)

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
            
            if block == 0:
                self.model.block_1[layer_in_block] = next_new_conv
            elif block == 1:
                self.model.block_2[layer_in_block] = next_new_conv
            elif block == 2:
                self.model.block_3[layer_in_block] = next_new_conv
            elif block == 3:
                self.model.block_4[layer_in_block] = next_new_conv
            elif block == 4:
                self.model.block_5[layer_in_block] = next_new_conv
            
            del next_new_conv
            del next_conv
        
        else:
            old_linear = self.model.classifier[0]
            params_per_input_channel = old_linear.in_features // conv.out_channels
            new_linear = nn.Linear(in_features=old_linear.in_features - params_per_input_channel,
                                out_features=old_linear.out_features)
            
            old_weights = old_linear.weight.data.cpu().numpy()
            new_weights = new_linear.weight.data.cpu().numpy() 

            new_weights[:, : filter_index * params_per_input_channel] = old_weights[:, : filter_index * params_per_input_channel]
            new_weights[:, filter_index * params_per_input_channel :] = old_weights[:, (filter_index + 1) * params_per_input_channel :]
            new_linear.weight.data = torch.from_numpy(new_weights)
            new_linear.weight.data = new_linear.weight.to(self.dev)

            new_linear.bias.data = old_linear.bias.data
            new_linear.bias.data = new_linear.bias.to(self.dev)

            self.model.classifier[0] = new_linear

            del new_linear
            del old_linear
        
        del new_conv
        del conv

    def vgg16_layer_list(self):
        layer_list = []
        for _, layer in list(self.model.block_1._modules.items()):
            layer_list.append(layer)
        for _, layer in list(self.model.block_2._modules.items()):
            layer_list.append(layer)
        for _, layer in list(self.model.block_3._modules.items()):
            layer_list.append(layer)
        for _, layer in list(self.model.block_4._modules.items()):
            layer_list.append(layer)
        for _, layer in list(self.model.block_5._modules.items()):
            layer_list.append(layer)

        layer = OrderedDict(
            {
                0: len(self.model.block_1),
                1: len(self.model.block_2),
                2: len(self.model.block_3),
                3: len(self.model.block_4),
                4: len(self.model.block_5)
            }
        )

        return layer_list, layer

class Pruner():
    def __init__(self, model):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}   # self.activation_to_layer[index] = layer index in real model

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
                    if _name != "Conv_DW":
                        x.register_hook(self.compute_rank)
                        self.activations.append(x)
                        self.activation_to_layer[activation_index] = layer_count
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
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.dev)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def search(self, train_dl):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()

        for xb, yb in train_dl:
            xb = xb.to(self.dev)
            yb = yb.to(self.dev)

            self.model.zero_grad()

            output = self.forward(xb)

            loss = loss_fn(output, yb)
            loss.backward()
    
    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        
        return nsmallest(num, data, itemgetter(2))  # layer_index, filter_index, 
    
    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()
    
    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        filters_to_prune = sorted(filters_to_prune, key=lambda filters_to_prune: filters_to_prune[1], reverse=True)

        f_l = []
        for (i, f, _) in filters_to_prune:
            f_l.append((i, f))

        return f_l