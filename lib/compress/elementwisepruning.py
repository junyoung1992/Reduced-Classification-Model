from ..training import evaluate
from torch import nn, optim

import copy
import math
import numpy as np
import os
import time
import torch

# Reference:
# S. Han, J. Pool, J. Tran, and W. J. Dally,
# "Learning both Weights and Connections for Efficient Neural Networks,"
# in International Conference on Neural Information Processing Systems (NIPS), 2015, vol. 1, pp. 1135-1143.

def iterative_pruning(Pruning, model, threshold, loops, train_dl, valid_dl, epochs, learning_rate, scheduling=None, steps=None, decay=None, partition=0, save_name=None, save_mode="best_acc"):
    old_score = evaluate(model, valid_dl)
    print()

    pct_pruned = 0.0
    
    for l in range(loops):
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        if scheduling is not None:
            scheduler = optim.lr_scheduler.MultiStepLR(opt, steps, decay)
        else:
            scheduler = None
        
        pruned_stat = Pruning.prune(threshold)
        new_pct_pruned, (pruned_p, orig_p) = pruned_stat["percentile"], pruned_stat["parameters"]
        
        print("Loop: {:d}".format(l + 1))
        print("Threshold: {:f}".format(threshold))
        print("Pruning: {:.2f}% ({:d} / {:d})".format(new_pct_pruned * 100, pruned_p, orig_p))
        
        i = 1
        for layer in pruned_stat["layers"]:
            if len(layer) == 2:
                print("\tLayer {:d} remaining parameters: {:.2f}% ({:d} / {:d})" \
                      .format(i, ((layer[1] - layer[0]) / float(layer[1])) * 100, (layer[1] - layer[0]), layer[1]))
                i += 1
            else:
                print("\tDropout rate: {:.2f}".format(layer[0]))
        
        new_score = evaluate(model, valid_dl)
        
        _save_name = save_name
        if _save_name is not None:
            if partition == 0:
                _save_name += "_pruned"
            else:
                _save_name += "_pruned" + "_" + str(partition)

        Pruning.fit(train_dl, valid_dl, loss_fn, opt, epochs, scheduler, _save_name, save_mode)
        print()

class Pruning():
    def __init__(self, model):
        self.model = model
        self.num_layers = 0
        self.num_dropout_layers = 0
        self.dropout_rates = {}

        self.count_layers()

        self.weight_masks = [None for _ in range(self.num_layers)]
        self.bias_masks = [None for _ in range(self.num_layers)]

    def count_layers(self):
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.num_layers += 1
            elif isinstance(layer, nn.Dropout):
                self.dropout_rates[self.num_dropout_layers] = layer.p
                self.num_dropout_layers += 1
                
    def prune_rate(self, ratio):
        all_weights = []
        non_zero_weights = []
        threshold_count = 0
        
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                all_weights += list(layer.weight.data.cpu().abs().numpy().flatten())
        
        non_zero_weights = [weight for weight in all_weights if weight > 0]
        threshold = torch.tensor(np.percentile(np.array(non_zero_weights), ratio))
        
        for weight in non_zero_weights:
            if weight >= threshold:
                threshold_count += 1
        
        # print("Threshold: {:f} ({:d}/{:d})".format(threshold, threshold_count,len(non_zero_weights)))
        
        return threshold

    def prune(self, threshold):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        index = 0
        dropout_index = 0

        num_pruned = 0
        num_weights = 0
        
        pruned_layer_stat = []

        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
                weight_mask = layer.weight.data.cpu().abs() >= threshold
                weight_mask = weight_mask.to(dtype=torch.float).to(dev)
                self.weight_masks[index] = weight_mask

                bias_mask = torch.ones(layer.bias.data.size())
                bias_mask = bias_mask.to(dev)
                for i in range(bias_mask.size(0)):
                    if len(torch.nonzero(weight_mask[i]).size()) == 0:
                        bias_mask[i] = 0
                self.bias_masks[index] = bias_mask

                weights_num = torch.numel(layer.weight.data)
                layer_pruned = weights_num - torch.nonzero(weight_mask).size(0)
                bias_num = torch.numel(bias_mask)
                bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)

                layer.weight.data *= weight_mask
                layer.bias.data *= bias_mask
                
                num_pruned += layer_pruned
                num_weights += weights_num
                
                pruned_layer_stat += [[layer_pruned, weights_num]]
                
                index += 1

            elif isinstance(layer, nn.Dropout):
                mask = self.weight_masks[index - 1]
                layer.p = self.dropout_rates[dropout_index] * math.sqrt(torch.nonzero(mask).size(0) / torch.numel(mask))
                dropout_index += 1
                pruned_layer_stat += [[layer.p]]
        
        statistics = {
            "percentile": num_pruned / num_weights,
            "parameters": (num_pruned, num_weights),
            "layers": pruned_layer_stat,
        }

        return statistics

    def set_grad(self):
        index = 0
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                layer.weight.grad.data *= self.weight_masks[index]
                layer.bias.grad.data *= self.bias_masks[index]
                index += 1
    
    def fit(self, train_dl, valid_dl, loss_fn, optimizer, num_epochs, scheduler=None, save_name=None, save_mode="all"):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        since = time.time()

        if save_name is not None:
            if not(os.path.isdir("save_models/pruned_model")):
                os.makedirs(os.path.join("save_models/pruned_model"))
            
            save_toggle = True
            best_model_wts = copy.deepcopy(self.model.state_dict())
        else:
            save_toggle = False

        best_acc, best_acc_epoch = 0.0, 0
        best_loss, best_loss_epoch = 99999.9, 0

        train_loss, train_acc = [], []
        valid_loss, valid_acc = [], []
        
        print('After prune, re-train')

        for epoch in range(num_epochs):
            print('\tEpoch {}/{}'.format(epoch + 1, num_epochs), end = ' ')

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                since_2 = time.time()
                if phase == 'train':
                    self.model.train()   # Set model to training mode
                    dl = train_dl
                    
                    for p in optimizer.param_groups:
                        print("(lr: {:f})".format(p['lr']), end = ' ')
                else:
                    self.model.eval()    # Set model to evaluate mode
                    dl = valid_dl

                running_loss = 0.0
                running_corrects = 0
                dataset_size = 0

                # Iterate over data
                for xb, yb in dl:
                    xb = xb.to(dev)
                    yb = yb.to(dev)
                    dataset_size += len(xb)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(xb)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fn(outputs, yb)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.set_grad()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * xb.size(0)
                    running_corrects += torch.sum(preds == yb.data)

                epoch_loss = running_loss / dataset_size
                epoch_acc = float((running_corrects.double() / dataset_size))

                # statistics
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                else:
                    valid_loss.append(epoch_loss)
                    valid_acc.append(epoch_acc)

                time_elapsed_2 = time.time() - since_2

                print('{} ({:.0f}m {:.0f}s) Loss: {:.4f} Acc: {:.2f}% ({:d}/{:d})'\
                      .format(phase, time_elapsed_2 // 60, time_elapsed_2 % 60,
                              epoch_loss, epoch_acc * 100, running_corrects, dataset_size), end = ' ')

                # deep copy the model
                if phase == 'val' and (epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss <= best_loss)):
                    best_acc = epoch_acc
                    best_acc_epoch = epoch

                    if save_toggle and save_mode == "best_acc":
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        save_name_pt = save_name + "_" + save_mode + ".pt"
                        save_path = os.path.join(os.getcwd(), 'save_models/pruned_model', save_name_pt)
                        torch.save(best_model_wts, save_path)

                if phase == 'val' and (epoch_loss < best_loss or (epoch_loss == best_loss and epoch_acc >= best_acc)):
                    best_loss = epoch_loss
                    best_loss_epoch = epoch

                    if save_toggle and save_mode == "best_loss":
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        save_name_pt = save_name + "_" + save_mode + ".pt"
                        save_path = os.path.join(os.getcwd(), 'save_models/pruned_model', save_name_pt)
                        torch.save(best_model_wts, save_path)

                if save_toggle and save_mode == "all":
                    save_name_pt = save_name + "_epoch_" + str(epoch + 1) + ".pt"
                    save_path = os.path.join(os.getcwd(), 'save_models/pruned_model', save_name_pt)
                    torch.save(self.model.state_dict(), save_path)
                
            if scheduler is not None:
                scheduler.step()

            print()

        time_elapsed = time.time() - since
        print('Re-training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('\tBest val Acc: {:.2f}% (epoch: {:d})'.format(best_acc * 100, best_acc_epoch + 1))
        print('\tBest val Loss: {:.4f} (epoch: {:d})'.format(best_loss, best_loss_epoch + 1))
        
        # load best model weights
        if save_mode == "best_acc" or save_mode == "best_loss":
            self.model.load_state_dict(best_model_wts)
            print('Save {} model'.format(save_mode))
            print('\treturn: {} epoch'.format(save_mode))
        elif save_mode == "all":
            print('Save {} models'.format(save_mode))
            print('\treturn: last epoch')
            
        statistics = {
            "train": (train_loss, train_acc),
            "valid": (valid_loss, valid_acc),
        }

        return statistics