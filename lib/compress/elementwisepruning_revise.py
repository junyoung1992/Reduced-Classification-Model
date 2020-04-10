from ..training import evaluate, save_model

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

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

class Pruning:
    def __init__(self, model, train_dl, valid_dl, threshold_ratio):
        self.__dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.__train_dl = train_dl
        self.__valid_dl = valid_dl

        self.__model = model.to(self.__dev)
        self.__pruner = Pruner(self.__model)
        self.__threshold = self.__pruner.prune_rate(threshold_ratio)
    
    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold_ratio):
        self.__threshold = self.pruner.prune_rate(threshold_ratio)

    def iterative_pruning(self, finetuning=False, epoch=None, lr=None, schedule=None, save_name=None, save_mode=None):
        print("* Evaluating Valid Sets")
        evaluate(self.__model, self.__valid_dl)
                
        print("* Pruning Weights")
        pruned_stat = self.__pruner.prune(self.__threshold)
        pruning_percent, (pruned_p, orig_p) = pruned_stat["percentile"], pruned_stat["parameters"]
        print("\tThreshold: {:f}".format(self.__threshold))
        print("\tPruning: {:.2f}% ({:d} / {:d})".format(pruning_percent * 100, (orig_p - pruned_p), orig_p))
        
        i = 1
        for layer in pruned_stat["layers"]:
            if layer[0] == "Conv2d" or layer[0] == "Linear":
                print("\tLayer {:d} remaining parameters: {:.2f}% ({:d} / {:d})" \
                    .format(i, ((layer[2] - layer[1]) / float(layer[2])) * 100, (layer[2] - layer[1]), layer[2]))
                i += 1
            else:
                print("\tDropout rate: {:.2f}".format(layer[1]))
        print()
        
        print("* Evaluating Valid Sets")
        evaluate(self.__model, self.__valid_dl)

        if finetuning == True:
            print("* Fine tuning")
            loss_fn = nn.CrossEntropyLoss()
            opt = optim.Adam(self.__model.parameters(), lr=lr)
        
            history = self.__pruner.fit(self.__train_dl, self.__valid_dl, loss_fn, opt, epoch, schedule, save_name, save_mode)
            return history
            
        elif finetuning == False:
            save_name_pt = save_name + "_No_Retraining" + ".pt"
            save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
            save_model(self.__model, path=save_path)
            return None

class Pruner:
    def __init__(self, model):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.__model = model
        self.__num_layers = 0
        self.__num_dropout_layers = 0
        self.__dropout_rates = {}

        self.__count_layers()

        self.__weight_masks = [None for _ in range(self.__num_layers)]
        self.__bias_masks = [None for _ in range(self.__num_layers)]

    def __count_layers(self):
        for layer in self.__model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.__num_layers += 1
            elif isinstance(layer, nn.Dropout):
                self.__dropout_rates[self.__num_dropout_layers] = layer.p
                self.__num_dropout_layers += 1
                
    def prune_rate(self, ratio):
        all_weights = []
        non_zero_weights = []
        threshold_count = 0
        
        for layer in self.__model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                all_weights += list(layer.weight.data.cpu().abs().numpy().flatten())
        
        non_zero_weights = [weight for weight in all_weights if weight > 0]
        threshold = torch.tensor(np.percentile(np.array(non_zero_weights), ratio))
        
        for weight in non_zero_weights:
            if weight >= threshold:
                threshold_count += 1
        
        return threshold

    def prune(self, threshold):
        

        index = 0
        dropout_index = 0

        num_pruned = 0
        num_weights = 0
        
        pruned_layer_stat = []

        for layer in self.__model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
                weight_mask = layer.weight.data.cpu().abs() >= threshold
                weight_mask = weight_mask.to(dtype=torch.float).to(self.dev)
                self.__weight_masks[index] = weight_mask

                bias_mask = torch.ones(layer.bias.data.size())
                bias_mask = bias_mask.to(self.dev)
                for i in range(bias_mask.size(0)):
                    if len(torch.nonzero(weight_mask[i]).size()) == 0:
                        bias_mask[i] = 0
                self.__bias_masks[index] = bias_mask

                weights_num = torch.numel(layer.weight.data)
                layer_pruned = weights_num - torch.nonzero(weight_mask).size(0)
                bias_num = torch.numel(bias_mask)
                bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)

                layer.weight.data *= weight_mask
                layer.bias.data *= bias_mask
                
                num_pruned += layer_pruned
                num_weights += weights_num
                
                if isinstance(layer, nn.Conv2d):
                    pruned_layer_stat += [("Conv2d", layer_pruned, weights_num)]
                else:
                    pruned_layer_stat += [("Linear", layer_pruned, weights_num)]
                
                index += 1

            elif isinstance(layer, nn.Dropout):
                mask = self.__weight_masks[index - 1]
                layer.p = self.__dropout_rates[dropout_index] * math.sqrt(torch.nonzero(mask).size(0) / torch.numel(mask))
                dropout_index += 1
                pruned_layer_stat += [("Dropout", layer.p)]
        
        statistics = {
            "percentile": num_pruned / num_weights,
            "parameters": (num_pruned, num_weights),
            "layers": pruned_layer_stat,
        }

        return statistics

    def __set_grad(self):
        index = 0
        for layer in self.__model.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                layer.weight.grad.data *= self.__weight_masks[index]
                layer.bias.grad.data *= self.__bias_masks[index]
                index += 1
    
    def fit(self, train_dl, valid_dl, loss_fn, optimizer, num_epochs, schedule=None, save_name=None, save_mode="all"):
        since = time.time()

        if save_name is not None:
            if not(os.path.isdir("save_models")):
                os.makedirs(os.path.join("save_models"))
                            
            save_toggle = True
            best_model_wts = copy.deepcopy(self.__model.state_dict())
        else:
            save_toggle = False

        best_acc, best_acc_epoch = 0.0, 0
        best_loss, best_loss_epoch = 99999.9, 0
        best_f1, best_f1_epoch = 0.0, 0

        train_loss, train_acc, train_f1 = [], [], []
        valid_loss, valid_acc, valid_f1 = [], [], []

        if schedule == "custom":
            countdown_epoch = 0
            toggle = 0
        if type(schedule) == list:
            scheduler = MultiStepLR(optimizer, milestones=schedule, gamma=0.1)

        for epoch in range(num_epochs):
            current_lr = 99999.9
            if schedule == "custom":
                if (toggle == 1) and (epoch > (countdown_epoch + 25)):
                    break

            print('Epoch {}/{}'.format(epoch + 1, num_epochs), end = ' ')

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                since_2 = time.time()
                if phase == 'train':
                    self.__model.train()   # Set model to training mode
                    dl = train_dl
                    
                    for p in optimizer.param_groups:
                        print("(lr: {:f})".format(p['lr']))
                else:
                    self.__model.eval()    # Set model to evaluate mode
                    dl = valid_dl

                running_loss = 0.0
                running_corrects = 0
                dataset_size = 0
                y_true, y_pred = [], []


                # Iterate over data
                for xb, yb in dl:
                    xb = xb.to(self.dev)
                    yb = yb.to(self.dev)
                    dataset_size += len(xb)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.__model(xb)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fn(outputs, yb)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.__set_grad()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * xb.size(0)
                    running_corrects += torch.sum(preds == yb.data)
                    y_true += yb.data.tolist()
                    y_pred += preds.tolist()
                
                time_elapsed_2 = time.time() - since_2

                epoch_loss = running_loss / dataset_size
                epoch_acc = float((running_corrects.double() / dataset_size))
                epoch_f1 = f1_score(y_pred, y_true, average='macro')

                print('\t{} ({:02.0f}m {:02.0f}s)\tLoss: {:.4f}\tAcc: {:.2f}% ({:d}/{:d})\tF1 score: {:.4f}'\
                  .format(phase, time_elapsed_2 // 60, time_elapsed_2 % 60,
                          epoch_loss, epoch_acc * 100, running_corrects, dataset_size, epoch_f1))

                # statistics
                if phase == 'train':
                    train_loss.append(float(epoch_loss))
                    train_acc.append(float(epoch_acc))
                    train_f1.append(float(epoch_f1))
                else:
                    valid_loss.append(float(epoch_loss))
                    valid_acc.append(float(epoch_acc))
                    valid_f1.append(float(epoch_f1))

                # deep copy the model
                if phase == 'valid' and (epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss <= best_loss)):
                    best_acc = epoch_acc
                    best_acc_epoch = epoch
                    
                    if save_toggle and save_mode == "best_acc":
                        best_model_wts = copy.deepcopy(self.__model.state_dict())
                        save_name_pt = save_name + "_" + save_mode + ".pt"
                        save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
                        # torch.save(best_model_wts, save_path)
                        save_model(self.__model, save_path)

                if phase == 'valid' and (epoch_loss < best_loss or (epoch_loss == best_loss and epoch_acc >= best_acc)):
                    best_loss = epoch_loss
                    best_loss_epoch = epoch
                    
                    if save_toggle and save_mode == "best_loss":
                        best_model_wts = copy.deepcopy(self.__model.state_dict())
                        save_name_pt = save_name + "_" + save_mode + ".pt"
                        save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
                        # torch.save(best_model_wts, save_path)
                        save_model(self.__model, save_path)
                
                if phase == 'valid' and (epoch_f1 > best_f1 or (epoch_f1 == best_f1 and epoch_loss <= best_loss)):
                    best_f1 = epoch_f1
                    best_f1_epoch = epoch
                    
                    if save_toggle and save_mode == "best_f1":
                        best_model_wts = copy.deepcopy(self.__model.state_dict())
                        save_name_pt = save_name + "_" + save_mode + ".pt"
                        save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
                        # torch.save(best_model_wts, save_path)
                        save_model(self.__model, save_path)
                
                if save_toggle and save_mode == "all":
                    save_name_pt = save_name + "_epoch_{:03d}.pt".format(epoch+1)
                    save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
                    # torch.save(self.model.state_dict(), save_path)
                    save_model(self.__model, save_path)
                
            if schedule == "custom":
                scheduler.step(valid_acc[epoch])
                if (toggle == 0) and (current_lr <= 1e-6):
                    countdown_epoch = epoch
                    toggle = 1
            elif type(schedule) == list:
                scheduler.step()

        time_elapsed = time.time() - since
        print()
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('\tBest val Acc: {:.2f}% (epoch: {:d})'.format(best_acc * 100, best_acc_epoch + 1))
        print('\tBest val Loss: {:.4f} (epoch: {:d})'.format(best_loss, best_loss_epoch + 1))
        print('\tBest val F1 Score: {:.4f} (epoch: {:d})'.format(best_f1, best_f1_epoch + 1))

        # load best model weights
        if save_toggle and (save_mode == "best_acc" or save_mode == "best_loss" or save_mode == "best_f1"):
            print('Save {} model'.format(save_mode))
            print('\treturn: {} epoch'.format(save_mode))
            self.__model.load_state_dict(best_model_wts)
        elif save_toggle and (save_mode == "all"):
            print('Save {} models'.format(save_mode))
            print('\treturn: last epoch')
        print()

        return {
            "train": (train_loss, train_acc, train_f1),
            "valid": (valid_loss, valid_acc, valid_f1),
        }