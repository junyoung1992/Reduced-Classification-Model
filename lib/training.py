from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import copy
import math
import numpy as np
import os
import time
import torch

def save_model(model, path):
    checkpoint = {
        '_modules': model._modules,
        'state_dict': model.state_dict(),
    }
    
    torch.save(checkpoint, path)

def fit(model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, schedule=None, save_name=None, save_mode="all"):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    since = time.time()

    if save_name is not None:
        if not(os.path.isdir("save_models")):
            os.makedirs(os.path.join("save_models"))
                        
        save_toggle = True
        best_model_wts = copy.deepcopy(model.state_dict())
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
                model.train()   # Set model to training mode
                dl = train_dl
                
                for p in optimizer.param_groups:
                    current_lr = round(p['lr'], 7)
                    print("(lr: {:f})".format(current_lr))
            else:
                model.eval()    # Set model to evaluate mode
                dl = valid_dl

            running_loss = 0.0
            running_corrects = 0
            dataset_size = 0
            y_true, y_pred = [], []

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
                    outputs = model(xb)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, yb)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
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
            
            print('{} ({:02.0f}m {:02.0f}s)\tLoss: {:.4f}\tAcc: {:.2f}% ({:d}/{:d})\tF1 score: {:.4f}'\
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
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_name_pt = save_name + "_" + save_mode + ".pt"
                    save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
                    # torch.save(best_model_wts, save_path)
                    save_model(model, save_path)

            if phase == 'valid' and (epoch_loss < best_loss or (epoch_loss == best_loss and epoch_acc >= best_acc)):
                best_loss = epoch_loss
                best_loss_epoch = epoch
                
                if save_toggle and save_mode == "best_loss":
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_name_pt = save_name + "_" + save_mode + ".pt"
                    save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
                    # torch.save(best_model_wts, save_path)
                    save_model(model, save_path)
            
            if phase == 'valid' and (epoch_f1 > best_f1 or (epoch_f1 == best_f1 and epoch_loss <= best_loss)):
                best_f1 = epoch_f1
                best_f1_epoch = epoch
                
                if save_toggle and save_mode == "best_f1":
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_name_pt = save_name + "_" + save_mode + ".pt"
                    save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
                    # torch.save(best_model_wts, save_path)
                    save_model(model, save_path)
            
            if save_toggle and save_mode == "all":
                save_name_pt = save_name + "_epoch_{:03d}.pt".format(epoch+1)
                save_path = os.path.join(os.getcwd(), 'save_models', save_name_pt)
                # torch.save(model.state_dict(), save_path)
                save_model(model, save_path)

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
        model.load_state_dict(best_model_wts)
    elif save_toggle and (save_mode == "all"):
        print('Save {} models'.format(save_mode))
        print('\treturn: last epoch')
    print()

    return {
        "train": (train_loss, train_acc, train_f1),
        "valid": (valid_loss, valid_acc, valid_f1),
    }

def evaluate(model, dl):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    since = time.time()

    corrects = 0
    y_true, y_pred = [], []

    model.eval()

    for xb, yb in dl:
        xb = xb.to(dev)
        yb = yb.to(dev)

        outputs = model(xb)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == yb.data)
        y_true += yb.data.tolist()
        y_pred += preds.tolist()

    time_elapsed = time.time() - since

    y_pred = torch.as_tensor(y_pred)
    acc = (corrects.double() / len(y_true))
    precision = precision_score(y_pred, y_true, average='macro')
    recall = recall_score(y_pred, y_true, average='macro')
    f1score = f1_score(y_pred, y_true, average='macro')

    target_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("\tTest Acc: {:.2f}% ({:d}/{:d})".format(acc * 100, corrects, len(y_true)))
    print("\tTest Precision: {:f} (".format(precision) + ", ".join(map(str, precision_score(y_pred, y_true, average=None))) + ")")
    print("\tTest Recall: {:f} (".format(recall) + ", ".join(map(str, recall_score(y_pred, y_true, average=None))) + ")")
    print("\tTest F1 Score: {:f} (".format(f1score) + ", ".join(map(str, f1_score(y_pred, y_true, average=None))) + ")")
    print()
    print(confusion_matrix(y_pred, y_true))
    print()
    
    return {
        "acc": acc,
        "f1score": f1score
    }

def distributed_evaluate(models, dl):
    def sub_eval(model, dl):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        since = time.time()
        outputs = []

        model.eval()
        softmax = nn.Softmax(dim=1)

        for xb, _ in dl:
            xb = xb.to(dev)
            output = model(xb)
            output = softmax(output)
            outputs += output.tolist()
            
        time_elapsed = time.time() - since
        
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return np.array(outputs)[:, :-1]
    
    sub_output = []
    for model in models:
        sub_output.append(sub_eval(model, dl))

    output = []
    if len(sub_output) == 2:
        output = np.hstack((sub_output[0], sub_output[1]))
    elif len(sub_output) == 5:
        output = np.hstack((sub_output[0], sub_output[1], sub_output[2], sub_output[3], sub_output[4]))
    elif len(sub_output) == 10:
        output = np.hstack((sub_output[0], sub_output[1], sub_output[2], sub_output[3], sub_output[4], \
                            sub_output[5], sub_output[6], sub_output[7], sub_output[8], sub_output[9]))
    output = torch.from_numpy(output)

    corrects = 0
    _, preds = torch.max(output, 1)
    y_true = torch.as_tensor(dl.dataset.targets)
    corrects += torch.sum(preds == y_true)
    acc = (corrects.double() / len(y_true))
    precision = precision_score(preds, y_true, average='macro')
    recall = recall_score(preds, y_true, average='macro')
    f1score = f1_score(preds, y_true, average='macro')

    target_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("\tTest Acc: {:.2f}% ({:d}/{:d})".format(acc * 100, corrects, len(y_true)))
    print("\tTest Precision: {:f} (".format(precision) + ", ".join(map(str, precision_score(preds, y_true, average=None))) + ")")
    print("\tTest Recall: {:f} (".format(recall) + ", ".join(map(str, recall_score(preds, y_true, average=None))) + ")")
    print("\tTest F1 Score: {:f} (".format(f1score) + ", ".join(map(str, f1_score(preds, y_true, average=None))) + ")")
    print()
    print(confusion_matrix(preds, y_true))
    print()

    return {
        "acc": acc,
        "f1score": f1score
    }
