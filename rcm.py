from lib import models
from lib.datasets.mnist import load_mnist
from lib.datasets.cifar10 import load_cifar10
from lib.training import fit, evaluate, distributed_evaluate
from lib.compress.filterpruning import Pruning as fp
from lib.compress.filterpruning import iterative_pruning as ifp
from lib.compress.filterpruning_revise import Pruning

from torch import optim, nn

import argparse
import csv
import os
import numpy as np
import torch

def train_baseline(option):
    _save_name = option["save_name"]
    if option["model"] == "VGG16":
        model = models.VGG(layers=16)
    elif option["model"] == "MobileNet":
        model = models.MobileNet(alpha=option["alpha"])
        if option["alpha"] == 1.0:
            _save_name = _save_name + "x1.0"
        else:
            _save_name = _save_name + "x{:n}".format(option["alpha"])
    elif option["model"] == "LeNet5":
        model = models.LeNet5()
    save_name = _save_name + "_baseline"
    
    if option["model"] == "LeNet5":
        train_dl = load_mnist("train", 1, 1, option["batch"])    # (train_dl, valid_dl)
        test_dl = load_mnist("test", 1, 1, option["batch"])
    else:
        train_dl = load_cifar10("train", 1, 1, option["batch"])
        test_dl = load_cifar10("test", 1, 1, option["batch"])

    model = model.to(option["dev"])
    
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=option["lr"])

    history = fit(model, train_dl[0], train_dl[1], loss_fn, opt, option["epoch"], option["schedule"], save_name, option["save_option"])
    result = evaluate(model, test_dl)

def train_rcm(option):
    _save_name = option["save_name"]
    if option["model"] == "VGG16":
        model = [models.VGG(layers=16, classification=10) for _ in range(option["rcm"])]
    elif option["model"] == "MobileNet":
        model = [models.MobileNet(alpha=option["alpha"], classification=10) for _ in range(option["rcm"])]
        if option["alpha"] == 1.0:
            _save_name = _save_name + "x1.0"
        else:
            _save_name = _save_name + "x{:n}".format(option["alpha"])
    elif option["model"] == "LeNet5":
        model = [models.LeNet5(classification=10) for _ in range(option["rcm"])]
    _save_name = _save_name + "_rcm{:d}".format(option["rcm"])

    if option["model"] == "LeNet5":
        train_dl = [load_mnist("train", option["rcm"], i + 1, option["batch"]) for i in range(option["rcm"])]   # ([train_dl[0], ...], [valid_dl[0], ...])
        test_dl = load_mnist("test", 1, 1, option["batch"])
    else:
        train_dl = [load_cifar10("train", option["rcm"], i + 1, option["batch"]) for i in range(option["rcm"])]
        test_dl = load_cifar10("test", 1, 1, option["batch"])

    load_model = torch.load(option["model_path"][0], map_location=option["dev"])

    for i in range(option["rcm"]):
        print("Reduced Classification model #{:d}".format(i + 1))

        classification = int(10 / option["rcm"]) + 1

        model[i].load_state_dict(load_model["state_dict"])
        last_in_features = model[i].classifier[-1].in_features
        model[i].classifier[-1] = nn.Linear(in_features=last_in_features, out_features=classification)
        model[i] = model[i].to(option["dev"])

        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(model[i].parameters(), lr=option["lr"])

        save_name = _save_name + "_{:d}".format(i + 1)
        history = fit(model[i], train_dl[i][0], train_dl[i][1], loss_fn, opt, option["epoch"], option["schedule"], save_name, option["save_option"])
    
    result = distributed_evaluate(model, test_dl)

def train(args):
    option = dict(
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        model = args.model,
        alpha = args.alpha,
        rcm = args.rcm,
        model_path = args.model_path,
        lr = args.lr,
        epoch = args.epoch,
        schedule = args.schedule,
        batch = args.batch,
        save_name = args.model,
        save_option = "best_acc"
    )

    if option["rcm"] == 1:   # Baseline
        train_baseline(option)
    else:   # Reduced Classification Model
        train_rcm(option)

def write_pruning_result(model, test_dl, rcm, sequence, save_name):
    if rcm == 1:
        result = evaluate(model, test_dl)
    else:
        result = distributed_evaluate(model, test_dl)

    if save_name is not None:
        if not(os.path.isdir("analysis")):
            os.makedirs(os.path.join("analysis"))

    with open('./analysis/{}_eval.csv'.format(save_name), 'a', newline='') as csvfile:
        acc_file = csv.writer(csvfile)
        acc_file.writerow([sequence, float(result["acc"]), float(result["f1score"])])

def prune_baseline(option):
    _save_name = option["save_name"]
    if option["model"] == "VGG16":
        model = models.VGG(layers=16)
    elif option["model"] == "MobileNet":
        model = models.MobileNet(alpha=option["alpha"])
        _save_name = _save_name + "x{:n}".format(option["alpha"])
    elif option["model"] == "LeNet5":
        model = models.LeNet5()
    _save_name = _save_name + "_fp_baseline"

    if option["model"] == "LeNet5":
        train_dl = load_mnist("train", 1, 1, option["batch"])    # (train_dl, valid_dl)
        test_dl = load_mnist("test", 1, 1, option["batch"])
    else:
        train_dl = load_cifar10("train", 1, 1, option["batch"])
        test_dl = load_cifar10("test", 1, 1, option["batch"])
    
    load_model = torch.load(option["model_path"][0], map_location=option["dev"])
    model._modules = load_model['_modules']
    model.load_state_dict(load_model['state_dict'])
    model = model.to(option["dev"])

    csv_name = _save_name
    write_pruning_result(model, test_dl, option["rcm"], 0, csv_name)

    for i in range(option["prune_step"]):
        print("Filter Pruning #{:d}".format(i + 1))

        save_name = _save_name + "_{:03d}".format(i + 1)
        pruning = Pruning(model, train_dl[0], train_dl[1])
        history = pruning.iterative_pruning(one_epoch_remove=option["filters_removed"], finetuning=option["retraining"],
                                            epoch=option["epoch"], lr=option["lr"], save_name=save_name, save_mode=option["save_option"])

        if option["retraining"]:
            if save_name is not None:
                if not(os.path.isdir("analysis")):
                    os.makedirs(os.path.join("analysis"))

            with open('./analysis/{}_train_loss.csv'.format(csv_name), 'a', newline='') as csvfile:
                training_file = csv.writer(csvfile)
                training_file.writerows([history['train'][0]])
            with open('./analysis/{}_train_acc.csv'.format(csv_name), 'a', newline='') as csvfile:
                training_file = csv.writer(csvfile)
                training_file.writerows([history['train'][1]])
            with open('./analysis/{}_valid_loss.csv'.format(csv_name), 'a', newline='') as csvfile:
                validation_file = csv.writer(csvfile)
                validation_file.writerows([history['valid'][0]])
            with open('./analysis/{}_valid_acc.csv'.format(csv_name), 'a', newline='') as csvfile:
                validation_file = csv.writer(csvfile)
                validation_file.writerows([history['valid'][1]])

        write_pruning_result(model, test_dl, option["rcm"], i + 1, csv_name)

def prune_rcm(option):
    _save_name = option["save_name"]
    classification = int(10 / option["rcm"]) + 1
    if option["model"] == "VGG16":
        model = [models.VGG(layers=16, classification=classification) for _ in range(option["rcm"])]
    elif option["model"] == "MobileNet":
        model = [models.MobileNet(alpha=option["alpha"], classification=classification) for _ in range(option["rcm"])]
        if option["alpha"] == 1.0:
            _save_name = _save_name + "x1.0"
        else:
            _save_name = _save_name + "x{:n}".format(option["alpha"])
    elif option["model"] == "LeNet5":
        model = [models.LeNet5(classification=classification) for _ in range(option["rcm"])]
    _save_name = _save_name + "_fp_rcm{:d}".format(option["rcm"])

    i = 0
    for m in model:
        load_model = torch.load(option["model_path"][i], map_location=option["dev"])
        m._modules = load_model['_modules']
        m.load_state_dict(load_model['state_dict'])
        m = m.to(option["dev"])
        i = i + 1
    
    if option["model"] == "LeNet5":
        train_dl = [load_mnist("train", option["rcm"], i + 1, option["batch"]) for i in range(option["rcm"])]   # ([train_dl[0], ...], [valid_dl[0], ...])
        test_dl = load_mnist("test", 1, 1, option["batch"])
    else:
        train_dl = [load_cifar10("train", option["rcm"], i + 1, option["batch"]) for i in range(option["rcm"])]
        test_dl = load_cifar10("test", 1, 1, option["batch"])

    csv_name = _save_name
    write_pruning_result(model, test_dl, option["rcm"], 0, csv_name)

    for i in range(option["prune_step"]):
        print("Filter Pruning #{:d}".format(i + 1))
        for j in range(option["rcm"]):
            print("Reduced Classification model #{:d}".format(j + 1))
            save_name = _save_name + "_{:03d}_{:d}".format(i + 1, j + 1)

            pruning = Pruning(model[j], train_dl[j][0], train_dl[j][1])
            history = pruning.iterative_pruning(one_epoch_remove=option["filters_removed"], finetuning=option["retraining"],
                                                epoch=option["epoch"], lr=option["lr"], save_name=save_name, save_mode=option["save_option"])

            if option["retraining"]:
                if save_name is not None:
                    if not(os.path.isdir("analysis")):
                        os.makedirs(os.path.join("analysis"))

                with open('./analysis/{}_{:d}_train_loss.csv'.format(csv_name, j + 1), 'a', newline='') as csvfile:
                    training_file = csv.writer(csvfile)
                    training_file.writerows([history['train'][0]])
                with open('./analysis/{}_{:d}_train_acc.csv'.format(csv_name, j + 1), 'a', newline='') as csvfile:
                    training_file = csv.writer(csvfile)
                    training_file.writerows([history['train'][1]])
                with open('./analysis/{}_{:d}_valid_loss.csv'.format(csv_name, j + 1), 'a', newline='') as csvfile:
                    validation_file = csv.writer(csvfile)
                    validation_file.writerows([history['valid'][0]])
                with open('./analysis/{}_{:d}_valid_acc.csv'.format(csv_name, j + 1), 'a', newline='') as csvfile:
                    validation_file = csv.writer(csvfile)
                    validation_file.writerows([history['valid'][1]])

        write_pruning_result(model, test_dl, option["rcm"], i + 1, csv_name)

def prune(args):
    option = dict(
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        model = args.model,
        alpha = args.alpha,
        rcm = args.rcm,
        model_path = args.model_path,
        lr = args.lr,
        epoch = args.epoch,
        schedule = args.schedule,
        batch = args.batch,
        save_name = args.model,
        save_option = "best_acc",
        filters_removed = args.filters_removed,
        retraining = args.retraining,
        prune_step = args.prune_step
    )

    if option["rcm"] == 1:
        prune_baseline(option)
    else:
        prune_rcm(option)

def finetuning_baseline(option):
    _save_name = option["save_name"]
    if option["model"] == "VGG16":
        model = models.VGG(layers=16)
    elif option["model"] == "MobileNet":
        model = models.MobileNet(alpha=option["alpha"])
        if option["alpha"] == 1.0:
            _save_name = _save_name + "x1.0"
        else:
            _save_name = _save_name + "x{:n}".format(option["alpha"])
    elif option["model"] == "LeNet5":
        model = models.LeNet5()
    save_name = _save_name + "_ft_baseline"
    
    if option["model"] == "LeNet5":
        train_dl = load_mnist("train", 1, 1, option["batch"])    # (train_dl, valid_dl)
        test_dl = load_mnist("test", 1, 1, option["batch"])
    else:
        train_dl = load_cifar10("train", 1, 1, option["batch"])
        test_dl = load_cifar10("test", 1, 1, option["batch"])

    load_model = torch.load(option["model_path"][0], map_location=option["dev"])
    model._modules = load_model['_modules']
    model.load_state_dict(load_model['state_dict'])
    model = model.to(option["dev"])
    
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=option["lr"])

    history = fit(model, train_dl[0], train_dl[1], loss_fn, opt, option["epoch"], option["schedule"], save_name, option["save_option"])
    result = evaluate(model, test_dl)

def finetuning_rcm(option):
    _save_name = option["save_name"]
    if option["model"] == "VGG16":
        model = [models.VGG(layers=16, classification=10) for _ in range(option["rcm"])]
    elif option["model"] == "MobileNet":
        model = [models.MobileNet(alpha=option["alpha"], classification=10) for _ in range(option["rcm"])]
        if option["alpha"] == 1.0:
            _save_name = _save_name + "x1.0"
        else:
            _save_name = _save_name + "x{:n}".format(option["alpha"])
    elif option["model"] == "LeNet5":
        model = [models.LeNet5(classification=10) for _ in range(option["rcm"])]
    _save_name = _save_name + "_ft_rcm{:d}".format(option["rcm"])

    if option["model"] == "LeNet5":
        train_dl = [load_mnist("train", option["rcm"], i + 1, option["batch"]) for i in range(option["rcm"])]   # ([train_dl[0], ...], [valid_dl[0], ...])
        test_dl = load_mnist("test", 1, 1, option["batch"])
    else:
        train_dl = [load_cifar10("train", option["rcm"], i + 1, option["batch"]) for i in range(option["rcm"])]
        test_dl = load_cifar10("test", 1, 1, option["batch"])

    i = 0
    for m in model:
        load_model = torch.load(option["model_path"][i], map_location=option["dev"])
        m._modules = load_model['_modules']
        m.load_state_dict(load_model['state_dict'])
        m = m.to(option["dev"])
        i = i + 1

    for i in range(option["rcm"]):
        print("Reduced Classification model #{:d}".format(i + 1))

        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(model[i].parameters(), lr=option["lr"])

        save_name = _save_name + "_{:d}".format(i + 1)
        history = fit(model[i], train_dl[i][0], train_dl[i][1], loss_fn, opt, option["epoch"], option["schedule"], save_name, option["save_option"])
    
    result = distributed_evaluate(model, test_dl)

def finetuning(args):
    option = dict(
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        model = args.model,
        alpha = args.alpha,
        rcm = args.rcm,
        model_path = args.model_path,
        lr = args.lr,
        epoch = args.epoch,
        schedule = args.schedule,
        batch = args.batch,
        save_name = args.model,
        save_option = "best_acc"
    )

    if option["rcm"] == 1:   # Baseline
        finetuning_baseline(option)
    else:   # Reduced Classification Model
        finetuning_rcm(option)

def evaluation_baseline(option):
    _save_name = option["save_name"]
    if option["model"] == "VGG16":
        model = models.VGG(layers=16)
    elif option["model"] == "MobileNet":
        model = models.MobileNet(alpha=option["alpha"])
    elif option["model"] == "LeNet5":
        model = models.LeNet5()
    
    if option["model"] == "LeNet5":
        test_dl = load_mnist("test", 1, 1, option["batch"])
    else:
        test_dl = load_cifar10("test", 1, 1, option["batch"])
    
    load_model = torch.load(option["model_path"][0], map_location=option["dev"])
    model._modules = load_model['_modules']
    model.load_state_dict(load_model['state_dict'])
    model = model.to(option["dev"])

    result = evaluate(model, test_dl)

def evaluation_rcm(option):
    if option["model"] == "VGG16":
        model = [models.VGG(layers=16, classification=10) for _ in range(option["rcm"])]
    elif option["model"] == "MobileNet":
        model = [models.MobileNet(alpha=option["alpha"], classification=10) for _ in range(option["rcm"])]
    elif option["model"] == "LeNet5":
        model = [models.LeNet5(classification=10) for _ in range(option["rcm"])]

    if option["model"] == "LeNet5":
        test_dl = load_mnist("test", 1, 1, option["batch"])
    else:
        test_dl = load_cifar10("test", 1, 1, option["batch"])

    i = 0
    for m in model:
        load_model = torch.load(option["model_path"][i], map_location=option["dev"])
        m._modules = load_model['_modules']
        m.load_state_dict(load_model['state_dict'])
        m = m.to(option["dev"])
        i = i + 1
    
    result = distributed_evaluate(model, test_dl)

def evaluation(args):
    option = dict(
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        model = args.model,
        alpha = args.alpha,
        rcm = args.rcm,
        model_path = args.model_path,
        batch = args.batch,
    )

    if option["rcm"] == 1:   # Baseline
        evaluation_baseline(option)
    else:   # Reduced Classification Model
        evaluation_rcm(option)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--finetuning", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--rcm", type=int, choices=[1, 2, 5, 10], default="1")
    parser.add_argument("--model_path", type=str, nargs="*")
    parser.add_argument("--model", choices=['LeNet5', 'VGG16', 'MobileNet'])
    parser.add_argument("--alpha", type=float, choices=[1.0, 0.75, 0.5, 0.25], default="1.0")
    parser.add_argument("--batch", type=int, default="32")
    parser.add_argument("--epoch", type=int, default="300")
    parser.add_argument("--lr", type=float, default="1e-2")
    parser.add_argument("--schedule", type=int, nargs="*")
    parser.add_argument("--filters_removed", type=int)   # during one epoch
    parser.add_argument("--retraining", action="store_true")
    parser.add_argument("--prune_step", type=int, default="10")

    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(finetuning=False)
    parser.set_defaults(eval=False)
    parser.set_defaults(retraining=False)

    args = parser.parse_args()

    return args

def error_check_args(args):
    if args.rcm != 1 and args.model_path is None:
        print("Model path is required.")
    elif ((args.train == True and args.prune == False and args.finetuning == False and args.eval == False) or \
          (args.train == False and args.prune == True and args.finetuning == False and args.eval == False) or \
          (args.train == False and args.prune == False and args.finetuning == True and args.eval == False) or \
          (args.train == False and args.prune == False and args.finetuning == False and args.eval == True)) is False:
        print("Only one option of train, prune, and finetuning option, must be used.")
    elif args.prune == True and args.filters_removed is None:
        print("To use pruning, filters_removed option must be entered.")
    else:
        return False    # An error is not detected

    return True    # An error is detected

def main():
    args = get_args()

    if error_check_args(args):
        return
    
    if args.train:
        train(args)
    elif args.prune:
        prune(args)
    elif args.finetuning:
        finetuning(args)
    elif args.eval:
        evaluation(args)

if __name__ == "__main__":
    main()
