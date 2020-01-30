from lib import models
from lib.datasets.cifar10 import load_cifar10
from lib.training import fit, evaluate, distributed_evaluate
from lib.compress.filterpruning import Pruning as fp
from lib.compress.filterpruning import iterative_pruning as ifp

from torch import optim, nn

import csv
import numpy as np
import argparse
import torch

def train_baseline(option):
    if option["model"] == "VGG16":
        model = models.VGG(layers=16)
    elif option["model"] == "VGG11":
        model = models.VGG(layers=11)
    elif option["model"] == "MobileNet":
        model = models.MobileNet(alpha=option["alpha"])
        save_name = option["save_name"] + "x{:n}".format(option["alpha"])
    save_name = save_name + "_baseline"
    
    train_dl = load_cifar10("train", 1, 1, option["batch"]) # (train_dl, valid_dl)
    test_dl = load_cifar10("test", 1, 1, option["batch"])

    model = model.to(option["dev"])
    
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=option["lr"])

    history = fit(model, train_dl[0], train_dl[1], loss_fn, opt, option["epoch"], option["schedule"], save_name, option["save_option"])
    result = evaluate(model, test_dl)

def train_rcm(option):
    classification = 10 / option["rcm"]
    if option["model"] == "VGG16":
        model = [models.VGG(layers=16, classification=classification) for _ in range(option["rcm"])]
    elif option["model"] == "VGG11":
        model = [models.VGG(layers=11, classification=classification) for _ in range(option["rcm"])]
    elif option["model"] == "MobileNet":
        model = [models.MobileNet(alpha=option["alpha"], classification=classification) for _ in range(option["rcm"])]
        save_name = option["save_name"] + "x{:n}".format(option["alpha"])
    save_name = save_name + "_rcm_{:d}".format(option["rcm"])

    train_dl = [load_cifar10("train", option["rcm"], i + 1, option["batch"]) for i in range(option["rcm"])] # ([train_dl[0], ...], [valid_dl[0], ...])
    test_dl = load_cifar10("test", 1, 1, 32)

    load_model = torch.load(option["model_path"], map_location=option["dev"])

    for i in range(option["rcm"]):
        model[i].load_state_dict(load_model["state_dict"])
        last_in_features = model[i].classifier[-1].in_features
        model[i].classifier[-1] = nn.Linear(in_features=last_in_features, out_features=classification)
        model[i] = model[i].to(option["dev"])

        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(model[i].parameters(), lr=option["lr"])

        save_name = save_name + "_{:d}".format(i + 1)
        history = fit(model[i], train_dl[0][i], train_dl[1][i], loss_fn, opt, option["epoch"], option["schedule"], save_name, option["save_option"])
    
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

def prune_vgg_baseline(option):
    if option["model"] == "VGG16":
        model = models.VGG(layers=16)
    elif option["model"] == "VGG11":
        model = models.VGG(layers=11)
    save_name = save_name + "_fp_baseline"

    train_dl = load_cifar10("train", 1, 1, 32)  # (train_dl, valid_dl)
    test_dl = load_cifar10("test", 1, 1, 32)
    
    load_model = torch.load(option["model_path"], map_location=option["dev"])
    model._modules = load_model['_modules']
    model.load_state_dict(load_model['state_dict'])
    model = model.to(option["dev"])

    csv_name = save_name
    write_pruning_result(model, test_dl, option["rcm"], 0, csv_name)

    compressor = fp(model)

    for i in range(option["prune_step"]):
        save_name = save_name + "_{:03d}".format(i + 1)

        history = ifp(Pruning=compressor, model=model, one_epoch_remove=option["filters_removed"], finetuning=option["finetuning"],
                      train_dl=train_dl[0], valid_dl=train_dl[1], epoch=option["epoch"], lr=option["lr"],
                      save_name=save_name, save_mode=option["save_option"])

        if option["finetuning"]:
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

def prune_vgg_rcm(option):
    classification = 10 / option["rcm"]
    if option["model"] == "VGG16":
        model = [models.VGG(layer=16, classification=classification) for _ in range(option["rcm"])]
    elif option["model"] == "VGG11":
        model = [models.VGG(layer=11, classification=classification) for _ in range(option["rcm"])]
    save_name = option["save_name"] + "_fp_rcm_{:d}".format(option["rcm"])

    i = 0
    for m in model:
        load_model = torch.load(option["model_path"][i], map_location=option["dev"])
        m._modules = load_model['_modules']
        m.load_state_dict(load_model['state_dict'])
        m = m.to(option["dev"])
        i = i + 1
    
    train_dl = [load_cifar10("train", option["rcm"], i + 1, option["batch"]) for i in range(option["rcm"])] # ([train_dl[0], ...], [valid_dl[0], ...])
    test_dl = load_cifar10("test", 1, 1, 32)

    csv_name = save_name
    write_pruning_result(model, test_dl, option["rcm"], 0, csv_name)

    compressor = [fp(m) for m in model]

    for i in range(option["prune_step"]):
        print("Filter Pruning #{:d}".format(i + 1))
        for j in range(option["rcm"]):
            print("Reduced Classification model #{:d}".format(j + 1))
            save_name = save_name + "_{:03d}_{:d}".format(i + 1, j + 1)

            history = ifp(Pruning=compressor[j], model=model[j], one_epoch_remove=option["filters_removed"], finetuning=option["finetuning"],
                          train_dl=train_dl[j][0], valid_dl=train_dl[j][1], epoch=option["epoch"], lr=option["lr"],
                          save_name=save_name, save_mode=option["save_option"])

            if option["finetuning"]:
                if save_name is not None:
                    if not(os.path.isdir("analysis")):
                        os.makedirs(os.path.join("analysis"))

                with open('./analysis/{}_{:d}_train_loss.csv'.format(csv_name, i + 1), 'a', newline='') as csvfile:
                    training_file = csv.writer(csvfile)
                    training_file.writerows([history['train'][0]])
                with open('./analysis/{}_{:d}_train_acc.csv'.format(csv_name, i + 1), 'a', newline='') as csvfile:
                    training_file = csv.writer(csvfile)
                    training_file.writerows([history['train'][1]])
                with open('./analysis/{}_{:d}_valid_loss.csv'.format(csv_name, i + 1), 'a', newline='') as csvfile:
                    validation_file = csv.writer(csvfile)
                    validation_file.writerows([history['valid'][0]])
                with open('./analysis/{}_{:d}_valid_acc.csv'.format(csv_name, i + 1), 'a', newline='') as csvfile:
                    validation_file = csv.writer(csvfile)
                    validation_file.writerows([history['valid'][1]])

        write_pruning_result(model, test_dl, option["rcm"], i + 1, csv_name)

def prune_mobilenet_baseline(option):
    return

def prune_mobilenet_rcm(option):
    return

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
        finetuning = args.finetuning,
        prune_step = args.prune_step
    )

    if option["rcm"] == 1:   # Baseline
        if option["model"] == "VGG16":
            prune_vgg_baseline(option)
        elif option["model"] == "MobileNet":
            prune_mobilenet_baseline(option)
    else:   # Reduced Classification Model
        if option["model"] == "VGG16":
            prune_vgg_rcm(option)
        elif option["model"] == "MobileNet":
            prune_mobilenet_rcm(option)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--rcm", type=int, choices=[1, 2, 5, 10], default="1")
    parser.add_argument("--model_path", type=str, nargs="*")
    parser.add_argument("--model", choices=['VGG11', 'VGG16', 'MobileNet'])
    parser.add_argument("--alpha", type=float, choices=[1.0, 0.75, 0.5, 0.25], default="1.0")
    parser.add_argument("--batch", type=int, default="32")
    parser.add_argument("--epoch", type=int, default="300")
    parser.add_argument("--lr", type=float, default="1e-2")
    parser.add_argument("--schedule", type=int, nargs="*")
    parser.add_argument("--filters_removed", type=int)   # during one epoch
    parser.add_argument("--finetuning", action="store_true")
    parser.add_argument("--prune_step", type=int, default="10")

    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(finetuning=False)

    args = parser.parse_args()

    return args

def error_check_args(args):
    if args.rcm != 1 and args.model_path is None:
        print("Model path is required.")
    elif args.train == True and args.prune == True:
        print("Train option and prune option cannot be used simultaneously.")
    elif args.train == False and args.prune == False:
        print("Either train option or prune option must be used.")
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

if __name__ == "__main__":
    main()