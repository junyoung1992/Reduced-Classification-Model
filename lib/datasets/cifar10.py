from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import torchvision, PIL
import numpy as np

def load_cifar10(dataset_type, distribution, partition, batch_size):
    '''
        dataset_type:
            "train" -> return train_loader, valid_loader
            "test" -> return test_loader
        distribution:
            1 -> classify 10 labels
            2 -> classify 5 labels (+ others)
            5 -> classify 2 labels (+ others)
            10 -> classify 1 labels (+ others)
        partition:
            1 -> if distribution == 5:
                     labels = [0, 1, others]
                     return label 0 -> 0
                                  1 -> 1
                                  others -> 2
            2 -> if distribution == 5:
                     labels = [2, 3, others]
                     return label 2 -> 0
                                  3 -> 1
                                  others -> 2
            (...)
        batch_size:
        
    '''
    transform = {
        "train": torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),
        "test": torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),
    }
    
    if dataset_type == "train":
        cifar10_train = torchvision.datasets.CIFAR10("./data", train=True, transform=transform["train"], download=True)
        cifar10_valid = torchvision.datasets.CIFAR10("./data", train=True, transform=transform["test"], download=True)
    
        num_train = len(cifar10_train)
        indices = list(range(num_train))
        split = 40000

        np.random.seed(42)
        np.random.shuffle(indices)
        
        labels = []
        if distribution == 2:
            if partition == 1:
                p = [0, 1, 2, 3, 4]
            elif partition == 2:
                p = [5, 6, 7, 8, 9]
            else:
                print("Partition Error!!")
                return
            
            for target in cifar10_train.targets:
                if target == p[0]:
                    labels += [torch.tensor(0)]
                elif target == p[1]:
                    labels += [torch.tensor(1)]
                elif target == p[2]:
                    labels += [torch.tensor(2)]
                elif target == p[3]:
                    labels += [torch.tensor(3)]
                elif target == p[4]:
                    labels += [torch.tensor(4)]
                else:
                    labels += [torch.tensor(5)]
            cifar10_train.targets = torch.as_tensor(labels)
            
            for target in cifar10_valid.targets:
                if target == p[0]:
                    labels += [torch.tensor(0)]
                elif target == p[1]:
                    labels += [torch.tensor(1)]
                elif target == p[2]:
                    labels += [torch.tensor(2)]
                elif target == p[3]:
                    labels += [torch.tensor(3)]
                elif target == p[4]:
                    labels += [torch.tensor(4)]
                else:
                    labels += [torch.tensor(5)]
            cifar10_valid.targets = torch.as_tensor(labels)

        elif distribution == 5:
            if partition == 1:
                p = [0, 1]
            elif partition == 2:
                p = [2, 3]
            elif partition == 3:
                p = [4, 5]
            elif partition == 4:
                p = [6, 7]
            elif partition == 5:
                p = [8, 9]
            else:
                print("Partition Error!!")
                return
            
            for target in cifar10_train.targets:
                if target == p[0]:
                    labels += [torch.tensor(0)]
                elif target == p[1]:
                    labels += [torch.tensor(1)]
                else:
                    labels += [torch.tensor(2)]
            cifar10_train.targets = torch.as_tensor(labels)
            
            for target in cifar10_valid.targets:
                if target == p[0]:
                    labels += [torch.tensor(0)]
                elif target == p[1]:
                    labels += [torch.tensor(1)]
                else:
                    labels += [torch.tensor(2)]
            cifar10_valid.targets = torch.as_tensor(labels)

        elif distribution == 10:
            if (0 < partition < 11) == False:
                print("Partition Error!!")
                return
            
            for target in cifar10_train.targets:
                if target == (partition - 1):
                    labels += [torch.tensor(0)]
                else:
                    labels += [torch.tensor(1)]
            cifar10_train.targets = torch.as_tensor(labels)
            
            for target in cifar10_valid.targets:
                if target == (partition - 1):
                    labels += [torch.tensor(0)]
                else:
                    labels += [torch.tensor(1)]
            cifar10_valid.targets = torch.as_tensor(labels)

        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(cifar10_valid, batch_size=batch_size, sampler=valid_sampler)
        
        return train_loader, valid_loader
    
    elif dataset_type == "test":
        cifar10_test = torchvision.datasets.CIFAR10("./data", train=False, transform=transform["test"], download=True)
        
        labels = []
        if distribution == 2:
            if partition == 1:
                p = [0, 1, 2, 3, 4]
            elif partition == 2:
                p = [5, 6, 7, 8, 9]
            else:
                print("Partition Error!!")
                return
            
            for target in cifar10_train.targets:
                if target == p[0]:
                    labels += [torch.tensor(0)]
                elif target == p[1]:
                    labels += [torch.tensor(1)]
                elif target == p[2]:
                    labels += [torch.tensor(2)]
                elif target == p[3]:
                    labels += [torch.tensor(3)]
                elif target == p[4]:
                    labels += [torch.tensor(4)]
                else:
                    labels += [torch.tensor(5)]
            cifar10_test.targets = torch.as_tensor(labels)

        elif distribution == 5:
            if partition == 1:
                p = [0, 1]
            elif partition == 2:
                p = [2, 3]
            elif partition == 3:
                p = [4, 5]
            elif partition == 4:
                p = [6, 7]
            elif partition == 5:
                p = [8, 9]
            else:
                print("Partition Error!!")
                return
            
            for target in cifar10_train.targets:
                if target == p[0]:
                    labels += [torch.tensor(0)]
                elif target == p[1]:
                    labels += [torch.tensor(1)]
                else:
                    labels += [torch.tensor(2)]
            cifar10_test.targets = torch.as_tensor(labels)

        elif distribution == 10:
            if (0 < partition < 11) == False:
                print("Partition Error!!")
                return
            
            for target in cifar10_train.targets:
                if target == (partition - 1):
                    labels += [torch.tensor(0)]
                else:
                    labels += [torch.tensor(1)]
            cifar10_test.targets = torch.as_tensor(labels)
        
        test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)
        
        return test_loader
