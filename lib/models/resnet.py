from torch import nn
from torch.nn import functional as f

import torch

class ResNet20(nn.Module):
    def __init__(self, classification=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            Residual_Small(16, 16, downsample=False),
            Residual_Small(16, 16, downsample=False),
            Residual_Small(16, 16, downsample=False)
        )
        self.conv3 = nn.Sequential(
            Residual_Small(16, 32, downsample=True),
            Residual_Small(32, 32, downsample=False),
            Residual_Small(32, 32, downsample=False),
        )
        self.conv4 = nn.Sequential(
            Residual_Small(32, 64, downsample=True),
            Residual_Small(64, 64, downsample=False),
            Residual_Small(64, 64, downsample=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*8*8, out_features=classification)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, layers, classification=10):
        super().__init__()
        
        if layers == 18:
            loop = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            loop = [3, 4, 6, 3]
        elif layers == 101:
            loop = [3, 4, 23, 3]
        elif layers == 152:
            loop = [3, 8, 36, 3]
        else:
            print("Not support layers", layers)
            return

        # Conv_1: input(3, 32, 32) / output(64, 16, 16)
        self.conv_7x7 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_7x7 = nn.BatchNorm2d(num_features=64)
        self.relu_7x7 = nn.ReLU(inplace=True)
        # MaxPool: input(64, 16, 16) / output(64, 8, 8)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                
        self.residual_1 = nn.Sequential()
        self.residual_2 = nn.Sequential()
        self.residual_3 = nn.Sequential()
        self.residual_4 = nn.Sequential()
        
        if layers in [18, 34]:
            # Residual_1: input(64, 8, 8) / output(64, 8, 8)
            self.residual_1.add_module("Block_1", Residual_Small(64, 64))
            for i in range(loop[0] - 1):
                block_name = "Block_" + str(i + 2)
                self.residual_1.add_module(block_name, Residual_Small(64, 64))
            
            # Residual_2: input(64, 8, 8) / output(128, 4, 4)
            self.residual_2.add_module("Block_1", Residual_Small(64, 128, downsample=True))
            for i in range(loop[1] - 1):
                block_name = "Block_" + str(i + 2)
                self.residual_2.add_module(block_name, Residual_Small(128, 128))
            
            # Residual_3: input(128, 4, 4) / output(256, 2, 2)
            self.residual_3.add_module("Block_1", Residual_Small(128, 256, downsample=True))
            for i in range(loop[2] - 1):
                block_name = "Block_" + str(i + 2)
                self.residual_3.add_module(block_name, Residual_Small(256, 256))
            
            # Residual_4: input(256, 2, 2) / output(512, 1, 1)
            self.residual_4.add_module("Block_1", Residual_Small(256, 512, downsample=True))
            for i in range(loop[3] - 1):
                block_name = "Block_" + str(i + 2)
                self.residual_4.add_module(block_name, Residual_Small(512, 512))
            
            out_channels = 512
            
        elif layers in [50, 101, 152]:
            # Residual_1: input(64, 8, 8) / output(256, 8, 8)
            self.residual_1.add_module("Block_1", Residual_Large(64, 256))
            for i in range(loop[0] - 1):
                block_name = "Block_" + str(i + 2)
                self.residual_1.add_module(block_name, Residual_Large(256, 256))
            
            # Residual_2: input(256, 8, 8) / output(512, 4, 4)
            self.residual_2.add_module("Block_1", Residual_Large(256, 512, downsample=True))
            for i in range(loop[1] - 1):
                block_name = "Block_" + str(i + 2)
                self.residual_2.add_module(block_name, Residual_Large(512, 512))
            
            # Residual_3: input(512, 4, 4) / output(1024, 2, 2)
            self.residual_3.add_module("Block_1", Residual_Large(512, 1024, downsample=True))
            for i in range(loop[2] - 1):
                block_name = "Block_" + str(i + 2)
                self.residual_3.add_module(block_name, Residual_Large(1024, 1024))
            
            # Residual_4: input(1024, 2, 2) / output(2048, 1, 1)
            self.residual_4.add_module("Block_1", Residual_Large(1024, 2048, downsample=True))
            for i in range(loop[3] - 1):
                block_name = "Block_" + str(i + 2)
                self.residual_4.add_module(block_name, Residual_Large(2048, 2048))
            
            out_channels = 2048
        
        # Classifier: input((512 or 2048), 1, 1) / output(10)
        ##self.avgPool = nn.AvgPool2d(kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=classification)
        )
        
    def forward(self, x):
        out = self.conv_7x7(x)
        out = self.bn_7x7(out)
        out = self.relu_7x7(out)
        out = self.maxPool(out)
        
        out = self.residual_1(out)
        out = self.residual_2(out)
        out = self.residual_3(out)
        out = self.residual_4(out)
        
        # out = self.avgPool(out)
        # out = torch.flatten(out, 1)
        out = out.mean([2, 3])
        out = self.classifier(out)
        return out

class Residual_Small(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        
        self.downsample = downsample
        if self.downsample == True:
            self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)
            self.projection_bn = nn.BatchNorm2d(num_features=out_channels)
            
            self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu_2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        if self.downsample == True:
            identity = self.projection(identity)
            identity = self.projection_bn(identity)
        out += identity
        out = self.relu_2(out)
        return out

class Residual_Large(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        
        self.downsample = downsample
        self.first = downsample == False and in_channels != out_channels
        bottleneck_channels = out_channels // 4
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_features=bottleneck_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        
        if self.first == True:
            self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
            self.projection_bn = nn.BatchNorm2d(num_features=out_channels)
        
        if self.downsample == True:
            self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)
            self.projection_bn = nn.BatchNorm2d(num_features=out_channels)
            
            self.conv_2 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv_2 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(num_features=bottleneck_channels)
        self.relu_2 = nn.ReLU(inplace=True)
        
        self.conv_3 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(num_features=out_channels)
        self.relu_3 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        out = self.conv_3(out)
        out = self.bn_3(out)
        if self.downsample == True or self.first == True:
            identity = self.projection(identity)
            identity = self.projection_bn(identity)
        out += identity
        out = self.relu_3(out)
        return out