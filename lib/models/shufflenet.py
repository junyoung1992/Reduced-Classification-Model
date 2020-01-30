from torch import nn

import torch

class ShuffleNet(nn.Module):
    def __init__(self, classification=10, groups=1, alpha=1.0):
        super().__init__()
        
        if groups == 1:
            self.channels = [24, 144, 288, 576]
        elif groups == 2:
            self.channels = [24, 200, 400, 800]
        elif groups == 3:
            self.channels = [24, 240, 480, 960]
        elif groups == 4:
            self.channels = [24, 272, 544, 1088]
        elif groups == 8:
            self.channels = [24, 384, 768, 1536]
        
        # Conv_1: input(3, 32, 32) / output(24, 16, 16)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ReLU(inplace=True)
        )
        # Max Pool: input(24, 16, 16) / output(24, 8, 8)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage_2 = nn.Sequential()
        self.stage_3 = nn.Sequential()
        self.stage_4 = nn.Sequential()
        
        # Stage_2: input(24, 8, 8) / output([1], 4, 4)
        self.stage_2.add_module("Shuffle_Unit_1", ShuffleUnit(in_channels=int(self.channels[0]*alpha), out_channels=int(self.channels[1]*alpha), groups=groups, grouped_conv=False, combine="concat"))
        for i in range(3):
            unit_name = "Shuffle_Unit_" + str(i + 2)
            self.stage_2.add_module(unit_name, ShuffleUnit(in_channels=int(self.channels[1]*alpha), out_channels=int(self.channels[1]*alpha), groups=groups, grouped_conv=False, combine="add"))
        
        # Stage_3: input([1], 4, 4) / output([2], 2, 2)
        self.stage_3.add_module("Shuffle_Unit_1", ShuffleUnit(in_channels=int(self.channels[1]*alpha), out_channels=int(self.channels[2]*alpha), groups=groups, grouped_conv=True, combine="concat"))
        for i in range(7):
            unit_name = "Shuffle_Unit_" + str(i + 2)
            self.stage_3.add_module(unit_name, ShuffleUnit(in_channels=int(self.channels[2]*alpha), out_channels=int(self.channels[2]*alpha), groups=groups, grouped_conv=True, combine="add"))
        
        # Stage_4: input([2], 2, 2) / output([3], 1, 1)
        self.stage_4.add_module("Shuffle_Unit_1", ShuffleUnit(in_channels=int(self.channels[2]*alpha), out_channels=int(self.channels[3]*alpha), groups=groups, grouped_conv=True, combine="concat"))
        for i in range(3):
            unit_name = "Shuffle_Unit_" + str(i + 2)
            self.stage_4.add_module(unit_name, ShuffleUnit(in_channels=int(self.channels[3]*alpha), out_channels=int(self.channels[3]*alpha), groups=groups, grouped_conv=True, combine="add"))
        
        # Classifier: input([3], 1, 1) / output(10)
        # self.avgPool = nn.AvgPool2d(kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(self.channels[3] * alpha), out_features=classification)
        )
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.maxPool(out)
        
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.stage_4(out)
        
        # out = self.avgPool(out)
        # out = torch.flatten(out, 1)
        out = out.mean([2, 3])
        out = self.classifier(out)
        return out

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups, grouped_conv, combine):
        super().__init__()
        self.combine = combine
        self.groups = groups
        
        compress_groups = self.groups if grouped_conv else 1
        bottleneck_channels = out_channels // 4
        
        if self.combine == "concat":
            self.avgPool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 2
            self.combine_func = self.concat
            out_channels -= in_channels
        elif self.combine == "add":
            stride = 1
            self.combine_func = self.add
        
        self.conv_PW = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, groups=compress_groups)
        self.conv_PW_BN = nn.BatchNorm2d(num_features=bottleneck_channels)
        self.conv_PW_ReLU = nn.ReLU(inplace=True)
        self.conv_DW = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=3, stride=stride, padding=1, groups=bottleneck_channels)
        self.conv_DW_BN = nn.BatchNorm2d(num_features=bottleneck_channels)
        self.conv_PW_Linear = nn.Conv2d(in_channels=bottleneck_channels, out_channels=out_channels, kernel_size=1, groups=self.groups)
        self.conv_PW_Linear_BN = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        
        channels_per_group = num_channels // groups
        out = x.view(batchsize, groups, channels_per_group, height, width)
        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        out = torch.transpose(out, 1, 2).contiguous()
        out = out.view(batchsize, -1, height, width)
        return out
        
    def add(self, x, out):
        return x + out
        
    def concat(self, x, out):
        return torch.cat((x, out), 1)
        
    def forward(self, x):
        identity = x
        
        if self.combine == "concat":
            identity = self.avgPool(identity)
        
        out = self.conv_PW(x)
        out = self.conv_PW_BN(out)
        out = self.conv_PW_ReLU(out)
        out = self.channel_shuffle(out, self.groups)
        out = self.conv_DW(out)
        out = self.conv_DW_BN(out)
        out = self.conv_PW_Linear(out)
        out = self.conv_PW_Linear_BN(out)
        
        out = self.combine_func(identity, out)
        out = self.relu(out)
        return out