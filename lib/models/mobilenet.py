from torch import nn

import torch

class MobileNet(nn.Module):
    def __init__(self, classification=10, alpha=1.0):
        super().__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=int(32*alpha), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=int(32*alpha)),
            nn.ReLU(inplace=True)
        ) # 32x32 -> 16x16
        
        self.conv_DW_2 = Depthwise_Seperable_Conv(int(32*alpha), int(64*alpha), 1)
        self.conv_DW_3 = Depthwise_Seperable_Conv(int(64*alpha), int(128*alpha), 2) # 16x16 -> 8x8
        self.conv_DW_4 = Depthwise_Seperable_Conv(int(128*alpha), int(128*alpha), 1)
        self.conv_DW_5 = Depthwise_Seperable_Conv(int(128*alpha), int(256*alpha), 2) # 8x8 -> 4x4
        self.conv_DW_6 = Depthwise_Seperable_Conv(int(256*alpha), int(256*alpha), 1)
        self.conv_DW_7 = Depthwise_Seperable_Conv(int(256*alpha), int(512*alpha), 2) # 4x4 -> 2x2
        self.conv_DW_8_1 = Depthwise_Seperable_Conv(int(512*alpha), int(512*alpha), 1)
        self.conv_DW_8_2 = Depthwise_Seperable_Conv(int(512*alpha), int(512*alpha), 1)
        self.conv_DW_8_3 = Depthwise_Seperable_Conv(int(512*alpha), int(512*alpha), 1)
        self.conv_DW_8_4 = Depthwise_Seperable_Conv(int(512*alpha), int(512*alpha), 1)
        self.conv_DW_8_5 = Depthwise_Seperable_Conv(int(512*alpha), int(512*alpha), 1)
        self.conv_DW_9 = Depthwise_Seperable_Conv(int(512*alpha), int(1024*alpha), 2) # 2x2 -> 1x1
        self.conv_DW_10 = Depthwise_Seperable_Conv(int(1024*alpha), int(1024*alpha), 1)
        
        # self.avgPool = nn.AvgPool2d(kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(1024*alpha), out_features=classification)
        )
    
    def forward(self, x):
        out = self.conv_1(x)
        
        out = self.conv_DW_2(out)
        out = self.conv_DW_3(out)
        out = self.conv_DW_4(out)
        out = self.conv_DW_5(out)
        out = self.conv_DW_6(out)
        out = self.conv_DW_7(out)
        out = self.conv_DW_8_1(out)
        out = self.conv_DW_8_2(out)
        out = self.conv_DW_8_3(out)
        out = self.conv_DW_8_4(out)
        out = self.conv_DW_8_5(out)
        out = self.conv_DW_9(out)
        out = self.conv_DW_10(out)
        
        # out = self.avgPool(out)
        # out = torch.flatten(out, 1)
        out = out.mean([2, 3])
        out = self.classifier(out)
        return out
        
class MobileNetV2(nn.Module):
    def __init__(self, classification=10, alpha=1.0):
        super().__init__()
        
        # Conv_1: input(3, 32, 32) / output(32, 16, 16)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU6(inplace=True)
        )
        
        # Bottleneck_2: input(32, 16, 16) / output(16, 16, 16)
        self.bottleneck_2 = InvertedResidual(int(32*alpha), int(16*alpha), 1, 1)
        # Bottleneck_3, 4: input(16, 16, 16) / output(24, 8, 8)
        self.bottleneck_3 = InvertedResidual(int(16*alpha), int(24*alpha), 6, 2)
        self.bottleneck_4 = InvertedResidual(int(24*alpha), int(24*alpha), 6, 1)
        # Bottleneck_5, 6, 7: input(24, 8, 8) / output(32, 4, 4)
        self.bottleneck_5 = InvertedResidual(int(24*alpha), int(32*alpha), 6, 2)
        self.bottleneck_6 = InvertedResidual(int(32*alpha), int(32*alpha), 6, 1)
        self.bottleneck_7 = InvertedResidual(int(32*alpha), int(32*alpha), 6, 1)
        # Bottleneck_8, 9, 10, 11: input(32, 4, 4) / output(64, 2, 2)
        self.bottleneck_8 = InvertedResidual(int(32*alpha), int(64*alpha), 6, 2)
        self.bottleneck_9 = InvertedResidual(int(64*alpha), int(64*alpha), 6, 1)
        self.bottleneck_10 = InvertedResidual(int(64*alpha), int(64*alpha), 6, 1)
        self.bottleneck_11 = InvertedResidual(int(64*alpha), int(64*alpha), 6, 1)
        # Bottleneck_12, 13, 14: input(64, 2, 2) / output(96, 2, 2)
        self.bottleneck_12 = InvertedResidual(int(64*alpha), int(96*alpha), 6, 1)
        self.bottleneck_13 = InvertedResidual(int(96*alpha), int(96*alpha), 6, 1)
        self.bottleneck_14 = InvertedResidual(int(96*alpha), int(96*alpha), 6, 1)
        # Bottleneck_15, 16, 17: input(96, 2, 2) / output(160, 1, 1)
        self.bottleneck_15 = InvertedResidual(int(96*alpha), int(160*alpha), 6, 2)
        self.bottleneck_16 = InvertedResidual(int(160*alpha), int(160*alpha), 6, 1)
        self.bottleneck_17 = InvertedResidual(int(160*alpha), int(160*alpha), 6, 1)
        # Bottleneck_18: input(160, 1, 1) / output(320, 1, 1)
        self.bottleneck_18 = InvertedResidual(int(160*alpha), int(320*alpha), 6, 1)
        
        # Conv_19: input(320, 1, 1) / output(1280, 1, 1)
        self.conv_19 = nn.Sequential(
            nn.Conv2d(in_channels=int(320*alpha), out_channels=int(1280*alpha), kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1280),
            nn.ReLU6(inplace=True)
        )
        
        # Classifier_20: input(1280, 1, 1) / output(10, 1, 1)
        # self.avgPool = nn.AvgPool2d(kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=int(1280*alpha), out_features=classification)
        )
    
    def forward(self, x):
        out = self.conv_1(x)
        
        out = self.bottleneck_2(out)
        out = self.bottleneck_3(out)
        out = self.bottleneck_4(out)
        out = self.bottleneck_5(out)
        out = self.bottleneck_6(out)
        out = self.bottleneck_7(out)
        out = self.bottleneck_8(out)
        out = self.bottleneck_9(out)
        out = self.bottleneck_10(out)
        out = self.bottleneck_11(out)
        out = self.bottleneck_12(out)
        out = self.bottleneck_13(out)
        out = self.bottleneck_14(out)
        out = self.bottleneck_15(out)
        out = self.bottleneck_16(out)
        out = self.bottleneck_17(out)
        out = self.bottleneck_18(out)
        
        out = self.conv_19(out)
        
        # out = self.avgPool(out)
        # out = torch.flatten(out, 1)
        out = out.mean([2, 3])
        out = self.classifier(out)
        return out

class Depthwise_Seperable_Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride):
        super(Depthwise_Seperable_Conv, self).__init__()

        self.add_module("Conv_DW", nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        self.add_module("Conv_DW_BN", nn.BatchNorm2d(num_features=in_channels))
        self.add_module("Conv_DW_ReLU", nn.ReLU(inplace=True))
        self.add_module("Conv_1x1", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False))
        self.add_module("Conv_1x1_BN", nn.BatchNorm2d(num_features=out_channels))
        self.add_module("Conv_1x1_ReLU", nn.ReLU(inplace=True))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(InvertedResidual, self).__init__()
        
        self.shortcut = stride == 1 and in_channels == out_channels
        
        if expansion != 1:
            self.expansion = True
            expansion_channels = in_channels * expansion
            self.conv_pw = nn.Conv2d(in_channels=in_channels, out_channels=expansion_channels, kernel_size=1, bias=False)
            self.conv_pw_bn = nn.BatchNorm2d(num_features=expansion_channels)
            self.conv_pw_relu6 = nn.ReLU6(inplace=True)
        else:
            self.expansion = False
            expansion_channels = in_channels
        
        self.conv_dw = nn.Conv2d(in_channels=expansion_channels, out_channels=expansion_channels, kernel_size=3, stride=stride, padding=1, groups=expansion_channels, bias=False)
        self.conv_dw_bn = nn.BatchNorm2d(num_features=expansion_channels)
        self.conv_dw_relu6 = nn.ReLU6(inplace=True)
        
        self.conv_pw_linear = nn.Conv2d(in_channels=expansion_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_pw_linear_bn = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        identity = x
        
        if self.expansion == True:
            x = self.conv_pw(x)
            x = self.conv_pw_bn(x)
            x = self.conv_pw_relu6(x)
        
        out = self.conv_dw(x)
        out = self.conv_dw_bn(out)
        out = self.conv_dw_relu6(out)
        out = self.conv_pw_linear(out)
        out = self.conv_pw_linear_bn(out)
        
        if self.shortcut == True:
            out += identity
        return out