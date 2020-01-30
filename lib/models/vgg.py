from torch import nn

import torch

class VGG(nn.Module):
    def __init__(self, layers = 16, alpha=1.0, classification=10):
        super().__init__()
        
        if layers == 11:
            loop = [1, 1, 2, 2, 2]
        elif layers == 13:
            loop = [2, 2, 2, 2, 2]
        elif layers == 16:
            loop = [2, 2, 3, 3, 3]
        elif layers == 19:
            loop = [2, 2, 4, 4, 4]
        else:
            print("Not support layers", layers)
            return
        
        self.block_1 = VGG_Block(loop[0], 3, int(64*alpha))
        self.block_2 = VGG_Block(loop[1], int(64*alpha), int(128*alpha))
        self.block_3 = VGG_Block(loop[2], int(128*alpha), int(256*alpha))
        self.block_4 = VGG_Block(loop[3], int(256*alpha), int(512*alpha))
        self.block_5 = VGG_Block(loop[4], int(512*alpha), int(512*alpha))
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(512*alpha)*1*1, out_features=int(512*alpha)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=int(512*alpha), out_features=int(512*alpha)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=int(512*alpha), out_features=classification)
        )
    
    def forward(self, x):
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class VGG_Block(nn.Sequential):
    def __init__(self, layers, in_channels, out_channels):
        super(VGG_Block, self).__init__()

        self.add_module("Conv_1", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
        self.add_module("Conv_1_ReLU", nn.ReLU(inplace=True))

        for i in range(layers - 1):
            self.add_module("Conv_%d" % (i + 2), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))
            self.add_module("Conv_%d_ReLU" % (i + 2), nn.ReLU(inplace=True))
        
        self.add_module("MaxPooling", nn.MaxPool2d(kernel_size=2))
    
class fuckingVGG16(nn.Module):
    def __init__(self, classification=10):
        super().__init__()
        
        self.block_1 = fuckingBlock(2, 3, 64)
        self.block_2 = fuckingBlock(2, 64, 128)
        self.block_3 = fuckingBlock(3, 128, 256)
        self.block_4 = fuckingBlock(3, 256, 512)
        self.block_5 = fuckingBlock(3, 512, 512)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1*1*512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=classification)
        )
    
    def forward(self, x):
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
class fuckingBlock(nn.Module):
    def __init__(self, layers, in_channels, out_channels):
        super().__init__()
        
        self.layers = layers - 1
        
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.conv_in(x)
        x = self.relu(x)
        
        for i in range(self.layers):
            x = self.conv_out(x)
            x = self.relu(x)
            
        out = self.maxpool(x)
        
        return out
    
class VGG_multi(nn.Module):
    def __init__(self, layers = 16, alpha=1.0):
        super().__init__()
        
        if layers == 11:
            loop = [1, 1, 2, 2, 2]
        elif layers == 13:
            loop = [2, 2, 2, 2, 2]
        elif layers == 16:
            loop = [2, 2, 3, 3, 3]
        elif layers == 19:
            loop = [2, 2, 4, 4, 4]
        else:
            print("Not support layers", layers)
            return
        
        self.block_1 = VGG_Block(loop[0], 3, int(64*alpha))
        self.block_2 = VGG_Block(loop[1], int(64*alpha), int(128*alpha))
        self.block_3 = VGG_Block(loop[2], int(128*alpha), int(256*alpha))
        self.block_4 = VGG_Block(loop[3], int(256*alpha), int(512*alpha))
        self.block_5 = VGG_Block(loop[4], int(512*alpha), int(512*alpha))
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(512*alpha)*1*1, out_features=int(512*alpha)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=int(512*alpha), out_features=int(512*alpha)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        
        self.output_1 = nn.Linear(in_features=int(512*alpha), out_features=2)
        self.output_2 = nn.Linear(in_features=int(512*alpha), out_features=2)
        self.output_3 = nn.Linear(in_features=int(512*alpha), out_features=2)
        self.output_4 = nn.Linear(in_features=int(512*alpha), out_features=2)
        self.output_5 = nn.Linear(in_features=int(512*alpha), out_features=2)
    
    def forward(self, x):
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        out_1 = self.output_1(out)
        out_2 = self.output_2(out)
        out_3 = self.output_3(out)
        out_4 = self.output_4(out)
        out_5 = self.output_5(out)
        return out_1, out_2, out_3, out_4, out_5