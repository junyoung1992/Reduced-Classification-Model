from torch import nn

import torch

class LeNet5(nn.Module):
    def __init__(self, classification=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, classification),
        )
    
    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class LeNet300100(nn.Module):
    def __init__(self, classification=10):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(28*28*1, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, classification),
        )
    
    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.classifier(out)
        return out