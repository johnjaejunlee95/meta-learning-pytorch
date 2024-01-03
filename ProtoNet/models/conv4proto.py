import torch.nn as nn
import torch.nn.functional as F


def convblock(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Conv4Proto(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.in_channels = args.imgc
        self.out_channels = args.filter_size
        # self.hidden_size = hidden_size
        
        
        self.feature_extractor = nn.Sequential(
            convblock(self.in_channels, self.out_channels),
            convblock(self.out_channels, self.out_channels),
            convblock(self.out_channels, self.out_channels),
            convblock(self.out_channels, self.out_channels)
        )

    def forward(self, x):
        
        feature = self.feature_extractor(x)
        feature = feature.view(feature.size(0), -1)
        
        return feature