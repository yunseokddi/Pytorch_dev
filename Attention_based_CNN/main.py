import torch
import torch.nn as nn
import torch.nn.functional as F


class ch_attention(nn.Module):
    def __init__(self, n_ch, r=16):
        super(ch_attention, self).__init__()
        layers = []
        layers += [nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Linear(in_features=n_ch, out_features=n_ch // r, bias=False),
                   nn.BatchNorm1d(n_ch // r),
                   nn.ReLU(True),
                   nn.Linear(in_features=n_ch // r, out_features=n_ch)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).unsqueeze(2).unsqueeze(3).expand_as(x)


class spatial_attention(nn.Module):
    def __init__(self, n_ch, r=16, dilation=4):
        super(spatial_attention, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=n_ch, out_channels=n_ch // r, kernel_size=1, bias=False),
                   nn.BatchNorm2d(n_ch // r),
                   nn.ReLU(True),
                   nn.Conv2d(in_channels=n_ch // r, out_channels=n_ch // r, kernel_size=3, padding=dilation,
                             dilation=dilation, bias=False),
                   nn.BatchNorm2d(n_ch // r),
                   nn.ReLU(True),
                   nn.Conv2d(in_channels=n_ch // r, out_channels=n_ch // r, kernel_size=3, padding=dilation,
                             dilation=dilation, bias=False),
                   nn.BatchNorm2d(n_ch // r),
                   nn.ReLU(True),
                   nn.Conv2d(in_channels=n_ch // r, out_channels=1, kernel_size=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).expand_as(x)
