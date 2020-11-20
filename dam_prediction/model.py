import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels, kernel_size=3, strdie=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strdie,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strdie,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_bn_leru_shortcut(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
    )


def conv_bn_leru_3x3_3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down_pooling():
    return nn.MaxPool2d(2)


def up_pooling(inchannels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_pooling2():
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest')
    )


class UNet(nn.Module):
    def __init__(self, input_channels, start_neuron=64):
        super().__init__()

        self.conv1 = conv_bn_relu(input_channels, start_neuron)
        self.conv2 = conv_bn_relu(start_neuron, start_neuron * 2)
        self.conv3 = conv_bn_relu(start_neuron * 2, start_neuron * 4)
        self.conv4 = conv_bn_relu(start_neuron * 4, start_neuron * 8)
        self.conv5 = conv_bn_relu(start_neuron * 8, start_neuron * 16)
        self.down_pooling = nn.MaxPool2d(2)

        self.up_pool6 = up_pooling(start_neuron * 16, start_neuron * 8)
        self.conv6 = conv_bn_relu(start_neuron * 16, start_neuron * 8)
        self.up_pool7 = up_pooling(start_neuron * 8, start_neuron * 4)
        self.conv7 = conv_bn_relu(start_neuron * 8, start_neuron * 4)
        self.up_pool8 = up_pooling(start_neuron * 4, start_neuron * 2)
        self.conv8 = conv_bn_relu(start_neuron * 4, start_neuron * 2)
        self.up_pool9 = up_pooling(start_neuron * 2, start_neuron)
        self.conv9 = conv_bn_relu(start_neuron * 2, start_neuron)

        self.conv10 = nn.Conv2d(start_neuron, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)

        p7 = self.up_pool7(x4)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = F.relu(output)
        output = output.permute(0, 2, 3, 1)

        return output
