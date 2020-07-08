from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from torchsummary import summary

import sys


path = './models/mb1-ssd-Epoch-99-Loss-2.7295520901679993.pth'
net = create_mobilenetv1_ssd(2)
# net.load_state_dict()
#
summary(net, (300,300,3))

print(net)