import torch.nn as nn

from torchvision.models import resnet101
from torchsummary import summary

model = resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1049)
model.to('cuda')

summary(model, (3,224,224))
