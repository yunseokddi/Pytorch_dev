import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='./data/2_class_img_data', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

# for i, data in enumerate(train_loader):
