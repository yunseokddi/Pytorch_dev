import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid

classes = {0: 'normal',
           1: 'pvc'}

def imshow(img, labels):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    mapping_labels = []

    for label in labels:
        print(classes[label.item()])
        mapping_labels.append(classes[label.item()])

    plt.title(mapping_labels)
    plt.show()

train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='./data/2_class_img_data', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(make_grid(images), labels)
