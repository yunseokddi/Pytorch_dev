import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid


def imshow(inp, title=None, df=None):
    cvt_title = []

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    title = title.numpy()

    for label in title:
        cvt_title.append(df.iloc[label, 0])

    plt.imshow(inp)

    if title is not None:
        plt.title(cvt_title)

    plt.pause(3)  # edit showing time


df = pd.read_excel('./sample_data/KFace_data_information_Folder1_400.xlsx')  # change xlsx file directory

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    ]),
    transforms.ToTensor(),
    transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
])

if __name__ == '__main__':
    train_dataset = ImageFolder(root='./sample_data/image', transform=train_transforms)  # change image directory
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    for batch, (input, targets) in enumerate(train_loader):
        out = make_grid(input)
        imshow(out, title=targets, df=df)
