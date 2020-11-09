import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import matplotlib as mpl

from PIL import Image
from DataLoader import LandmarkDataset
from torch.utils.data import DataLoader

mpl.rcParams['axes.unicode_minus'] = False

plt.rcParams["font.family"] = 'NanumGothic'


def imshow(inp, title=None, df=None):
    cvt_title = []

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    for idx in title:
        cvt_title.append(df["landmark_name"].loc[idx.item()])

    plt.imshow(inp)

    if title is not None:
        plt.title(cvt_title)
    plt.pause(0)


parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', dest='train_dir', default="./dataset/train/")
parser.add_argument('--train_csv_dir', dest='train_csv_dir', default="./dataset/train.csv")
parser.add_argument('--train_csv_exist_dir', dest='train_csv_exist_dir', default="./dataset/train_exist.csv")
parser.add_argument('--category_dir', dest='category_dir', default="./dataset/category.csv")
parser.add_argument('--image_size', dest='image_size', type=int, default=256)

args = parser.parse_args()

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomResizedCrop(224),
        transforms.RandomAffine(
            degrees=15, translate=(0.2, 0.2),
            scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
    ]),
    transforms.ToTensor(),
    transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
])

df = pd.read_csv(args.category_dir)

if __name__ == "__main__":
    train_dataset = LandmarkDataset(mode='train', transforms=transforms_train)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    df = pd.read_csv(args.category_dir)

    for image, label in train_dataloader:
        out = torchvision.utils.make_grid(image)
        print(type(label[0].item()))
        imshow(out, title=label, df=df)
