import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import matplotlib as mpl

from dataloader import TrainDataSet
from torch.utils.data import DataLoader

mpl.rcParams['axes.unicode_minus'] = False

plt.rcParams["font.family"] = 'NanumGothic'


def imshow(inp, title=None, df=None):
    title_ls = title.tolist()
    cvt_title = []

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    for idx in title_ls:
        cvt_title.append(df["landmark_name"].loc[idx])
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

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

df = pd.read_csv(args.category_dir)

if __name__ == "__main__":
    train_dataset = TrainDataSet(args, transform=data_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    data = next(iter(train_dataloader))
    out = torchvision.utils.make_grid(data['image'])
    imshow(out, title=data['label'], df=df)
