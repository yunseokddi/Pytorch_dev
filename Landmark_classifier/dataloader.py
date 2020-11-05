from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob
from PIL import Image

import pandas as pd
import os
import torch


class TrainDataSet(Dataset):
    def __init__(self, train_dir=None, train_csv_dir=None, train_csv_exist_dir=None, transform=None):
        self.train_dir = train_dir
        self.train_csv_dir = train_csv_dir
        self.train_csv_exist_dir = train_csv_exist_dir
        self.train_image = []
        self.train_label = []
        self.transform = transform
        if not os.path.isfile(self.train_csv_exist_dir):
            self.train_csv = pd.read_csv(self.train_csv_dir)
            self.train_csv_exist = self.train_csv.copy()
            self.load_full_data()
            self.train_csv_exist.to_csv(self.train_csv_exist_dir, index=False)
        else:
            self.load_exist_data()

    def load_full_data(self):
        for i in tqdm(range(len(self.train_csv))):
            filename = self.train_csv['id'][i]
            fullpath = glob(self.train_dir + "*/*/" + filename.replace('[', '[[]') + ".JPG")[0]
            label = self.train_csv['landmark_id'][i]
            self.train_csv_exist.loc[i, 'id'] = fullpath
            self.train_image.append(fullpath)
            self.train_label.append(label)

    def load_exist_data(self):
        self.train_csv_exist = pd.read_csv(self.train_csv_exist_dir)
        for i in tqdm(range(len(self.train_csv_exist))):
            fullpath = self.train_csv_exist['id'][i]
            label = self.train_csv_exist['landmark_id'][i]
            self.train_image.append(fullpath)
            self.train_label.append(label)

    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, idx):
        image = Image.open(self.train_image[idx])
        image = self.transform(image)
        label = self.train_label[idx]

        return {'image': image, 'label': label}

def collate_fn(batch) :
    image = [x['image'] for x in batch]
    label = [x['label'] for x in batch]

    return torch.tensor(image).float().cuda(), torch.tensor(label).long().cuda()

