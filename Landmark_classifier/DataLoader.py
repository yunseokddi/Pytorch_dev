import glob
import csv
import os

from torch import Tensor
from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset
from torchvision import transforms


class LandmarkDataset(Dataset):
    def __init__(self, mode: str = 'train', transforms: transforms = None):
        self.mode = mode

        self.image_ids = glob.glob(f'./dataset/{mode}/**/**/*')
        if self.mode == 'train':
            self.image_ids = glob.glob(f'./dataset/{mode}/**/**/*')
            with open('./dataset/train.csv') as f:
                labels = list(csv.reader(f))[1:]
                self.labels = {label[0]: int(label[1]) for label in labels}

        else:
            self.image_ids = glob.glob(f'./dataset/{mode}/**/*')

        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image = Image.open(self.image_ids[index]).convert('RGB')
        image_id = os.path.splitext(os.path.basename(self.image_ids[index]))[0]
        if self.transforms is not None:
            image = self.transforms(image)

        if self.mode == 'train':
            label = self.labels[image_id]
            return image, label
        else:
            return image_id, image