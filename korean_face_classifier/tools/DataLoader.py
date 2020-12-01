import pandas as pd
import torch
import os

from torch.utils.data import Dataset

age_dic = {'20대': 20,
           '30대': 30,
           '40대': 40,
           '50대': 50}

sex_dic = {'남': 0,
           '여': 1}

class k_face_dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = pd.read_excel(csv_file)
        self.transform = transform

        self.csv_file['연령대'] = self.csv_file['연령대'].replace(age_dic)
        self.csv_file['성별'] = self.csv_file['성별'].replace(sex_dic)

    def __len__(self):
        return len(next(os.walk(self.root_dir+'/image/'))[2])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        print(idx)
        img_name = os.path.join(self.root_dir, 'image', )

        return idx
