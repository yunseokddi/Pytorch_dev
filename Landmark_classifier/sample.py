import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch

from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import Conv2d, AdaptiveAvgPool2d, Linear
import torch.nn.functional as F

# arguments
# train_csv_exist, test_csv_exist는 glob.glob이 생각보다 시간을 많이 잡아먹어서 iteration 시간을 줄이기 위해 생성되는 파일입니다.
# 이미 생성되어 있을 경우 train_csv_exist.csv 파일로 Dataset을 생성합니다.
parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', dest='train_dir', default="./dataset/train/")
parser.add_argument('--train_csv_dir', dest='train_csv_dir', default="./dataset/train.csv")
parser.add_argument('--train_csv_exist_dir', dest='train_csv_exist_dir', default="./dataset/train_exist.csv")

parser.add_argument('--test_dir', dest='test_dir', default="./dataset/test/")
parser.add_argument('--test_csv_dir', dest='test_csv_dir', default="./dataset/sample_submisstion.csv")
parser.add_argument('--test_csv_exist_dir', dest='test_csv_exist_dir',
                    default="./dataset/sample_submission_exist.csv")

parser.add_argument('--test_csv_submission_dir', dest='test_csv_submission_dir',
                    default="./dataset/my_submission.csv")
parser.add_argument('--model_dir', dest='model_dir', default="./checkpoint/")

parser.add_argument('--image_size', dest='image_size', type=int, default=256)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

parser.add_argument('--train', dest='train', type=bool, default=True)
parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=29)

args = parser.parse_args()

# 경로 생성
if not os.path.isdir(args.model_dir) :
    os.makedirs(args.model_dir)

# 파이토치 Dataset 생성 for Train / Test
class TrainDataset(Dataset) :
    def __init__(self, args) :
        self.train_dir = args.train_dir
        self.train_csv_dir = args.train_csv_dir
        self.train_csv_exist_dir = args.train_csv_exist_dir
        self.args = args
        self.train_image = list()
        self.train_label = list()
        if not os.path.isfile(self.train_csv_exist_dir) :
            self.train_csv = pd.read_csv(self.train_csv_dir)
            self.train_csv_exist = self.train_csv.copy()
            self.load_full_data()
            self.train_csv_exist.to_csv(self.train_csv_exist_dir, index=False)
        else :
            self.load_exist_data()

    def load_full_data(self) :
        for i in tqdm(range(len(self.train_csv))) :
            filename = self.train_csv['id'][i]
            fullpath = glob(self.train_dir + "*/*/" + filename.replace('[', '[[]') + ".JPG")[0]
            label = self.train_csv['landmark_id'][i]
            self.train_csv_exist.loc[i,'id'] = fullpath
            self.train_image.append(fullpath)
            self.train_label.append(label)


    def load_exist_data(self) :
        self.train_csv_exist = pd.read_csv(self.train_csv_exist_dir)
        for i in tqdm(range(len(self.train_csv_exist))) :
            fullpath = self.train_csv_exist['id'][i]
            label = self.train_csv_exist['landmark_id'][i]
            self.train_image.append(fullpath)
            self.train_label.append(label)


    def __len__(self) :
        return len(self.train_image)

    def __getitem__(self, idx) :
        image = Image.open(self.train_image[idx])
        image = image.resize((self.args.image_size, self.args.image_size))
        image = np.array(image) / 255.
        image = np.transpose(image, axes=(2, 0, 1))
        label = self.train_label[idx]

        return {'image' : image, 'label' :label}

# class TestDataset(Dataset) :
#     def __init__(self, args) :
#         self.test_dir = args.test_dir
#         self.test_csv_dir = args.test_csv_dir
#         self.test_csv_exist_dir = args.test_csv_exist_dir
#         self.args = args
#         self.test_image = list()
#         self.test_label = list()
#         if not os.path.isfile(self.test_csv_exist_dir) :
#             self.test_csv = pd.read_csv(self.test_csv_dir)
#             self.test_csv_exist = self.test_csv.copy()
#             self.load_full_data()
#             self.test_csv_exist.to_csv(self.test_csv_exist_dir, index=False)
#         else :
#             self.load_exist_data()
#
#     def load_full_data(self) :
#         for i in tqdm(range(len(self.test_csv))) :
#             filename = self.test_csv['id'][i]
#             fullpath = glob(self.test_dir + "*/" + filename.replace('[', '[[]') + ".JPG")[0]
#             label = self.test_csv['id'][i]
#
#             self.test_csv_exist.loc[i,'id'] = fullpath
#             self.test_image.append(fullpath)
#             self.test_label.append(label)
#
#
#     def load_exist_data(self) :
#         self.test_csv_exist = pd.read_csv(self.test_csv_exist_dir)
#         for i in tqdm(range(len(self.test_csv_exist))) :
#             fullpath = self.test_csv_exist['id'][i]
#             label = self.test_csv_exist['id'][i]
#
#             self.test_image.append(fullpath)
#             self.test_label.append(label)


    # def __len__(self) :
    #     return len(self.test_image)
    #
    # def __getitem__(self, idx) :
    #     image = Image.open(self.test_image[idx])
    #     image = image.resize((self.args.image_size, self.args.image_size))
    #     image = np.array(image) / 255.
    #     image = np.transpose(image, axes=(2, 0, 1))
    #     label = self.test_label[idx]
    #
    #     return {'image' : image, 'label' :label}

# DataLoader 생성을 위한 collate_fn
def collate_fn(batch) :
    image = [x['image'] for x in batch]
    label = [x['label'] for x in batch]

    return torch.tensor(image).float().cuda(), torch.tensor(label).long().cuda()

def collate_fn_test(batch) :
    image = [x['image'] for x in batch]
    label = [x['label'] for x in batch]

    return torch.tensor(image).float().cuda(), label

# Dataset, Dataloader 정의
train_dataset = TrainDataset(args)
# test_dataset = TestDataset(args)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_test)

# Model
# 여기서는 간단한 CNN 3개짜리 모델을 생성하였습니다.
class Network(nn.Module) :
    def __init__(self) :
        super(Network, self).__init__()
        self.conv1 = Conv2d(3, 64, (3,3), (1,1), (1,1))
        self.conv2 = Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.conv3 = Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.fc = Linear(64, 1049)

    def forward(self, x) :
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = AdaptiveAvgPool2d(1)(x).squeeze()
        x = self.fc(x)
        return x

model = Network()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
# Training
# 매 epoch마다 ./ckpt 파일에 모델이 저장됩니다.
# validation dataset 없이 모든 train data를 train하는 방식입니다.
if args.train :
    model.train()
    for epoch in range(args.epochs) :
        epoch_loss = 0.
        for iter, (image, label) in enumerate(train_dataloader) :
            print(image)
            print(label)
            # pred = model(image)
    #         loss = criterion(input=pred, target=label)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.detach().item()
    #         print('epoch : {0} step : [{1}/{2}] loss : {3}'.format(epoch, iter, len(train_dataloader), loss.detach().item()))
    #     epoch_loss /= len(train_dataloader)
    #     print('\nepoch : {0} epoch loss : {1}\n'.format(epoch, epoch_loss))
    #
    #     torch.save(model.state_dict(), args.model_dir + "epoch_{0:03}.pth".format(epoch))
    # # 모든 epoch이 끝난 뒤 test 진행
    # model.eval()
    # submission = pd.read_csv(args.test_csv_dir)
    # for iter, (image, label) in enumerate(test_dataloader):
    #     pred = model(image)
    #     pred = nn.Softmax(dim=1)(pred)
    #     pred = pred.detach().cpu().numpy()
    #     landmark_id = np.argmax(pred, axis=1)
    #     confidence = pred[0,landmark_id]
    #     submission.loc[iter, 'landmark_id'] = landmark_id
    #     submission.loc[iter, 'conf'] = confidence
    # submission.to_csv(args.test_csv_submission_dir, index=False)

# Test
# argument의 --train을 False로 두면 Test만 진행합니다.
# Softmax로 confidence score를 계산하고, argmax로 class를 추정하여 csv 파일로 저장합니다.
# 현재 batch=1로 불러와서 조금 느릴 수 있습니다.
# else :
#     model.load_state_dict(torch.load(args.model_dir + "epoch_{0:03}.pth".format(args.load_epoch)))
#     model.eval()
#     submission = pd.read_csv(args.test_csv_dir)
#     for iter, (image, label) in enumerate(test_dataloader):
#         pred = model(image)
#         pred = nn.Softmax(dim=1)(pred)
#         pred = pred.detach().cpu().numpy()
#         landmark_id = np.argmax(pred, axis=1)
#         confidence = pred[0,landmark_id]
#         submission.loc[iter, 'landmark_id'] = landmark_id
#         submission.loc[iter, 'conf'] = confidence
#     submission.to_csv(args.test_csv_submission_dir, index=False)