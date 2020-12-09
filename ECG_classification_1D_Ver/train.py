import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy

from preprocess import load_ECG_dataset, split_dataset
from models import LSTM
from torch.optim import lr_scheduler


def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, device,
                num_epochs=25, early_stopping_epoch=10, savedirpath="result"):
    if not os.path.exists(savedirpath):
        os.makedirs(savedirpath)

    since = time.time()
    log_list = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping_count = 0
    start_time = time.time()
    print('TRAINING starts')

    for epoch in range(num_epochs):
        epoch = epoch + 1
        print('-' * 70)
        print('epoch : {}'.format(epoch))

        epoch_result = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long()


gpu = 0
num_epochs = 100
stopping_epoch = 20
learning_rate = 1.0

input_size = 1
hidden_size = 128
num_layers = 1
batch_size = 128

num_classes = 5
out = "./result"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = './data'

if __name__ == "__main__":
    train_dataset_dict, test_dataset_dict = load_ECG_dataset(root_dir)
    train_dataset, validation_dataset = split_dataset(train_dataset_dict, val_num=100, seed=0)

    model = LSTM(num_classes, input_size, hidden_size, num_layers, device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
