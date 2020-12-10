import os
import torch
import torch.nn as nn
import time
import copy
import numpy as np

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
                print(inputs.shape)
                inputs = inputs.to(device)
                labels = labels.to(device).long()


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)

            epoch_time = time.time() - start_time

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = np.float64(running_corrects.double() / len(dataloaders_dict[phase].dataset))

            if phase == "train":
                print("train      loss: {:.4f}, accuracy : {:.4f}, elapsed time: {:.4f}"
                      .format(epoch_loss, epoch_acc, epoch_time))
            else:
                print("validation loss: {:.4f}, accuracy : {:.4f}, elapsed time: {:.4f}"
                      .format(epoch_loss, epoch_acc, epoch_time))

            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                save_model_weight_path = os.path.join(savedirpath, "trained_model.pt")
                torch.save(model.state_dict(), save_model_weight_path)

                early_stopping_count = 0
            elif phase == "val" and epoch_acc <= best_acc:
                early_stopping_count += 1

            epoch_result["epoch"] = epoch
            epoch_result["elapsed_time"] = epoch_time
            if phase == "train":
                epoch_result["train/loss"] = epoch_loss
                epoch_result["train/accuracy"] = epoch_acc

            else:
                epoch_result["validation/loss"] = epoch_loss
                epoch_result["validation/accuracy"] = epoch_acc

        log_list.append(epoch_result)
        # plot_learning_curve(savedirpath)

        # early stopping
        if early_stopping_count == early_stopping_epoch:
            print("Eearly stopping have been performed in this training")
            print("Epoch : {}".format(epoch))
            break

    time_elapsed = time.time() - since
    print("---------------------------------------------")
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best epoch   : {}".format(best_epoch))
    print('Best validation Accuracy: {:4f}'.format(best_acc))
    print("---------------------------------------------")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=16, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    model = LSTM(num_classes, input_size, hidden_size, num_layers, device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    trained_model = train_model(model, dataloaders_dict, criterion,
                                optimizer, exp_lr_scheduler, device,
                                num_epochs, stopping_epoch,
                                savedirpath=out)