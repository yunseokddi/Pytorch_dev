import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import eyes_dataset
from model import Net
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


writer = SummaryWriter('runs/experiment')

plt.style.use('dark_background')

PATH = 'weights/trained.pth'

x_train = np.load('./dataset/x_train.npy').astype(np.float32)  # (2586, 26, 34, 1)
y_train = np.load('./dataset/y_train.npy').astype(np.float32)  # (2586, 1)
x_val = np.load('./dataset/x_val.npy').astype(np.float32)  # (288, 26, 34, 1)
y_val = np.load('./dataset/y_val.npy').astype(np.float32)  # (288, 1)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = eyes_dataset(x_train, y_train, transform=train_transform)
val_dataset = eyes_dataset(x_val, y_val, transform=val_transform)
#
# fig = plt.figure()
#
# for i in range(len(val_dataset)):
#     x ,y  = val_dataset[i]
#
#     plt.subplot(2, 1, 1)
#     plt.title(str(y_val[i]))
#     plt.imshow(x_val[i].reshape((26, 34)), cmap='gray')
#
#     plt.show()



train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

model = Net()
model.to('cuda')

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 50

for epoch in range(epochs):
    running_loss = 0.0
    running_acc = 0.0

    model.train()

    for i, data in enumerate(train_dataloader, 0):
        input_1, labels = data[0].to('cuda'), data[1].to('cuda')

        input = input_1.transpose(1, 3).transpose(2, 3)

        optimizer.zero_grad()

        outputs = model(input)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)


        if i % 80 == 79:
            # with torch.no_grad():
            #     val_loss = 0.0
            #     val_acc = 0.0
            #
            #     for j, val_data in enumerate(val_dataloader, 0):
            #         val_inputs, val_labels = val_data[0].to('cuda'), val_data[1].to('cuda')
            #
            #         val_inputs = val_inputs.transpose(1, 3).transpose(2, 3)
            #
            #         val_outputs = model(val_inputs)
            #         val_loss = criterion(val_outputs, val_labels)
            #         val_loss += val_loss
            #         val_acc += accuracy(val_outputs, val_labels)


            print('epoch: [%d/%d] train_loss: %.5f train_acc: %.5f' % (epoch + 1, epochs, running_loss / 80, running_acc/80))
            # print('epoch: [%d/%d] val_loss: %.5f val_acc: %.5f' % (epoch + 1, epochs, val_loss / 80, val_acc / 80))
            running_loss = 0.0


print("learning finish")
torch.save(model.state_dict(), PATH)

