import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
BATCH_SIZE = 6
FINE_TUNE = False

writer = SummaryWriter('runs/TEST')

# random data 생성하여 data 수 증가
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder('hymenoptera_data/train/', train_transform)
test_data = datasets.ImageFolder('hymenoptera_data/val/', test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def sample_show():
    inp, classes = next(iter(train_loader))
    title = [train_data.classes[i] for i in classes]
    inp = torchvision.utils.make_grid(inp, nrow=8)

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# sample_show()

inception = models.inception_v3(pretrained=True)

inception.aux_logits = False

if not FINE_TUNE:
    for parameter in inception.parameters():
        parameter.requires_grad = False  # 모든 layer에 대해 학습을 막는다.

n_features = inception.fc.in_features  # fc로 들어오는 2048
inception.fc = nn.Linear(n_features, 2)

if USE_CUDA:
    inception = inception.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, inception.parameters()), lr=0.001)


def train_model(model, criterion, optimizer, epochs=30):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    img_grid = torchvision.utils.make_grid(images)

    matplotlib_imshow(img_grid, one_channel=True)

    writer.add_image('test', img_grid)

    for epoch in range(epochs):
        epoch_loss = 0
        for step, (inputs, y_true) in enumerate(train_loader):
            if USE_CUDA:
                x_sample, y_true = inputs.cuda(), y_true.cuda()

            x_sample, y_true = Variable(x_sample), Variable(y_true)

            optimizer.zero_grad()

            y_pred = inception(x_sample)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            _loss = loss.item()
            epoch_loss += _loss

            if step % 1000 == 999:
                writer.add_scalar('training loss', _loss/1000, epoch*len(train_loader)+ step)




        print(f'[{epoch + 1}] loss: {epoch_loss / step:.4}')


train_model(inception, criterion, optimizer)


# reference by https://github.com/AndersonJo/pytorch-examples/blob/master/10%20%5BTL%5D%20Transfer%20Learning.ipynb
