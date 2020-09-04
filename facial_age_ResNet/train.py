import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

from net import Dc_model
from torchvision.models.resnet import resnet18
from torch.utils.tensorboard import SummaryWriter

batch_size = 8
epochs = 3000
WEIGHT_PATH = './checkpoint/'
writer = SummaryWriter('./runs/experiment1')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.ImageFolder('./data/', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=True).to(device)
model_ = Dc_model().to(device)
model.fc = model_

dataiter = iter(train_loader)
images, labels = dataiter.next()

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer.add_graph(model, images.cuda())
writer.close()
count = 0

for i in range(epochs):
    dataiter = iter(train_loader)
    x, y = dataiter.next()
    x = x.cuda()
    y = .01 + y.reshape((batch_size, 1)).cuda().type(torch.float32) / 100

    z = model(x)

    loss = criterion(z, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if i % 100 == 0:
        writer.add_scalar('training loss', loss.item(), i)
        print('epoch: {} loss: {}'.format(i, loss.item()))
        torch.save(model.state_dict(), WEIGHT_PATH + 'facial_age_classifier' + str(i) + '.pth')
        print('save model epoch: {}'.format(i))
        correct = 0
