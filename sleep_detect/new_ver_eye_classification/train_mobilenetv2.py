from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


from torchvision import datasets
from torchvision.models import mobilenet_v2

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = './data/'
weight_PATH = './weights/'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize((24, 24)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transforms = transforms.Compose([
    transforms.Resize((24, 24)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder('./data/train/', transform=train_transforms)
test_data = datasets.ImageFolder('./data/test/', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True)

model = mobilenet_v2(pretrained=True).to(device)
model.classifier[1] = nn.Linear(1280,2)
model.to(device)


for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

train_loss = []
val_loss = []

epochs = 30

for epoch in range(epochs):
    print("epoch {}/{}".format(epoch + 1, epochs))
    running_loss = 0.0
    running_score = 0.0
    #       model.train()
    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        y_pred = model.forward(image)
        loss = criterion(y_pred, label)
        loss.backward()  # calculate derivatives
        optimizer.step()  # update parameters
        val, index_ = torch.max(y_pred, axis=1)
        running_score += torch.sum(index_ == label.data).item()
        running_loss += loss.item()

    epoch_score = running_score / len(train_loader.dataset)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss.append(epoch_loss)
    print("Training loss: {}, accuracy: {}".format(epoch_loss, epoch_score))

    if epoch % 9 == 0:
        torch.save(model.state_dict(), weight_PATH+'eyes_crop_new_ver_epoch'+str(epoch+1)+'.pth')
        print('save model epoch: {}'.format(epoch+1))

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_score = 0.0
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(image)
            loss = criterion(y_pred, label)
            running_loss += loss.item()

            val, index_ = torch.max(y_pred, axis=1)
            running_score += torch.sum(index_ == label.data).item()

        epoch_score = running_score / len(test_loader.dataset)
        epoch_loss = running_loss / len(test_loader.dataset)
        val_loss.append(epoch_loss)
        print("Validation loss: {}, accuracy: {}".format(epoch_loss, epoch_score))

print("learning finish")


plt.plot(train_loss,label='train loss')
plt.plot(val_loss,label='test loss')
plt.legend()
plt.show()