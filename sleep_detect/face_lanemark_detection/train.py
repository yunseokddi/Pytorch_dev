import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import FacialKeypointsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet import ResNet18
from torch.autograd import Variable
from transforms import *

transform = transforms.Compose([Rescale(250),
                                RandomCrop(224),
                                Normalize(),
                                ToTensor()])

train_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                       root_dir='./data/training/',
                                       transform=transform)

test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                      root_dir='./data/test/',
                                      transform=transform)

# ------show image--------
# count = 0
# for i in range(10):
#     sample = data.__getitem__(count)
#     count += 1
#
#     image, landmark = sample['image'], sample['keypoints']
#     print(image.shape)
#
#     plt.imshow(image)
#     plt.scatter(landmark[:,0], landmark[:,1], s=10, marker='.', c='r')
#     plt.show()

batch_size = 256

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=4)

net = ResNet18(136).to('cuda')

criterion = nn.MSELoss().cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0001, amsgrad=True, weight_decay=0)

n_epochs = 500

num_iter = 0

for epoch in range(n_epochs):
    running_loss = 0.0

    net.train()

    # training
    for batch_i, data in enumerate(train_loader):
        images = data['image']
        key_pts = data['keypoints']

        key_pts = key_pts.view(key_pts.size(0), -1)

        images, key_pts = Variable(images), Variable(key_pts)

        key_pts = key_pts.type(torch.cuda.FloatTensor)
        images = images.type(torch.cuda.FloatTensor)
        images.to('cuda')

        output_pts = net(images)

        loss = criterion(output_pts, key_pts)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_i + 1) % 14 == 0:
            print('Epoch [{}/{}],Iter [{}/{}] Loss: {:.4f}, average_loss: {:.4f}'.format(
                epoch + 1, n_epochs, batch_i + 1, len(train_loader), loss.item(), running_loss / (batch_i + 1)))
            avg_loss = running_loss / (batch_i + 1)

            num_iter += 1

    val_loss = 0.0
    net.eval()

    # validation
    for batch_i, data in enumerate(test_loader):
        with torch.no_grad():
            images = data['image']
            key_pts = data['keypoints']

            key_pts = key_pts.view(key_pts.size(0), -1)

            images, key_pts = Variable(images), Variable(key_pts)

            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)
            images.to('cuda')

            output_pts = net(images)

            loss = criterion(output_pts, key_pts)
            val_loss += loss.item()


    val_loss /= len(test_dataset) / batch_size
    print('loss of val is {}'.format(val_loss))

    if (epoch + 1) % 50 == 0:
        torch.save(net.state_dict(), './weights/model_keypoints_68pts_iter_{}.pt'.format(epoch + 1))

print('Finish')