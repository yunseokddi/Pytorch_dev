import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision.models.resnet import resnet18
from net import Dc_model

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



batch_size = 16

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.ImageFolder('./data/', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

weight_path = './checkpoint/facial_age_classifier900.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=True).to(device)
model_ = Dc_model().to(device)
model.fc = model_

model.load_state_dict(torch.load(weight_path))

for i in range(10):
    dataiter = iter(train_loader)
    x, y = dataiter.next()
    x = x.cuda()
    y = .01 + y.reshape((batch_size, 1)).cuda().type(torch.float32) / 100
    z = model(x)

    imshow(torchvision.utils.make_grid(x.cpu()))
    y = (y * 100).type(torch.int64)
    z = (z * 100).type(torch.int64)
    print(y.reshape(1, 16).tolist()[0])
    print(z.reshape(1, 16).tolist()[0])
