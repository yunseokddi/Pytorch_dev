import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from test_transform import *
from resnet import ResNet18
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

PATH = './weights/model_keypoints_68pts_iter_450.pt'

net = ResNet18(136).to(device)
net.load_state_dict(torch.load(PATH))
net.eval()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, origin_image = cap.read()

    if not ret:
        break

    image = cv2.resize(origin_image, (224,224))
    origin_image = image.copy()
    image = Normalize(image)
    image = ToTensor(image)
    image = image.unsqueeze(0)

    with torch.no_grad():

        if (torch.cuda.is_available()):
            image = image.type(torch.cuda.FloatTensor)
            image.to(device)

        else:
            image = image.type(torch.FloatTensor)

        output_pts = net(image)

        output_pts = output_pts.view(output_pts.size()[0], -1, 2)

        image = image.squeeze()
        image = image.data
        if (torch.cuda.is_available()):
            image = image.cpu()
        print(image.shape)
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))


        output_pts = output_pts[0].data

        if (torch.cuda.is_available()):
            output_pts = output_pts.cpu()

        output_pts = output_pts.numpy()
        output_pts = (output_pts * 50) + 100


        for i in range(36,48):
            cv2.circle(image, (int(output_pts[i, 0]), int(output_pts[i, 1])), 1, (0, 0, 255), -1)


        cv2.imshow('result', image)
        if cv2.waitKey(1) == ord('q'):
            break
