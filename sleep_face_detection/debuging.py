import cv2
import numpy as np
from model import Net
import torch

IMG_SIZE = (34, 26)
PATH = './weights/trained.pth'

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()


def predict(pred):
    pred = pred.transpose(1, 3).transpose(2, 3)

    outputs = model(pred)

    pred_tag = torch.round(torch.sigmoid(outputs))

    return pred_tag


image_eye = cv2.imread('./dataset/5.jpg')
image_eye = cv2.resize(image_eye, dsize=IMG_SIZE)
image_eye = cv2.cvtColor(image_eye, cv2.COLOR_BGR2GRAY)
image_eye = image_eye.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
image_eye = torch.from_numpy(image_eye)

pred= predict(image_eye)

print(pred.item())
