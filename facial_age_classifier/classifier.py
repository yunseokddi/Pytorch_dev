import cv2
import torch

from torchvision.models.resnet import resnet18
from net import Dc_model

weight_path = './checkpoint/facial_age_classifier1500.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=True).to(device)
model_ = Dc_model().to(device)
model.fc = model_

model.load_state_dict(torch.load(weight_path))


ori_img = cv2.imread('./6_sample.png')
img = cv2.resize(ori_img, (128,128))
img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
img = img.cuda()


output = model(img)
print(output.item())