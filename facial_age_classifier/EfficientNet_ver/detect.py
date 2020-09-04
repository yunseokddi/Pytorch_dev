import torch
import torchvision.transforms as transforms
import cv2
import argparse

from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='./sample/sample.jpg')
opt = parser.parse_args()

age_dict = {0:'1~20', 1:'21~27', 2:'28~45', 3:'46~65'}

data_dir = opt.data
weight_path = 'weights/best_weights_b3.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=4)
model.load_state_dict(torch.load(weight_path))
model.to(device)

model.eval()

inputs = cv2.imread(data_dir)
inputs = inputs[:, :, ::-1].transpose((2, 0, 1)).copy()
inputs = torch.from_numpy(inputs).float().div(255.0).unsqueeze(0)
inputs = inputs.cuda()


outputs = model(inputs)
_, preds = torch.max(outputs, 1)

print('당신의 나이는 {}세 입니다.'.format(age_dict[preds.item()]))