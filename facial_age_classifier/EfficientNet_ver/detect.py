import torch
import cv2
import argparse

from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='./sample/sample13.jpg')
opt = parser.parse_args()

age_dict = {0:'1~5', 1:'6~10', 2:'11~15', 3:'16~20',4 :'21~25', 5:'26~30', 6:'31~35', 7:'36~40', 8:'41~45',
            9:'46~50', 10:'51~55', 11:'56~60', 12:'61~65', 13:'66~70', 14:'71~'}

data_dir = opt.data
weight_path = 'weights/class_15_weights/best_weights_acc78.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=15)
model.load_state_dict(torch.load(weight_path))
model.to(device)

model.eval()

inputs = cv2.imread(data_dir)
inputs = cv2.resize(inputs, (200,200))
ori_img = inputs.copy()
inputs = inputs[:, :, ::-1].transpose((2, 0, 1)).copy()
inputs = torch.from_numpy(inputs).float().div(255.0).unsqueeze(0)
inputs = inputs.cuda()


outputs = model(inputs)
_, preds = torch.max(outputs, 1)

cv2.putText(ori_img, str(age_dict[preds.item()]),(75, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0))
cv2.imshow('result', ori_img)
cv2.waitKey(0)