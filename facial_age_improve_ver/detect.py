import torch
import torchvision.transforms as transforms
import cv2

from efficientnet_pytorch import EfficientNet

age_dict = {0:10, 1:20, 2:30, 3:40, 4:50}

data_dir = './sample/sample4.jpg'
weight_path = 'weights/past_data_weights/best_weights_b3.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
model.load_state_dict(torch.load(weight_path))
model.to(device)

model.eval()

inputs = cv2.imread(data_dir)
inputs = inputs[:, :, ::-1].transpose((2, 0, 1)).copy()
inputs = torch.from_numpy(inputs).float().div(255.0).unsqueeze(0)
inputs = inputs.cuda()


outputs = model(inputs)
_, preds = torch.max(outputs, 1)

print('당신의 나이는 {}대 입니다.'.format(age_dict[preds.item()]))