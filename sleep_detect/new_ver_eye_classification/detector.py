import cv2
import torch

from net import Dc_model
from torchvision.models import resnet18

def preprocess(img):
    img = cv2.resize(img, (50,50))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

weight_PATH = './weights/50_50_size/eyes_crop_new_ver_epoch98.pth'

model = resnet18(pretrained=True).to(device)
model_ = Dc_model().to(device)
model.fc = model_

model.load_state_dict(torch.load(weight_PATH))

def detect_img(img_path):
    img = cv2.imread(img_path)

    img = preprocess(img)
    img = img.to(device)

    with torch.no_grad():
        model.eval()
        output = model(img)
        output = torch.round(torch.sigmoid(output))


    return int(output[0][1].item())

if __name__ == '__main__':
    img_PATH = './sample_data/look_sample.png'
    result = detect_img(img_PATH)
    print(result)
    img_PATH = './sample_data/close_sample.png'
    result = detect_img(img_PATH)
    print(result)
    img_PATH = './close_sample_gray2.jpg'
    result = detect_img(img_PATH)
    print(result)