import cv2
import torch
import os
import torch.nn as nn

from torchvision.models import mobilenet_v2


def preprocess(img):
    img = cv2.resize(img, (24, 24))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

weight_PATH = './weights/mobilenetv2/24_24/eyes_crop_new_ver_epoch11.pth'

model = mobilenet_v2(pretrained=True).to(device)
model.classifier[1] = nn.Linear(1280,2)
model.to(device)


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
    path_dir = './data/test/close_look'

    file_list = os.listdir(path_dir)

    all = 0
    count = 0

    for img in file_list:
        result = detect_img(path_dir + '/' + img)

        if result == 0:
            count += 1

        all += 1

    print(str(count/all*100)+'%')