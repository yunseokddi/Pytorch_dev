import torch
import cv2
from efficientnet_pytorch import EfficientNet
import datetime


def get_epochtime_ms():
    return round(datetime.datetime.utcnow().timestamp() * 1000)

age_dict = {0: '1~5', 1: '6~10', 2: '11~15', 3: '16~20', 4: '21~25', 5: '26~30', 6: '31~35', 7: '36~40', 8: '41~45',
            9: '46~50', 10: '51~55', 11: '56~60', 12: '61~65', 13: '66~70', 14: '71~'}


class detect:
    def __init__(self, weigth_path, data_path):
        self.weight = weigth_path
        self.data = data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=15)
        self.model = self.model
        self.model.load_state_dict(torch.load(self.weight))
        self.model.to(self.device)
        self.model.eval()

    def detect(self):
        inputs = cv2.imread(self.data)
        inputs = cv2.resize(inputs, (200, 200))
        inputs = inputs[:, :, ::-1].transpose((2, 0, 1)).copy()
        inputs = torch.from_numpy(inputs).float().div(255.0).unsqueeze(0)
        inputs = inputs.cuda()
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        return age_dict[preds.item()]


detection = detect('./weights/class_15_weights/best_weights_acc78.pth', './sample.png')
start = get_epochtime_ms()
output = detection.detect()
total_time = get_epochtime_ms() - start
print(total_time)
print(output)
