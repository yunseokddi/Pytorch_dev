from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

import cv2
import timeutil

class top_detect():
    def __init__(self, weigth_PATH):
        self.IMG_SIZE = 320
        self.weigth_PATH = weigth_PATH
        self.device = 'cuda'
        self.model = Darknet('./config/yolov3-custom.cfg', img_size=self.IMG_SIZE).to(self.device)

        self.model.load_state_dict(torch.load(self.weigth_PATH))
        self.model.eval()  # Set in evaluation mode

        self.classes = load_classes('./label/classes.names')  # Extracts class labels from file
        self.Tensor = torch.cuda.FloatTensor

    def detect(self, IMG_PATH, conf_thres, nms_thres):
        imgs = []
        img_detections = []
        result_x1 = []
        result_y1 = []
        result_x2 = []
        result_y2 = []

        input_imgs = cv2.imread(IMG_PATH)
        ori_img = np.array(input_imgs)
        input_imgs = transforms.ToTensor()(input_imgs)
        input_imgs, _ = pad_to_square(input_imgs, 0)
        input_imgs = resize(input_imgs, 320)
        input_imgs = Variable(input_imgs.type(self.Tensor))
        input_imgs = torch.unsqueeze(input_imgs,0)

        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)
        imgs.extend(IMG_PATH)
        img_detections.extend(detections)

        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            if detections is not None:
                detections = rescale_boxes(detections, self.IMG_SIZE, ori_img.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    result_x1.append(x1.item())
                    result_x2.append(x2.item())
                    result_y1.append(y1.item())
                    result_y2.append(y2.item())
                    # cv2.rectangle(ori_img, (x1,y1),(x2,y2),(255,0,0))
                    # cv2.imshow('asd', ori_img)
                    # q = cv2.waitKey(0)
                    #
                    # if q == 27:
                    #     cv2.destroyAllWindows()

        return result_x1, result_y1, result_x2, result_y2

detection = top_detect(weigth_PATH='./weights/yolov3_ckpt_99.pth')  # change your image path and weight path

total_time = 0.0
count = 0

for i in range(1000):
    start = timeutil.get_epochtime_ms()
    x1, x2, y1, y2 = detection.detect(IMG_PATH='sample.jpg', conf_thres=0.5, nms_thres=0.5)  # output
    total_time += timeutil.get_epochtime_ms() - start
    count += 1

print('avg time is '+str(total_time/count))