from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import timeutil


class top_detect():
    def __init__(self, IMG_PATH, weigth_PATH):
        self.IMG_PATH = IMG_PATH
        self.IMG_SIZE = 320
        self.weigth_PATH = weigth_PATH
        self.device = 'cuda'
        self.model = Darknet('./config/yolov3-custom.cfg', img_size=self.IMG_SIZE).to(self.device)

        self.model.load_state_dict(torch.load(self.weigth_PATH))
        self.model.eval()  # Set in evaluation mode

        self.dataloader = DataLoader(
            ImageFolder(IMG_PATH, img_size=self.IMG_SIZE),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        self.classes = load_classes('./label/classes.names')  # Extracts class labels from file
        self.Tensor = torch.cuda.FloatTensor

        self.imgs = []
        self.img_detections = []

    def detect(self):
        result_x1 = []
        result_y1 = []
        result_x2 = []
        result_y2 = []
        # start = timeutil.get_epochtime_ms()
        for batch_i, (img_paths, input_imgs) in enumerate(self.dataloader):
            input_imgs = Variable(input_imgs.type(self.Tensor))

            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, 0.7, 0.7)

            self.imgs.extend(img_paths)
            self.img_detections.extend(detections)

        for img_i, (path, detections) in enumerate(zip(self.imgs, self.img_detections)):
            img = np.array(Image.open(path))

            if detections is not None:
                detections = rescale_boxes(detections, self.IMG_SIZE, img.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    result_x1.append(x1)
                    result_x2.append(x2)
                    result_y1.append(y1)
                    result_y2.append(y2)

        return result_x1,result_y1, result_x2, result_y2

detection = top_detect(IMG_PATH='./sample', weigth_PATH='./weights/yolov3_ckpt_99.pth') #change your image path and weight path

# start = timeutil.get_epochtime_ms()
x1, x2, y1, y2 = detection.detect() #output

# print("Latency: %fms" % (timeutil.get_epochtime_ms() - start))
