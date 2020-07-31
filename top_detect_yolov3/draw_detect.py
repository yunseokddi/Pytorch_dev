from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

from PIL import Image

import cv2

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import timeutil
from matplotlib.ticker import NullLocator


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

        self.imgs = []
        self.img_detections = []

    def detect(self, IMG_PATH):
        result_x1 = []
        result_y1 = []
        result_x2 = []
        result_y2 = []
        result_box_h = []

        dataloader = DataLoader(
            ImageFolder(IMG_PATH, img_size=self.IMG_SIZE),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        # start = timeutil.get_epochtime_ms()
        for batch_i, (img_paths, input_imgs, ori_img) in enumerate(dataloader):
            img = ori_img
            input_imgs = Variable(input_imgs.type(self.Tensor))

            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, 0.5, 0.5)

            self.imgs.extend(img_paths)
            self.img_detections.extend(detections)

        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        classes = load_classes('./classes.names')

        for img_i, (path, detections) in enumerate(zip(self.imgs, self.img_detections)):
            plt.figure()
            fig, ax = plt.subplots(1)
            img = np.squeeze(img, 0)
            ax.imshow(img)

            if detections is not None:
                detections = rescale_boxes(detections, self.IMG_SIZE, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    result_x1.append(x1.item())
                    result_y1.append(y1.item())
                    result_x2.append(x2.item())
                    result_y2.append(y2.item())

                    box_w = x2 - x1
                    box_h = y2 - y1

                    result_box_h.append(box_h.item())

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig(f"output/{filename}.jpg", bbox_inches="tight", pad_inches=0.0)
            plt.close()

        return result_x1, result_y1, result_x2, result_y2, result_box_h


detection = top_detect(weigth_PATH='./weights/yolov3_ckpt_99.pth')  # change your image path and weight path

start = timeutil.get_epochtime_ms()
x1, y1, x2, y2, box_h = detection.detect(IMG_PATH='sample/files_Test616.jpg', )  # output

# img = cv2.imread('./sample/resize_2.jpg')
# img = cv2.rectangle(img, (int(x1[0]), int(y1[0]+box_h[0])), (int(x2[0]), int(y1[0])), (0,0,255), 2)
# cv2.imshow('result', img)
# cv2.waitKey(0)

print("{} {} {} {}".format(x1[0], y1[0]+box_h[0], x2[0], y1[0]))
print("Latency: %fms" % (timeutil.get_epochtime_ms() - start))

