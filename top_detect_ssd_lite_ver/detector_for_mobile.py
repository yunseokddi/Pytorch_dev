from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import torch

net_type = 'mb2-ssd-lite'
model_path = './models/top_weight2_batch_32/mb2-ssd-lite-Epoch-295-Loss-1.7627840042114258.pth'
label_path = './models/voc-model-labels.txt'
img_path = './top_sample.jpg'


class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)

net.load(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device='cpu')

timer = Timer()
orig_image = cv2.imread(img_path)

image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
timer.start()
boxes, labels, probs = predictor.predict(image, 10, 0.4)
interval = timer.end()
print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
for i in range(boxes.size(0)):
    box = boxes[i, :]
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
cv2.imwrite('./result.jpg', orig_image)
