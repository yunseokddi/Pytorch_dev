import torch
from torch.backends import cudnn
import time

from backbone import EfficientDetBackbone
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, \
    plot_one_box

compound_coef = 0
force_input_size = None

anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.75
iou_threshold = 0.75

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['TOP']

color_list = standard_to_bgr(STANDARD_COLORS)

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size


model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'./weights/efficientdet-d0_99_12000.pth')) #weight 경로 수정
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


def display(preds, imgs, imshow=True, imwrite=False):
    x = []
    y = []
    result_1 = []
    result_2 = []
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            x, y =  plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
            result_1.append(x)
            result_2.append(y)

    return result_1, result_2


def detect(img_path):
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    with torch.no_grad():
        t1 = time.time()
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
    out = invert_affine(framed_metas, out)
    c1, c2 = display(out, ori_imgs, imshow=True, imwrite=False)
    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    return c1, c2

img_path = './top_sample.jpg'  # image path 수정하세여

result1, result2 = detect(img_path) #사각형 왼쪽위, 오른쪽 아래 좌표, 최종 return value
print(result1, result2)