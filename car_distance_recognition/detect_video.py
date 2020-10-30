import argparse
import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (non_max_suppression, scale_coords, plot_one_box)
from utils.torch_utils import select_device


def detect(source, weights, view_img=True, imgsz=640, conf_thres=0.8, iou_thres=0.7, classes=None, agnostic_nms=True,
           focal_distance=0.03, car_height=1.7):
    device = select_device('0')
    half = True
    model = attempt_load(weights, map_location=device)
    model.half()
    dataset = LoadImages(source, img_size=imgsz)
    colors = (0, 0, 255)
    names = model.module.names if hasattr(model, 'module') else model.names

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half())

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=True)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(det)
                # print(det.shape)
                # print(det[0][4])
                # max_det = det[0]
                for *xyxy, conf, cls in reversed(det):
                    if view_img:

                        car_img_height = xyxy[3] - xyxy[1]
                        label = '%s %.2f' % (names[int(cls)], conf)
                        distance = (
                                           focal_distance / car_img_height) * car_height * 10000  # distance = (f/obj height) * real height
                        result_distance = str(round(distance.item())) + 'm'
                        inner = plot_one_box(xyxy, im0, label=label, color=colors, line_thickness=3)
                        tl = 3 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        tf = max(tl - 1, 1)

                        if inner is True:
                            cv2.putText(im0, result_distance, (c1[0] - 15, c1[1] + 75), 0, tl / 3, [225, 255, 255],
                                        thickness=tf,
                                        lineType=cv2.LINE_AA)

            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./runs/exp2/weights/best.pt', help='enter weight path')
    parser.add_argument('--source', type=str, default='./sample/sample2.mp4', help='enter source path')
    parser.add_argument('--focal_distance', type=float, default='0.03', help='enter focal distance')
    parser.add_argument('--car_height', type=float, default='1.7', help='enter car''s avg height')
    opt = parser.parse_args()

    weights = opt.weights
    source = opt.source
    focal_distance = opt.focal_distance
    car_height = opt.car_height

    with torch.no_grad():
        detect(source=source, weights=weights, focal_distance=focal_distance, car_height=car_height)
