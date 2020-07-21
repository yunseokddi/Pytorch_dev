import cv2
import numpy as np
from util.model import Net
import torch
from landmark_detector import dectector

IMAGE_PATH = './data/sample4.jpg'
IMG_SIZE = (34, 26)
CLASSIFIER_PATH = './weights/classifier_weights.pth'

classifier_model = Net()
classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH))
classifier_model.eval()

n_count = 0


def crop_eye(gray, eye_points):
    x1, y1 = np.amin(eye_points[:, 0], axis=0), np.amin(eye_points[:, 1], axis=0)
    x2, y2 = np.amax(eye_points[:, 0], axis=0), np.amax(eye_points[:, 1], axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


def predict(pred):
    pred = pred.transpose(1, 3).transpose(2, 3)

    outputs = classifier_model(pred)
    print(torch.sigmoid(outputs))
    pred_tag = torch.round(torch.sigmoid(outputs))

    return pred_tag


def main_detection():
    img_ori = cv2.imread(IMAGE_PATH)

    shapes, image = dectector(img_ori)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    # cv2.imshow('l', eye_img_l)
    # cv2.imshow('r', eye_img_r)
    # cv2.waitKey(0)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

    cv2.imshow('l', eye_img_l)
    cv2.imshow('r', eye_img_r)
    cv2.waitKey(0)

    eye_input_l = torch.from_numpy(eye_input_l)
    eye_input_r = torch.from_numpy(eye_input_r)

    pred_l = predict(eye_input_l)
    pred_r = predict(eye_input_r)

    # if pred_l.item() == 0.0 and pred_r.item() == 0.0:
    #     n_count += 1
    #
    # else:
    #     n_count = 0
    #
    # if n_count > 100:
    #     cv2.putText(img, "Wake up", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # visualize
    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    # cv2.rectangle(img_ori, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
    # cv2.rectangle(img_ori, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)
    #
    # cv2.putText(img_ori, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(img_ori, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # 감으면 0, 뜨면 0.1
    print('left result is : ' + state_l)
    print('right result is : ' + state_r)
    # cv2.imwrite('./result.jpg',image)
    # cv2.imshow('result', image)
    # cv2.waitKey(0)


main_detection()
