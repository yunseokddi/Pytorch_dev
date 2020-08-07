import cv2

from utils.landmark_transform import *
from utils.landmark_resnet import ResNet18
from utils.classifier_net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANDMARK_WEIGHT_PATH = './weights/model_keypoints_68pts_iter_450.pt'  # change the weights path
LANDMARK_NET = ResNet18(136).to(device)
LANDMARK_NET.load_state_dict(torch.load(LANDMARK_WEIGHT_PATH))
LANDMARK_NET.eval()

CLASSIFIER_WEIGHT_PATH = './weights/classifier_weights_iter_50.pt'
CLASSIFIER_NET = Net()
CLASSIFIER_NET.load_state_dict(torch.load(CLASSIFIER_WEIGHT_PATH))
CLASSIFIER_NET.eval()

EYES_SIZE = (34, 26)

n_count = 0


def crop_eye(gray, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * EYES_SIZE[1] / EYES_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


def landmark_detect(IMAGE_FILE):

    image = cv2.resize(IMAGE_FILE, (224, 224))
    image = Normalize(image)
    image = ToTensor(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        image = image.type(torch.cuda.FloatTensor)
        image.to(device)

        output_pts = LANDMARK_NET(image)
        output_pts = output_pts.view(output_pts.size()[0], -1, 2)
        output_pts = output_pts[0].data

        output_pts = output_pts.cpu()

        output_pts = output_pts.numpy()
        output_pts = (output_pts * 50) + 100

    return output_pts  # output_pts[i,0], output_pts[i,1] (i is 36~47)


def eyes_classifier(pred):
    pred = pred.transpose(1, 3).transpose(2, 3)

    outputs = CLASSIFIER_NET(pred)

    pred_tag = torch.round(torch.sigmoid(outputs))

    return pred_tag

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img_ori = cap.read()
        gray_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        showing_img = cv2.resize(img_ori, (224,224))

        if not ret:
            break

        output_pts = landmark_detect(img_ori)

        eye_img_l, eye_rect_l = crop_eye(gray_img, eye_points=output_pts[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray_img, eye_points=output_pts[42:48])

        try:
            eye_img_l = cv2.resize(eye_img_l, dsize=EYES_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=EYES_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        except:
            continue

        eye_input_l = eye_img_l.copy().reshape((1, EYES_SIZE[1], EYES_SIZE[0], 1)).astype(np.float32)
        eye_input_r = eye_img_r.copy().reshape((1, EYES_SIZE[1], EYES_SIZE[0], 1)).astype(np.float32)

        eye_input_l = torch.from_numpy(eye_input_l)
        eye_input_r = torch.from_numpy(eye_input_r)

        pred_l = eyes_classifier(eye_input_l)
        pred_r = eyes_classifier(eye_input_r)

        if pred_l.item() == 0.0 and pred_r.item() == 0.0:
            n_count += 1

        else:
            n_count = 0

        if n_count > 100:
            cv2.putText(showing_img, "Wake up", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # visualize
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        cv2.rectangle(showing_img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
        cv2.rectangle(showing_img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

        cv2.putText(showing_img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(showing_img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # for i in range(36,48):
        #     cv2.circle(showing_img, (int(output_pts[i, 0]), int(output_pts[i, 1])), 1, (0, 0, 255), -1)

        cv2.imshow('asd', showing_img)



        if cv2.waitKey(1) == ord('q'):
            break
        #
        # eye_input_l = eye_img_l.copy().reshape((1, EYES_SIZE[1], EYES_SIZE[0], 1)).astype(np.float32)
        # eye_input_r = eye_img_r.copy().reshape((1, EYES_SIZE[1], EYES_SIZE[0], 1)).astype(np.float32)