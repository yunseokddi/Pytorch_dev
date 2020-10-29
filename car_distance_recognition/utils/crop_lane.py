import numpy as np
import cv2


def region_of_interest(img, vertieces):
    mask = np.zeros_like(img)

    channel_count = img.shape[2]

    match_mask_color = (255,) * channel_count

    cv2.fillPoly(mask, vertieces, match_mask_color)

    masked_img = cv2.bitwise_and(img, mask)

    return masked_img


cap = cv2.VideoCapture('../sample/sample3.mp4')

height = cap.get(4)
width = cap.get(3)

region_of_interest_vertices = [
    (300, height),
    ((300+830)/2, (height / 2) + 180),
    (830, height),
]

while True:
    ret, frame = cap.read()

    if ret:
        cropped_image = region_of_interest(frame, np.array([region_of_interest_vertices], np.int32))

        p = (width/2 - 100 , height-100)
        p1 = region_of_interest_vertices[0]
        p2 = region_of_interest_vertices[1]
        p3 = region_of_interest_vertices[2]

        alpha = ((p2[1] - p3[1]) * (p[0] - p3[0]) + (p3[0] - p2[0]) * (p[1] - p3[1])) / (
                    (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1]))

        beta = ((p3[1] - p1[1]) * (p[0] - p3[0]) + (p1[0] - p3[0]) * (p[1] - p3[1])) / (
                    (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1]))

        gamma = 1.0 - alpha - beta

        inner = False

        if alpha > 0 and beta > 0 and gamma > 0:
            inner = True

        print(inner)
        # print('{}, {}'.format(p[0], p[1]))
        cv2.circle(cropped_image, (int(p[0]), int(p[1])), 10, (255, 0, 0))
        cv2.imshow('asd', cropped_image)

        k = cv2.waitKey(1)

        if k == 27:
            break

    else:
        break
