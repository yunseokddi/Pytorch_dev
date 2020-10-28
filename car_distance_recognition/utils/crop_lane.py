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
    (0, height),
    (width / 2, height / 2),
    (width, height),
]

while True:
    ret, frame = cap.read()

    if ret:
        cropped_image = region_of_interest(frame, np.array([region_of_interest_vertices], np.int32))
        cv2.imshow('asd',cropped_image)

        k = cv2.waitKey(1)

        if k == 27:
            break

    else:
        break