import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img_ori = cap.read()
    gray_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    if not ret:
        break

    cv2.imshow('reaa', img_ori)

    if cv2.waitKey(1) == ord('q'):
        break