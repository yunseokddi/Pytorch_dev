import cv2

img = cv2.imread('./sample/original.jpg')
resize_img = cv2.resize(img, (960,540))
cv2.imwrite('sample/resize_2.jpg', resize_img)