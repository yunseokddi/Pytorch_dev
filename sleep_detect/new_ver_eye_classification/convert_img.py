import cv2

img = cv2.imread('./sample_data/close_sample2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./close_sample_gray2.jpg', gray)