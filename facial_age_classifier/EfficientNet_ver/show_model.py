import torch
import cv2

# from torchsummary import summary
# from efficientnet_pytorch import EfficientNet
#
# model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=15)
# model.to('cpu')
#
# summary(model, (3,200,200))

img = cv2.imread('./sample.png')
rs_img = cv2.resize(img, (256,256))

cv2.imshow('asd', rs_img)
cv2.waitKey(0)