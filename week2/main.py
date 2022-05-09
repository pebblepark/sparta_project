import cv2
import numpy as np

# 딥러닝 모델 로드하기
net = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')

img = cv2.imread('imgs/01.jpg')

cv2.imshow('img', img)
cv2.waitKey(0)