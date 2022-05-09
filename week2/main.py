import cv2
import numpy as np

# 딥러닝 모델 로드하기
net = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')

img = cv2.imread('imgs/01.jpg')

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))

print(img.shape)

# 전처리
MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

print(blob.shape)

cv2.imshow('img', img)
cv2.waitKey(0)