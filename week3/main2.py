import numpy as np
import cv2

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

img = cv2.imread('imgs/02.jpg')

h, w, c = img.shape

# 이미지 전처리하기
blob = cv2.dnn.blobFromImage(img, size=(300, 300), mean=(104., 177., 123.))

# 얼굴 영역 탐지 모델로 추론하기
facenet.setInput(blob)
dets = facenet.forward()

# 각 얼굴에 대해서 반복문 돌기
for i in range(dets.shape[2]):
    confidence = dets[0, 0, i, 2]

    if confidence < 0.5:
        continue

    # 사각형 꼭지점 찾기
    x1 = int(dets[0, 0, i, 3] * w)
    y1 = int(dets[0, 0, i, 4] * h)
    x2 = int(dets[0, 0, i, 5] * w)
    y2 = int(dets[0, 0, i, 6] * h)

    face = img[y1:y2, x1:x2]

    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)


cv2.imshow('result', img)
cv2.waitKey(0)