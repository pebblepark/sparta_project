from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# 얼굴 영역 탐지 모델
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# 마스크 판단 모델
model = load_model('models/mask_detector.model')

cap = cv2.VideoCapture('videos/04.mp4')

while True:
    ret, img = cap.read()

    if ret == False:
        break

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break