import cv2

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/EDSR_x3.pb')
sr.setModel('edsr', 3)

img = cv2.imread('imgs/06.jpg')

result = sr.upsample(img)

resized_img = cv2.resize(img, dsize=None, fx=3, fy=3)

cv2.imshow('img', img)
cv2.imshow('reuslt', result)
cv2.imshow('resized_img', resized_img)
cv2.waitKey(0)