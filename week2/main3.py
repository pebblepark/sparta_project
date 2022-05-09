import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/instance_norm/starry_night.t7')

cap = cv2.VideoCapture('imgs/03.mp4')

while True:
	ret, img = cap.read()

	if ret == False:
		break

	h, w, c = img.shape
	img = cv2.resize(img, dsize=(int(w / h * 300), 300))

	MEAN_VALUE = [103.939, 116.779, 123.680]
	blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

	net.setInput(blob)
	output = net.forward()

	output = output.squeeze().transpose((1, 2, 0))

	output += MEAN_VALUE
	output = np.clip(output, 0, 255)
	output = output.astype('uint8')

	cv2.imshow('result', output)

	if cv2.waitKey(1) == ord('q'):
		break