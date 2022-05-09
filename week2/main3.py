import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/instance_norm/feathers.t7')

img = cv2.imread('imgs/hw.jpg')

h, w, c = img.shape

cropped_img = img[144:368, 480:812]
# img = cv2.resize(img, dsize=(500, int(h / w * 500)))

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(cropped_img, mean=MEAN_VALUE)

net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0))

output += MEAN_VALUE
output = np.clip(output, 0, 255)
output = output.astype('uint8')

img[144:368, 480:812] = output

cv2.imshow('img', img)
cv2.waitKey(0)