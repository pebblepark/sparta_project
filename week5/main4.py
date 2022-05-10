import cv2
import numpy as np

proto = 'models/colorization_deploy_v2.prototxt'
weights = 'models/colorization_release_v2.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto, weights)

pts_in_hull = np.load('models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

cap = cv2.VideoCapture('videos/02.mp4')

while True:
    ret, img = cap.read()

    if ret == False:
        break

    h, w, c = img.shape

    img = cv2.resize(img, dsize=(int(w/h*500), 500))

    h, w, c = img.shape
    img_input = img.copy()

    img_input = img_input.astype('float32') / 255.
    img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2Lab)
    img_l = img_lab[:, :, 0:1]

    blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])

    net.setInput(blob)
    output = net.forward()

    output = output.squeeze().transpose((1, 2, 0))

    output_resized = cv2.resize(output, (w, h))

    output_lab = np.concatenate([img_l, output_resized], axis = 2)

    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
    output_bgr = output_bgr * 255
    output_bgr = np.clip(output_bgr, 0, 255)
    output_bgr = output_bgr.astype('uint8')

    cv2.imshow('output', output_bgr)
    if cv2.waitKey(1) == ord('q'):
        break



