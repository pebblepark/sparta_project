import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/02.mp4')
sticker_img = cv2.imread('imgs/pig.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)

    for det in dets:
        shape = predictor(img, det)

        try:
            center_x = shape.parts()[4].x
            center_y = shape.parts()[4].y

            # print('%s: %d, %d' % ('center', center_x, center_y))
            # cv2.circle(img, center=(center_x, center_y), radius=2, color=(0, 0, 255), thickness=-1)

            h, w, c = sticker_img.shape

            pig_w = int((det.right() - det.left())/3)
            pig_h = int(h / w * pig_w)

            pig_x1 = int(center_x - pig_w/2)
            pig_x2 = pig_x1 + pig_w

            pig_y1 = int(center_y - pig_h/2) - 10
            pig_y2 = pig_y1 + pig_h

            # print('[%d, %d] : [%d, %d] - %d, %d' % (pig_x1, pig_y1, pig_x2, pig_y2, pig_w, pig_h))

            overlay_img = sticker_img.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(pig_w, pig_h))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            img[pig_y1:pig_y2, pig_x1:pig_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[pig_y1:pig_y2, pig_x1:pig_x2]

        except:
            pass

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break