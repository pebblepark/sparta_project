import cv2
import dlib

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture('videos/01.mp4')
sticker_img = cv2.imread('imgs/sticker01.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)
    print("number of faces detected:", len(dets))

    for det in dets:
        x1 = det.left()
        y1 = det.top()
        x2 = det.right()
        y2 = det.bottom()

        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255,0,0), thickness=2)

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break
