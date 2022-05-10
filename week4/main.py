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
        x1 = det.left() - 40
        y1 = det.top() - 50
        x2 = det.right() + 40
        y2 = det.bottom() + 30

        # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255,0,0), thickness=2)

        try:
            overlay_img = sticker_img.copy()

            overlay_img = cv2.resize(overlay_img, dsize=(x2 - x1, y2 - y1))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]

        except:
            pass

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break
