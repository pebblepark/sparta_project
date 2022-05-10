import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture('videos/01.mp4')
sticker_img = cv2.imread('imgs/glasses.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)

    for det in dets:
        shape = predictor(img, det)

        try:
            # for i, point in enumerate(shape.parts()):
            #     # thickness = -1 -> 안이 채워진 원
            #     cv2.circle(img, center=(point.x, point.y), radius=2, color=(0, 0, 255), thickness=-1)
            #     cv2.putText(img, text=str(i), org=(point.x, point.y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)

            glasses_x1 = shape.parts()[2].x - 20    # 왼쪽 눈꼬리
            glasses_x2 = shape.parts()[0].x + 20    # 오른쪽 눈꼬리

            h, w, c = sticker_img.shape

            glasses_w = glasses_x2 - glasses_x1
            glasses_h = int(h / w * glasses_w)

            center_y = (shape.parts()[0].y + shape.parts()[2].y) / 2

            glasses_y1 = int(center_y - glasses_h / 2)
            glasses_y2 = glasses_y1 + glasses_h

            overlay_img = sticker_img.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(glasses_w, glasses_h))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            img[glasses_y1:glasses_y2, glasses_x1:glasses_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[glasses_y1:glasses_y2, glasses_x1:glasses_x2]

        except:
            pass    # 얼굴 영역이 이미지 밖으로 벗어날 경우 발생하는 에러 방지

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break