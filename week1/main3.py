import cv2

img = cv2.imread('01.jpg')
overlay_img = cv2.imread('dices.png', cv2.IMREAD_UNCHANGED)

overlay_img = cv2.resize(overlay_img, dsize=(150, 150))
# 주사위 이미지의 투명값 구하기
overlay_alpha = overlay_img[:, :, 3:] / 255.0
# 배경 이미지의 투명도 구하기 -> 주사위 이미지 투명값 반전
background_alpha = 1.0 - overlay_alpha

x1 = 100
y1 = 100
x2 = x1 + 150
y2 = y1 + 150

#  B  G  R  A(alpha)
# [0, 1, 2, 3]
img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]

cv2.imshow('img', img)
cv2.waitKey(0)