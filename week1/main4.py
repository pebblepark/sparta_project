import cv2

# cap = cv2.VideoCapture('04.mp4')
# 웹캠 연결
cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()

	# ret = frame 없거나 에러 발생 -> False
	if ret == False:
		break

	cv2.rectangle(img, pt1=(721, 183), pt2=(878, 465), color=(255,0,0), thickness=2)

	# 영상 그레이스케일로 변환
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# 영상 크기변환
	img = cv2.resize(img, dsize=(640, 360))

	# 영상 자르기(Crop)
	# img = img[100:200, 150: 250]

	cv2.imshow('result', img)

	# waitKey의 delay 많이 줄수록 속도 느려짐 (100 -> 100ms)
	if cv2.waitKey(100) == ord('q'):
		break