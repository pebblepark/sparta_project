# [스파르타코딩클럽] 이미지처리로 시작하는 딥러닝

https://www.notion.so/bce413d8987d45f0bb94a9a24e429935

# 목차

- [Week1](#Week1)
- [Week5](#Week5)

# Week1

## 이미지 처리 기초

### 이미지 불러오기

```py
import cv2
img = cv.imread('01.jpg')
print(img.shape) # (404, 640, 3) = (높이, 너비, 채널)
```

### 이미지 띄우기

```py
cv2.imshow('result', img) # 첫번째 인자: window 창 이름
cv2.waitKey(0) # 아무 키나 입력할 때까지 창 띄움
```

### 사각형 그리기

```py
# img: 사각형 그릴 이미지, pt1: 사각형 좌측상단 좌표, pt2: 사각형 우측하단 좌표, color: 사각형 색깔(BGR), thickness: 도형선 두께(-1은 내부 채우기)
cv2.rectangle(img, pt1=(259, 89), pt2=(380, 348), color=(255, 0, 0), thickness=2)
```

### 원 그리기

```py
# center: 중심좌표, radius: 반지름
cv2.circle(img, center=(320, 220), radius=100, color=(255, 0, 0), thickness=2)
```

### 이미지 자르기

- 이미지를 자를 때는 **y, x 순서**

```py
cropped_img = img[89:348, 259:380] # y축으로 89에서 348까지 자르고 x축으로 259에서 380까지 자름
cv2.imshow('cropped', cropped_img)
```

### 이미지 크기 변경

```py
img_resized = cv2.resize(img, dsize=(512, 256)) # 가로 512, 세로 256
```

### 이미지 컬러 시스템 변경

```py
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 에서 RGB로 변경
img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # RGB 에서 그레이스케일로 변경
```

## 이미지 처리 심화

### 오버레이 띄우기

- 오버레이 이미지는 확장자가 .png인 배경이 투명한 이미지
- `cv2.imread()`를 사용하여 png 이미지 로드 -> **`cv2.IMREAD_UNCHANGED` 붙여줘야 투명도 유지한채로 로드**됨
- 투명도가 있는 이미지는 채널이 BGR에 A(alpha)까지 더해 채널이 총 4개

```py
import cv2

img = cv2.imread('01.jpg')
overlay_img = cv2.imread('dices.png', cv2.IMREAD_UNCHANGED)
```

### 오버레이 이미지

- 오버레이 이미지 : ( 배경 + 이미지 )
- 배경 이미지: 오버레이 이미지의 투명 부분은 배경 이미지 자체 유지, 오버레이 이미지의 이미지 있는 부분은 투명하기
  - 오버레이 이미지 투명도 반전시키기 (1.0 - overlay_alpha)

```py
overlay_alpha = overlay_img[:, :, 3:] / 255.0 # 투명도 채널만 분리함, 0~1
background_alpha = 1.0 - overlay_alpha # 배경이미지의 투명도 구함
```

### 이미지 합성하기

- 오버레이 이미지 합성: 배경 이미지 + 오버레이 이미지

```py
x1 = 100
y1 = 100
x2 = x1 + 150
y2 = y1 + 150

# overlay_alpha 곱해주면 투명하지 않은 부분만 가져옴 -> overlay_img[:, :, :3] BGR 채널만 가져오기
img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]
```

## 동영상 처리

### 동영상 플레이어

```py
import cv2

cap = cv2.VideoCapture('04.mp4')

while True:
	ret, img = cap.read()

	if ret == False:
		break

	cv2.imshow('result', img)

	if cv2.waitKey(1) == ord('q'):
		break
```

# Week5

## 그레이스케일 사진에 색 입히기

### 이미지 전처리

```py
img = cv2.imread('imgs/02.jpg')

h, w, c = img.shape

img_input = img.copy()

# img_input의 타입을 uint8(unsigned integer 8bit) -> float32(floating point 32 bit)로 변경
# 딥러닝 모델 학습시 99% 이상 소수점으로 학습시킴
img_input = img_input.astype('float32') / 255.

# 이미지 컬러 시스템 BRR -> Lab 변경
img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2Lab)

# L 채널만 추출
img_l = img_lab[:, :, 0:1]

# resize해주고 mean값 빼주고 차원변형
blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])
```

### 결과 추론

```py
net.setInput(blob)
output = net.forward()
```

### 결과 후처리

```py
# squeeze: 차원 축소, transpose: 인간이 이해할 수 있도록 차원 변형
output = output.squeeze().transpose((1, 2, 0))

# 줄어든 이미지를 원본 이미지 크기로 되돌리기
output_resized = cv2.resize(output, (w, h))

# 0: 세로, 1: 가로, 2: 채널 방향으로 합쳐줌
output_lab = np.concatenate([img_l, output_resized], axis=2)

# 컬러시스템 Lab -> BGR
output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)

# 전처리에서 나눠준 255를 후처리에서 곱해줌(전처리 반대 -> 후처리)
output_bgr = output_bgr * 255

# 255를 넘는 값을 clip으로 잘라냄(0~255)
output_bgr = np.clip(output_bgr, 0, 255)

# 전처리시 float32로 변경해준 것을 정수형태로 다시 변경
output_bgr = output_bgr.astype('uint8')
```

## 특정부분만 컬러로 합성하기

### 마스크 만들기

```py
# 0으로 채운 이미지 만들기 -> 사이즈랑 채널 다 동일한데 brg이 0 -> 검은색
mask = np.zeros_like(img, dtype='uint8')
# mask에 원으로 마스킹 -> 마스킹한 부분만 (1,1,1)로 채우기
mask = cv2.circle(mask, center=(260, 260), radius=200, color=(1, 1, 1), thickness=-1)
```

### 이미지 합성

```py
# 마스크 한 부분 -> 컬러 복원
color = output_bgr * mask
# 마스크 안한 부분 -> 그레이스케일(img) * mask 반전(1 - mask)
gray = img * (1 - mask)

output2 = color + gray

cv2.imshow('result2', output2)
```

## 해상도 향상 시키기

```py
import cv2

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/EDSR_x3.pb')
# 해상도 3배 향상 -> 원본이미지 크기 100x100 사이즈였으면 300x300
sr.setModel('edsr', 3)

img = cv2.imread('imgs/06.jpg')

# img 해상도 3배 늘린 결과 result에 저장
result = sr.upsample(img)

# 원하는 사이즈 픽셀지정(dsize) x -> x축 y축 3배 늘림
resized_img = cv2.resize(img, dsize=None, fx=3, fy=3)

cv2.imshow('img', img)
cv2.imshow('reuslt', result)
cv2.imshow('resized_img', resized_img)
cv2.waitKey(0)
```
