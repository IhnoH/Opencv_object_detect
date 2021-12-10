import cv2
import os
import pickle
import sys
import numpy as np
import glob

test = np.array(
[[[[ 80, 412]],
 [[ 70, 445]],
 [[ 83, 411]],
 [[110, 445]]],

 [[[276, 305]],
 [[277, 307]],
 [[275, 308]],
 [[274, 307]]],

 [[[277, 305]],
 [[274, 306]],
 [[275, 309]],
 [[278, 307]]],

 [[[249, 276]],
 [[288, 307]],
 [[262, 342]],
 [[282, 305]]],

 [[[187, 214]],
 [[227, 229]],
 [[200, 314]],
 [[159, 301]]],

 [[[187, 214]],
 [[159, 302]],
 [[201, 314]],
 [[227, 228]]],

 [[[157, 175]],
 [[340, 258]],
 [[280, 387]],
 [[ 98, 303]]],

 [[[156, 175]],
 [[ 98, 304]],
 [[281, 387]],
 [[340, 258]]]]
)

src = cv2.imread('cali.jpg')
rate = []

for i, cordi in enumerate(test):
    squr = np.zeros((4, 2), dtype=np.float32)
    dot = np.array(list((map(lambda x: x[0], cordi))))
    src_ = src.copy()
    print(i, dot)
    #map(lambda x: cv2.circle(src_, tuple(x), 5, (255, 0, 255), 2, cv2.LINE_AA), dot)

    x = abs(dot[2][0] - dot[0][0])
    y = abs(dot[2][1] - dot[0][1])

    print(x, y, min(x, y)/max(x, y))
    rate.append(abs(0.70707 - (min(x, y)/max(x, y))))

    for i, x in enumerate(dot): cv2.circle(src_, tuple(x), 5, (70*i, 0, 0), 2, cv2.LINE_AA)

    #cv2.imshow('result', src_)
    #cv2.waitKey(0)

print(min(rate))
a4 = test[rate.index(min(rate))]

for dot in a4:
    cv2.circle(src, tuple(dot[0]), 5, (255, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('result', src_)
cv2.waitKey(0)




'''

files = glob.glob('images2/*.png')
for f in files:
    img = cv2.imread(f)
    img2 = img.copy()

    # 그레이 스케일로 변환 ---①
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 스레시홀드로 바이너리 이미지로 만들어서 검은배경에 흰색전경으로 반전 ---②
    ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

    # 가장 바깥쪽 컨투어에 대해 모든 좌표 반환 ---③
    contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 가장 바깥쪽 컨투어에 대해 꼭지점 좌표만 반환 ---④
    contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 각각의 컨투의 갯수 출력 ---⑤
    print('도형의 갯수: %d(%d)'% (len(contour), len(contour2)))

    # 모든 좌표를 갖는 컨투어 그리기, 초록색  ---⑥
    #cv2.drawContours(img, contour, -1, (0,255,0), 3)
    # 꼭지점 좌표만을 갖는 컨투어 그리기, 초록색  ---⑦
    cv2.drawContours(img2, contour2, -1, (0,255,0), 3)

    
    # 컨투어 모든 좌표를 작은 파랑색 점(원)으로 표시 ---⑧
    for i in contour:
        for j in i:
            cv2.circle(img, tuple(j[0]), 1, (255,0,0), -1)

    # 컨투어 꼭지점 좌표를 작은 파랑색 점(원)으로 표시 ---⑨
    
    for i in contour2:
        for j in i:
            cv2.circle(img2, tuple(j[0]), 1, (255,0,0), -1)
    
    # 결과 출력 ---⑩
    cv2.imshow('CHAIN_APPROX_NONE', img)
    cv2.imshow('CHAIN_APPROX_SIMPLE', img2)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    
    
'''