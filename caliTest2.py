import numpy as np
import cv2 as cv
import glob
import pickle
import os
import time
import math


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
wc = 7  # 체스 보드 가로 패턴 개수 - 1
hc = 7  # 체스 보드 세로 패턴 개수 - 1
objp = np.zeros((wc * hc, 3), np.float32)
objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)

prmt = []
mtx = 0
dist = 0
newcameramtx = 0

objpoints = []
imgpoints = []
grayShape = 0

cordi_x = -1

px_per_mm = ()

thres = 0




'''
조명고르게 분산 후 그림자 안에 있음
dst = edge_detection(gray, 1500, 500, 20)
'''

# files = glob.glob('iPhone13Pro_images/*.jpg')
# files = glob.glob('iPhone7_images/*.jpg')
# file = 'C:\\Users\\sampl\\PycharmProjects\\Opencv\\images\\right01.jpg'  ## 체스 보드 이미지


def existFile(fname, path='.'):
    listdir = os.listdir(os.getcwd() + '\\' + path)
    if fname in listdir:
        return True
    else:
        return False
    # print(listdir)


def savePickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def loadPickle(fname):
    with open(fname, 'rb') as fr:
        return pickle.load(fr)


def initCalib(img):
    # img = cv.imread(frame)
    #_img = cv.resize(img, dsize=(640, 480), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # gray scale로 바꾸기

    # cv.imshow('img', img)
    # cv.waitKey(0)

    ret, corners = cv.findChessboardCorners(gray, (wc, hc), None)  # 체스 보드 찾기
    # 만약 ret값이 False라면, 체스 보드 이미지의 패턴 개수를 맞게 했는지 확인하거나 (wc, hc)
    # 체스 보드가 깔끔하게 나온 이미지를 가져와야 한다
    global grayShape
    print('ret:', ret, len(objpoints))
    if ret == True:
        cv.imwrite('tabletChess{}.jpg'.format(len(objpoints)), img)
        objpoints.append(objp)

        # Canny86 알고리즘으로 도형이 겹치는 코너 점을 찾는다
        corners2 = cv.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 찾은 코너 점들을 이용해 체스 보드 이미지에 그려넣는다
        img = cv.drawChessboardCorners(img, (wc, hc), corners2, ret)
        # cv.imshow('img', img)
        grayShape = gray.shape[::-1]

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)  # 왜곡 펴기

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)

        # getOptimalNewCameraMatrix 함수를 쓰지 않은 이미지
        dst = cv.undistort(img, mtx, dist)
        dst2 = cv.undistort(img, mtx, dist, None, newcameramtx)  # 함수를 쓴 이미지

        #cv.imshow('num1', dst)
        # fname = frame.replace('images', 'calibraition')
        # cv.imwrite(fname, dst2)
        cv.imshow('num2', dst2)

    if len(objpoints) > 25:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)  # 왜곡 펴기
        savePickle('param.pckl', [mtx, dist])
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
        savePickle('newcameramtx.pckl', newcameramtx)

    time.sleep(0.4)


def loadCalib(img):
    global prmt, newcameramtx, mtx, dist
    #_img = cv.resize(img, dsize=(640, 480), interpolation=cv.INTER_AREA)

    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  ## 왜곡 펴기
    if not prmt and newcameramtx == 0:
        prmt = loadPickle('param.pckl')
        mtx, dist = prmt
        newcameramtx = loadPickle('newcameramtx.pckl')

    # dst = cv.undistort(img, mtx, dist)  ## getOptimalNewCameraMatrix 함수를 쓰지 않은 이미지
    dst2 = cv.undistort(img, mtx, dist, None, newcameramtx)  # 함수를 쓴 이미지

    #cv.imshow('num1', dst)
    #cv.imshow('num2', dst2)
    return side_slice(dst2)


def corner_detection(src, dst):
    corners = cv.goodFeaturesToTrack(dst, 80, 0.01, 10)
    # 실수 좌표를 정수 좌표로 변환
    print(np.shape(corners))
    corners = np.int32(corners)

    # 좌표에 동그라미 표시
    for corner in corners:
        x, y = corner[0]
        cv.circle(src, (x, y), 5, (0, 0, 255), 1, cv.LINE_AA)

    cv.imshow('Corners', src)
    return src


def edge_detection(gray, th1, th2, e):
    _, gray = cv.threshold(gray, e, 255, cv.THRESH_TOZERO)  # 이진 임계처리 (임계값 90)
    # 가우시안 블러 (src, ksize, sigmaX, sigmaY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    #gray = cv.medianBlur(gray, 17)
    #gray = cv.bilateralFilter(gray, 9, sigmaColor=75, sigmaSpace=75)
    #gray = cv.edgePreservingFilter(gray, flags=1, sigma_s=60, sigma_r=0.05)

    #cv.imshow('blur', gray)

    gray = cv.Canny(gray, th1, th2, apertureSize=5, L2gradient=True)

    #edged = cv.dilate(gray, None, iterations=1)
    #gray = cv.erode(edged, None, iterations=1)

    return gray


def side_slice(img):
    global cordi_x
    if cordi_x < 0:
        src = img.copy()
        gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
        corners = cv.goodFeaturesToTrack(
            gray, 100, 0.01, 300, blockSize=3, useHarrisDetector=True, k=0.03)
        cn = []
        for i in corners:
            cordi = tuple(map(lambda x: int(x), *i))
            cn.append(cordi[0])
            cn.append(cordi[1])
            cv.circle(src, cordi, 3, (0, 0, 255), 2)
        cordi_x = max(list(filter(lambda x: x < 100, cn))) + 3
    #print('side slice func')
    return img[cordi_x:480-cordi_x, cordi_x:640-cordi_x]


def contour_sideband(src, contours):
    for cnt in contours:
        for p in cnt:
            cv.circle(src, (p[0][0], p[0][1]), 1, (255, 0, 0), -1)

    cv.imshow('contour_sideband', src)
    return src


def contour_estimate(src, contours):
    approx = []
    for j, cnt in enumerate(contours):
        epsilon = 0.12 * cv.arcLength(cnt, True)
        approx.append(cv.approxPolyDP(cnt, epsilon, True))

        #cv.drawContours(src, [approx[j]], -1, (0, 0, 255), 1)

        for i, coord in enumerate(approx[j]):
            dot = tuple(*coord)
            # a4[i] = dot
            #a4_.append(dot)
            cv.circle(src, dot, 5, (255, 0, 255), 1, cv.LINE_AA)


    cv.imshow('contour_estimate', src)
    return approx


def line_detection(img, gray):
    lines = cv.HoughLinesP(gray, 0.9, np.pi / 180, 90, minLineLength=5, maxLineGap=50)

    if len(lines) > 0:
        for i in lines:
            cv.line(img, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
    else: return

    cv.imshow("line_detection", img)


def a4_init(src):
    global thres
    approx = []
    #while len(approx) < 4:
    #thres += 2
    dst = edge_detection(gray, 1500, 500, 20)

    src_ = src.copy()

    contours, _ = cv.findContours(dst.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow('a4 edge', dst)

    approx = contour_estimate(src_, contours)
    #print(thres, len(approx))

    #if thres > 255: thres = 0


    a4 = []
    a4_list = []
    rate = []
    a4_rate = []
    shape_rate = []
    approx = np.array(list(filter(lambda x: len(x) == 4, approx)))
    #print('approx shape:', np.shape(approx))
    if len(approx) == 0: return

    for cnt in approx:
        for p in cnt:
            cv.circle(src, (p[0][0], p[0][1]), 3, (0, 255, 255), -1)


    for cordi in approx:
        # dots type (n, 4)
        dots = np.array(list((map(lambda a: a[0], cordi))))
        #print(dots)
        abs_distance = []
        for i in range(len(dots)):
            x = abs(dots[i%4][0] - dots[(i+1)%4][0])
            y = abs(dots[i%4][1] - dots[(i+1)%4][1])
            #print(x, y, math.sqrt(x**2 + y**2))

            # 네 점마다의 거리와 가로세로 비율
            abs_distance.append([x, y, math.sqrt(x**2 + y**2)])
        #print('abs_distance', abs_distance)

        if len(abs_distance) == 4:

            # a4 대비 가로 세로 비율
            r1 = min(abs_distance[0][2], abs_distance[3][2]) / max(abs_distance[0][2], abs_distance[3][2])
            r2 = min(abs_distance[1][2], abs_distance[2][2]) / max(abs_distance[1][2], abs_distance[2][2])

            shape_rate.append(abs(r1 - r2))
            a4_rate.append(abs(np.mean([r1, r2]) - 0.70707070707))
            rate.append(a4_rate[-1] + shape_rate[-1])
    '''
    print('a4_rate:', a4_rate)
    print('shape_rate:', shape_rate)
    print('rate:', rate)
    #print('approx:', approx)
    print(len(shape_rate), len(a4_rate), len(approx))
    '''
    result = np.array([])

    if len(approx) > 0:
        for dot in approx:
            #print(a4_rate.index(min(a4_rate)), shape_rate.index(min(shape_rate)))

            #a4 = approx[a4_rate.index(min(a4_rate))]

            #dot = approx[a4_rate.index(min(a4_rate))]
            dot = np.array(list(map(lambda x:x[0], dot)))
            result = affin(src, dot)


            if str(type(result)) == "<class 'NoneType'>" : break
            elif len(result) == 0: break

            x, y = np.shape(result)[0], np.shape(result)[1]
            #print('{}, {}:'.format(x, y), min(x, y)/max(x, y))
            a4_list.append(abs(min(x, y)/max(x, y) - 0.70707070707))
            a4.append(dot)

        print(a4_list)
        if len(a4_list) == 0: return
        for xy in a4[a4_list.index(min(a4_list))]: cv.circle(src, tuple(xy), 5, (255, 0, 255), 2, cv.LINE_AA)
        if str(type(result)) != "<class 'NoneType'>": cv.imshow('affin', result)
    cv.imshow('src', src)


    '''

        global px_per_mm
        px_per_mm = [297/width, 210/height]
        return result
    #cv.imshow('a4 init', src)
    '''


def affin(src, dot):
    if len(dot) == 4:

        sm = dot.sum(axis=1)
        diff = np.diff(dot, axis=1)

        # print(sm, diff)

        topLeft = dot[np.argmin(sm)]  # x+y가 가장 작은 값이 좌상단 좌표
        bottomRight = dot[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
        topRight = dot[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
        bottomLeft = dot[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        w1 = abs(bottomRight[0] - bottomLeft[0])
        w2 = abs(topRight[0] - topLeft[0])
        h1 = abs(topRight[1] - bottomRight[1])
        h2 = abs(topLeft[1] - bottomLeft[1])
        width = int(max([w1, w2]))  # 두 좌우 거리간의 최대값이 서류의 폭
        height = int(max([h1, h2]))  # 두 상하 거리간의 최대값이 서류의 높이

        if not width or not height: return

        # elif width < height: width, height = height, width

        # print(width, height, width/height)

        # print(width, height)

        pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # 변환 행렬 계산
        mtrx = cv.getPerspectiveTransform(pts1, pts2)

        # 원근 변환 적용
        result = cv.warpPerspective(src, mtrx, (width, height))
        #print(height/width)
        #result = cv.resize(result, dsize=(297, 210), interpolation=cv.INTER_AREA)
        #cv.imshow('scanned', result)
        return result


if __name__ == '__main__':
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    i = 100
    j = 300
    e = 200
    while cv.waitKey(33) < 0:

        ret, frame = capture.read()
        if existFile('param.pckl') and existFile('newcameramtx.pckl'):
            cali = loadCalib(frame)  # cali color
            cv.imwrite('cali.jpg', cali)
            gray = cv.cvtColor(cali, cv.COLOR_BGR2GRAY)

            dst = a4_init(gray)

            #print(px_per_mm)

            edge = edge_detection(gray, 1500, 500, 143)

            #cv.imshow('edge', edge)

            ''' 
            e -= 5
            print(i, j, e)
            j += 50
            if j >= 2000:
                i += 50
                j = 0
                if i >= 7000:
                    i = 0
            '''

            contours, hierarchy = cv.findContours(edge.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            #sided = contour_sideband(cali, contours)
            #line_detection(cali, dst.copy())


            #contour_estimate(cali, contours)

            #corner_detection(cali, gray)
            cv.imshow("result", edge)
            cv.imwrite('cardtest.jpg', edge)

        else:
            initCalib(frame)
            cv.waitKey(25)

        if len(objpoints) > 25:
            break

    capture.release()
    cv.destroyAllWindows()
