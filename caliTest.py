import numpy as np
import cv2 as cv
import glob
import pickle
import os
import time
import math

thres = 0

'''
조명고르게 분산 후 그림자 안에 있음
dst = edge_detection(gray, 1500, 500, 20)
'''


# files = glob.glob('iPhone13Pro_images/*.jpg')
# files = glob.glob('iPhone7_images/*.jpg')
# file = 'C:\\Users\\sampl\\PycharmProjects\\Opencv\\images\\right01.jpg'  ## 체스 보드 이미지


class Detect_object:
    def __init__(self):
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.wc = 7  # 체스 보드 가로 패턴 개수 - 1
        self.hc = 7  # 체스 보드 세로 패턴 개수 - 1
        self.objp = np.zeros((self.wc * self.hc, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.wc, 0:self.hc].T.reshape(-1, 2)

        self.prmt = []
        self.mtx = 0
        self.dist = 0
        self.newcameramtx = 0

        self.objpoints = []
        self.imgpoints = []
        self.grayShape = 0

        self.cordi_x = -1

        self.px_per_mm = ()

        self.thres = 0

        self.shape_rate = []
        self.a4_rate = []
        self.a4_list = []

    def existFile(self, fname, path='.'):
        listdir = os.listdir(os.getcwd() + '\\' + path)
        if fname in listdir:
            return True
        else:
            return False

    def savePickle(self, fname, obj):
        with open(fname, 'wb') as f: pickle.dump(obj, f)

    def loadPickle(self, fname):
        with open(fname, 'rb') as fr: return pickle.load(fr)

    def initCalib(self, img):
        # img = cv.imread(frame)
        # _img = cv.resize(img, dsize=(640, 480), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # gray scale로 바꾸기

        # cv.imshow('img', img)

        ret, corners = cv.findChessboardCorners(gray, (self.wc, self.hc), None)  # 체스 보드 찾기
        # 만약 ret값이 False라면, 체스 보드 이미지의 패턴 개수를 맞게 했는지 확인하거나 (wc, hc)
        # 체스 보드가 깔끔하게 나온 이미지를 가져와야 한다

        print('ret:', ret, len(self.objpoints))
        if ret == True:
            cv.imwrite('tabletChess{}.jpg'.format(len(self.objpoints)), img)
            self.objpoints.append(objp)

            # Canny86 알고리즘으로 도형이 겹치는 코너 점을 찾는다
            corners2 = cv.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
            self.imgpoints.append(corners2)

            # 찾은 코너 점들을 이용해 체스 보드 이미지에 그려넣는다
            img = cv.drawChessboardCorners(img, (wc, hc), corners2, ret)
            # cv.imshow('img', img)
            self.grayShape = gray.shape[::-1]

            ret, self.mtx, self.dist, rvecs, tvecs = cv.calibrateCamera(
                objpoints, self.imgpoints, gray.shape[::-1], None, None)  # 왜곡 펴기

            h, w = img.shape[:2]
            self.newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1)

            # getOptimalNewCameraMatrix 함수를 쓰지 않은 이미지
            dst = cv.undistort(img, self.mtx, self.dist)
            dst2 = cv.undistort(img, self.mtx, self.dist, None, self.newcameramtx)  # 함수를 쓴 이미지

            # cv.imshow('num1', dst)
            # fname = frame.replace('images', 'calibraition')
            # cv.imwrite(fname, dst2)
            cv.imshow('num2', dst2)

        if len(objpoints) > 25:
            ret, self.mtx, self.dist, rvecs, tvecs = cv.calibrateCamera(
                self.objpoints, self.imgpoints, gray.shape[::-1], None, None)  # 왜곡 펴기
            self.savePickle('param.pckl', [mtx, dist])
            h, w = img.shape[:2]
            self.newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
            self.savePickle('newcameramtx.pckl', self.newcameramtx)

        time.sleep(0.4)

    def loadCalib(self, img):
        # _img = cv.resize(img, dsize=(640, 480), interpolation=cv.INTER_AREA)

        # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  ## 왜곡 펴기
        if not self.prmt and self.newcameramtx == 0:
            self.prmt = self.loadPickle('param.pckl')
            self.mtx, self.dist = self.prmt
            self.newcameramtx = self.loadPickle('newcameramtx.pckl')

        # dst = cv.undistort(img, mtx, dist)  ## getOptimalNewCameraMatrix 함수를 쓰지 않은 이미지
        dst2 = cv.undistort(img, self.mtx, self.dist, None, self.newcameramtx)  # 함수를 쓴 이미지

        # cv.imshow('num1', dst)
        # cv.imshow('num2', dst2)
        return self.side_slice(dst2)

    def corner_detection(self, src, dst):
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

    def edge_detection(self, gray, th1, th2, e):
        _, gray = cv.threshold(gray, e, 255, cv.THRESH_TOZERO)  # 이진 임계처리 (임계값 90)
        # 가우시안 블러 (src, ksize, sigmaX, sigmaY)
        gray = cv.GaussianBlur(gray, (7, 7), 0)
        # gray = cv.medianBlur(gray, 17)
        # gray = cv.bilateralFilter(gray, 9, sigmaColor=75, sigmaSpace=75)
        # gray = cv.edgePreservingFilter(gray, flags=1, sigma_s=60, sigma_r=0.05)

        # cv.imshow('blur', gray)

        gray = cv.Canny(gray, th1, th2, apertureSize=5, L2gradient=True)

        # edged = cv.dilate(gray, None, iterations=1)
        # gray = cv.erode(edged, None, iterations=1)

        return gray

    def side_slice(self, img):
        if self.cordi_x < 0:
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
            self.cordi_x = max(list(filter(lambda x: x < 100, cn))) + 3
        # print('side slice func')
        return img[self.cordi_x:480 - self.cordi_x, self.cordi_x:640 - self.cordi_x]

    def contour_estimate(self, src, contours):
        approx = []
        for j, cnt in enumerate(contours):
            epsilon = 0.12 * cv.arcLength(cnt, True)
            approx.append(cv.approxPolyDP(cnt, epsilon, True))

            # cv.drawContours(src, [approx[j]], -1, (0, 0, 255), 1)

            for i, coord in enumerate(approx[j]):
                dot = tuple(*coord)
                # a4[i] = dot
                # a4_.append(dot)
                # cv.circle(src, dot, 5, (255, 0, 255), 1, cv.LINE_AA)

        # cv.imshow('contour_estimate', src)
        return approx

    def line_detection(self, img, gray):
        lines = cv.HoughLinesP(gray, 0.9, np.pi / 180, 90, minLineLength=5, maxLineGap=50)
        if len(lines) > 0:
            for i in lines: cv.line(img, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
        else:
            return

        cv.imshow("line_detection", img)

    def a4_init(self, src):
        global thres
        approx = []
        # thres += 2
        dst = self.edge_detection(gray, 1500, 500, 20)

        src_ = src.copy()

        contours, _ = cv.findContours(dst.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        approx = self.contour_estimate(src_, contours)

        cv.imshow('a4 edge', dst)

        # if thres > 255: thres = 0

        cordi = []
        rate = []
        self.a4_list = []
        self.a4_rate = []
        self.shape_rate = []
        res = []
        del_list = []

        approx = np.array(list(filter(lambda x: len(x) == 4, approx)))  # only square
        if len(approx) == 0: return

        approx = self.del_similar(approx)

        # perspective 변환, 잘못된 변환(0, n)의 인덱스들 삭제
        for i, dot in enumerate(approx):
            cordi.append(np.array(list(map(lambda a: a[0], dot))))  # dots shape (n, 4)
            result = self.perspective(src, cordi[i])
            if str(type(result)) == "<class 'NoneType'>": continue
            res.append(result)
            if len(result) == 0:
                del_list.append(i)
                continue

            x, y = np.shape(result)[0], np.shape(result)[1]

            if x + y < 20 or x == 0 or y == 0:
                del_list.append(i)
                continue

            self.a4_list.append(abs(min(x, y) / max(x, y) - 0.70707070707070707070707070707071))

            # print(x, y, self.a4_list[i])    # result 해상도

            for p in cordi[i]: cv.circle(src_, (p[0], p[1]), 2, (0, 255, 255), -1)  # 검은색 점

            print(cordi[i][0][0], cordi[i][0][1], x,
                  y)  # , round(self.shape_rate[i], 3), round(self.a4_rate[i], 3), round(self.a4_list[i], 3))
        print('-----')

        approx = list(approx)

        # 원치 않은 컨투어 삭제
        for i in range(len(del_list) - 1, -1, -1):
            del approx[i]
            del cordi[i]
            del res[i]

        for i, dot in enumerate(cordi):
            abs_distance = []

            # print(dot)
            for j in range(4):
                x = abs(dot[j % 4][0] - dot[(j + 1) % 4][0])
                y = abs(dot[j % 4][1] - dot[(j + 1) % 4][1])
                # print(x, y, math.sqrt(x**2 + y**2))

                abs_distance.append([x, y, math.sqrt(x ** 2 + y ** 2)])  # 네 점마다의 거리와 가로세로 비율

            # print('abs_distance', abs_distance)

            if len(abs_distance) == 4:
                # a4 대비 가로 세로 비율
                r1 = max(abs_distance[0][2], abs_distance[3][2]) - min(abs_distance[0][2], abs_distance[3][2])
                r2 = max(abs_distance[1][2], abs_distance[2][2]) - min(abs_distance[1][2], abs_distance[2][2])

                self.shape_rate.append(abs(r1 + r2))

                r1 = min(abs_distance[0][2], abs_distance[3][2]) / max(abs_distance[0][2], abs_distance[3][2])
                r2 = min(abs_distance[1][2], abs_distance[2][2]) / max(abs_distance[1][2], abs_distance[2][2])

                self.a4_rate.append(np.mean([r1, r2]) - 0.70707070707)
                rate.append(self.a4_rate[-1] + self.shape_rate[-1])

            try:
                x, y = np.shape(res[i])[0], np.shape(res[i])[1]
            except:
                # print('except')
                # print(res[i])
                return

            cv.putText(src_, '{0} {1} {2} {3} {4}'.format(x, y, round(self.shape_rate[i], 3), round(self.a4_rate[i], 3),
                                                          round(self.a4_list[i], 3)), (dot[0][0], dot[0][1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if len(self.shape_rate) == 0: return
        a4 = res[self.shape_rate.index(max(self.shape_rate))]

        # print(len(approx), len(cordi), len(res), len(self.shape_rate), len(self.a4_rate), len(rate))
        for xy in cordi[self.shape_rate.index(max(self.shape_rate))]:
            cv.circle(src_, tuple(xy), 5, (255, 0, 255), 2, cv.LINE_AA)

        self.px_per_mm = [297 / max(np.shape(a4)[0], np.shape(a4)[1]), 210 / min(np.shape(a4)[0], np.shape(a4)[1])]
        # print(self.px_per_mm)

        cv.imshow('a4', a4)
        cv.imshow('src', src_)
        cv.imwrite('src.jpg', src_)

        return a4

    def perspective(self, src, dot):
        if len(dot) == 4:
            sm = dot.sum(axis=1)
            diff = np.diff(dot, axis=1)

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

            if not width or not height:
                print('width or height is not')
                print(w1, w2, h1, h2, sm, diff)
                print(dot)

                return []

            pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

            # 변환 행렬 계산
            mtrx = cv.getPerspectiveTransform(pts1, pts2)

            # 원근 변환 적용
            result = cv.warpPerspective(src, mtrx, (width, height))
            # result = cv.resize(result, dsize=(297, 210), interpolation=cv.INTER_AREA)
            # cv.imshow('scanned', result)
            return result

    def object_sizing(self, src):
        edge = self.edge_detection(src, 1000, 500, 20)
        cv.imshow('a4 edge', edge)

        contours, _ = cv.findContours(edge.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        approx = self.contour_estimate(src, contours)

        approx = np.array(list(filter(lambda x: not (len(x) % 2), approx)))
        # print(approx)
        # print(np.shape(approx))

        approx = self.del_similar(approx)
        if str(type(approx)) == "<class 'NoneType'>": return
        # print(approx)
        approx_ = []
        abs_distance = []
        for i, cordi in enumerate(approx):
            cordi = np.array(list(map(lambda x: x[0], cordi)))

            # 점뿐인 좌표는 배제
            if len(cordi) < 2: continue

            # print(cordi)
            n = len(cordi)
            distance = []
            for i in range(n - 1):
                x = abs(cordi[i % n][0] - cordi[(i + 1) % n][0])
                y = abs(cordi[i % n][1] - cordi[(i + 1) % n][1])

                # 네 점마다의 거리와 가로세로 비율
                distance.append([x, y, math.sqrt(x ** 2 + y ** 2)])
            abs_distance.append(distance)
            # 길이가 매우 짧은 것은 배제
            # if abs_distance[i][2] < 5: continue

            # print(abs_distance, self.px_per_mm)
            # print(abs_distance[i][0], abs_distance[i][1], abs_distance[i][2], abs_distance[i][2]*self.px_per_mm[0])

            approx_.append(cordi)

        # print(len(approx), len(approx_), len(abs_distance))
        approx_n = len(approx_)
        del_list = []

        '''
        for i in range(approx_n-1):
            dif = np.abs(approx_[i] - approx_[i+1])
            dstc = np.abs(np.diff(dif, axis=0))
            lenth = math.sqrt(dstc[0][0]**2 + dstc[0][1]**2)
            if lenth < 5:
                if abs_distance[i][2] < abs_distance[i+1][2]:
                    del_list.append(i)

        for i in range(len(del_list)-1, -1, -1):
            del approx_[i]
            del abs_distance[i]
        '''

        for i, cordi in enumerate(approx_):
            a = round(abs_distance[i][0][2] * self.px_per_mm[0], 2)
            b = round(abs_distance[i][0][2] * self.px_per_mm[1], 2)
            # cv.putText(src, '{0} {1} {2} {3} {4}'.format(abs_distance[i][0], abs_distance[i][1], round(abs_distance[i][2], 2), round(abs_distance[i][2]*self.px_per_mm[0], 2), round(abs_distance[i][2]*self.px_per_mm[1], 2)), (int(cordi[0][0]+(abs_distance[i][0]/2)+2), int(cordi[0][1]+(abs_distance[i][1]/2))), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            cv.putText(src, '{} {} {}'.format(a, b, (a + b) / 2),
                       (int(cordi[0][0] + (abs_distance[i][0][0] / 2) + 2),
                        int(cordi[0][1] + (abs_distance[i][0][1] / 2))),
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            cv.line(src, (cordi[0][0], cordi[0][1]), (cordi[1][0], cordi[1][1]), (255, 0, 255), 1)
            for xy in cordi:
                cv.circle(src, tuple(xy), 1, (255, 0, 255), 1)

        # print('------')
        # print(approx_)

        cv.imshow('a4 src', src)
        return approx_

    def del_similar(self, approx):
        approx_n = len(approx)
        if approx_n == 0: return
        # print(np.shape(approx))
        # print(approx)
        diff_approx = []
        del_list = set()
        for cordi in approx: diff_approx.append(np.sort(cordi, axis=0))

        for i in range(len(diff_approx) - 1):
            if len(diff_approx[i]) != len(diff_approx[i + 1]): continue
            dlt = False
            diff = np.abs(diff_approx[i] - diff_approx[i + 1])
            # print(diff)
            dots = []
            for j in range(len(diff) - 1):
                dot = np.abs(diff[j][0] - diff[j + 1][0])
                dots.append(dot)
            for j in range(len(dots) - 1):
                d = np.abs(dots[j] - dots[j + 1])
                if math.sqrt(d[0] ** 2 + d[1] ** 2) < 5:
                    dlt = True
                else:
                    dlt = False
            if dlt: del_list.add(i)

        del_list = sorted(del_list, reverse=True)
        approx_ = []

        for i in range(approx_n):
            if i in del_list: continue
            approx_.append(np.array(approx[i]))
        # print(np.shape(approx_))
        # print(approx_)

        return approx_


if __name__ == '__main__':
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    dtob = Detect_object()

    i = 100
    j = 300
    e = 200
    while cv.waitKey(33) < 0:

        ret, frame = capture.read()
        if dtob.existFile('param.pckl') and dtob.existFile('newcameramtx.pckl'):
            cali = dtob.loadCalib(frame)  # cali color
            # cv.imwrite('cali.jpg', cali)
            gray = cv.cvtColor(cali, cv.COLOR_BGR2GRAY)

            a4_scaned = dtob.a4_init(gray)
            if str(type(a4_scaned)) == "<class 'NoneType'>": continue

            dtob.object_sizing(a4_scaned)

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

            # contours, hierarchy = cv.findContours(edge.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            # sided = contour_sideband(cali, contours)
            # line_detection(cali, dst.copy())

            # contour_estimate(cali, contours)

            # corner_detection(cali, gray)
            # cv.imshow("result", edge)
            # cv.imwrite('cardtest.jpg', edge)

        else:
            dtob.initCalib(frame)
            cv.waitKey(25)

        if len(dtob.objpoints) > 25:
            break

    capture.release()
    cv.destroyAllWindows()
