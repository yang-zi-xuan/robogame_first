"""
1.单目测距
利用相机内参数矩阵去畸变，然后单目测距。
认为看见的方块轮廓面积为一个面面积的一倍
2.物体方向估计
去畸变后，利用像素坐标和内参数矩阵估算物体相对相机的方向，表示为(x, y, z)
"""

import numpy as np
import cv2
import math

cap = cv2.VideoCapture(0)  # 视频对象
fourcc = cv2.VideoWriter.fourcc(*"XVID")  # 编码器
out = cv2.VideoWriter("C:\\Users\\yb028028\\Desktop\\my_video.avi", fourcc, 30.00, (640, 480))  # 视频输出对象
high = np.array([50, 255, 255])
low = np.array([0, 100, 60])  # 颜色识别阈值
L = 10.1
S = L*L*1  # 1.5个面的面积，单位平方厘米


class CamSet(object):
    def __init__(self):
        self.CameraMatrix = np.array([[543.2352805344083, 0.0, 335.6325913504112],
                                      [0, 540.0090338378359, 237.52912533756933],
                                      [0, 0, 1]])
        self.DistCoeffs = np.array([-0.041107079237683884, 0.09952660331229725,
                                    -0.002757530806781375, 0.0024557170640417568,
                                    0.1088243671433803])

    def CorrectImage(self, img):
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.CameraMatrix, self.DistCoeffs,(w, h),
                                                             1, (w, h), 0)
        # 计算无畸变和修正转换关系
        mapx, mapy = cv2.initUndistortRectifyMap(self.CameraMatrix, self.DistCoeffs, None, newCameraMatrix, (w, h),
                                                 cv2.CV_16SC2)
        # 重映射 输入是矫正后的图像
        CorrectImg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        return CorrectImg


def GetMask(img, low, high):
    """new_width = 300
    new_height = 200
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0, None)  # 高斯模糊
    mask = cv2.inRange(img, low, high)  # 生成掩膜

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)  # 对掩膜进行腐蚀和膨胀处理


    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = list()
    area = [cv2.contourArea(contours[i]) for i in range(len(contours))]  # 选择面积最大的掩膜作为识别到的对象（则一次只能识别一个）
    if len(area) > 0:
        contour.append(contours[area.index(max(area))])
        epsilon = 0.0135 * cv2.arcLength(contour[0], True)  # 根据轮廓周长计算逼近精度
        approx = cv2.approxPolyDP(contour[0], epsilon, True)  # 近似提取轮廓的顶点
        M = cv2.moments(contour[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])  # 计算轮廓中心坐标 小心轮廓面积为0
        img = cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
        cv2.drawContours(img, approx, -1, (55, 200, 255), 5)  # 顶点可视化
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = CamSet().CorrectImage(img)
        return img, max(area), (cX, cY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = CamSet().CorrectImage(img)
        return img, None, None



def PlayVideo(cap, low, high):
    while cap.isOpened():
        ret, frame = cap.read()
        video, area, pos = GetMask(frame, low, high)  # 处理视频
        distance = OneEyeDistance(543.2352805344083, 540.0090338378359, S, area)
        if pos is not None:
            direction = Direction(543.2352805344083, 540.0090338378359,
                                  (335.6325913504112, 237.52912533756933), pos)
            cv2.putText(video, "direction:" + str(direction), (50, 100), fontScale=0.5, color=(0, 0, 255), thickness=2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, )
        if distance is not None:
            cv2.putText(video, "distance:%.3f" % distance, (50, 50), fontScale=0.5, color=(255, 255, 255), thickness=2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        cv2.imshow("video", video)  # 可视化视频
        out.write(video)  # 输出视频
        c = cv2.waitKey(5)
        if c == 27:
            break    # 每5毫秒等一次，直到按下Esc键后退出
    return 0


def OneEyeDistance(fx, fy, S1, S2):
    if S2 is not None:
        d1 = math.sqrt(fx * fx * S1 / S2)
        d2 = math.sqrt(fy * fy * S1 / S2)
        return (d1 + d2) / 2
    else:
        return None


def Direction(fx, fy, center, pos):
    """输入焦距fx, fy,相机光心坐标center,物体轮廓中心坐标pos,输出估算的物体单位方向向量(x, y, z)"""
    x = (pos[0]-center[0])/fx
    y = (pos[1]-center[1])/fy
    direction = np.array([x, y, 1])  # 映射到z=1平面上的方向向量
    length = np.linalg.norm(direction)
    direction = direction/length  # 归一化
    return np.round(direction, decimals=3)




PlayVideo(cap, low, high)