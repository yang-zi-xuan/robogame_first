import cv2
import numpy as np


"""认为：主程序中封装camera对象（包含内参等信息）;主程序中给出了颜色识别阈值;主程序提供img"""


class PNPframe(object):
    """输入一帧画面，可获得方块顶点、方块体心在相机坐标系中的坐标、方块的朝向"""
    def __init__(self, img):
        self.img = img
        self.approx = None
        self.center = None
        self.direction = None

    def GetApprox(self, low, high):
        """输入：颜色阈值
        返回：对PNPframe对象提取的顶点
        并且将顶点的像素坐标记录到属性中"""
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img = cv2.GaussianBlur(img, (5, 5), 0, None)  # 高斯模糊
        mask = cv2.inRange(img, low, high)  # 生成掩膜
        print("get mask")

        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = list()
        area = [cv2.contourArea(contours[i]) for i in range(len(contours))]
        print("get area")
        # 选择面积最大的掩膜作为识别到的对象（则一次只能识别一个）

        if len(area) > 0:
            contour.append(contours[area.index(max(area))])
            epsilon = 0.015 * cv2.arcLength(contour[0], True)  # 根据轮廓周长计算逼近精度
            approx = cv2.approxPolyDP(contour[0], epsilon, True)  # 近似提取轮廓的顶点
# 主机调试处：顶点可视化
            # cv2.drawContours(img, approx, -1, (55, 200, 255), 8)
            print("get approx")
            self.approx = approx
            return approx
        else:
            return None


    def Center_Direction(self, camera_matrix, dist_coeffs):
        img, ret, rvec, tvec = PNP_Caculator(self.img, self.approx, camera_matrix,dist_coeffs)
        print("get rvec")
        if ret:
            self.center, self.direction = CaculateCube(rvec, tvec)
            print("get PNP center and direction")
            return 0
        else:
            print("fail to get PNP center and direction")
            return -1


def sort_points_clockwise(points, start_point=None):
    """
    对一组点集进行顺时针排序
    :param points: 输入的点集坐标,numpy数组格式
    :param start_point: 指定的起始点,如果为None则选取第一个点作为起始点
    :return: 排序后的点集坐标
    """
    # 计算中心点
    center = points.mean(axis=0)

    # 计算每个点相对于中心点的极坐标角度
    angles = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])

    # 对角度进行排序,并使用原始索引保存排序后的点集
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    # 如果指定了起始点,则将起始点移动到第一个位置
    if start_point is not None:
        start_idx = np.where(np.all(sorted_points == start_point, axis=1))[0][0]
        sorted_points = np.concatenate([sorted_points[start_idx:], sorted_points[:start_idx]])

    return sorted_points


def ApproxOrder(approx):
    """整理顶点顺序
    approx：输入一组闭环点集的坐标,形式为np.array([[[]], [[]], [[]]...])
    输出按y值由大到小排列的approx,形式为np.array([[], [], []...], dtype=np.float32)
    输出以左下顶点(y值最大的两个点中x更小的那个)为起点，顺时针排列的apx,形式为np.array([[], [], []...], dtype=np.float32)
    输出该起点startPoint
    """
    approx.reshape(-1, 2)
    list = np.array([approx[i][0] for i in range(len(approx))], dtype=np.float32)
    # print("list", list)
    approx = list[list[:, 1].argsort()]
    approx = np.array(approx, dtype=np.float32)  # 预处理顶点列表approx，按y值排序
    # print("approx", approx)
    Last_Points = approx[-3:, :]
    Last_Points = Last_Points[Last_Points[:, 0].argsort()]
    P1 = Last_Points[0]
    P2 = Last_Points[1]
    P3 = Last_Points[2]  # 获得y最大的三个点，并且按x从小到大排列
    d1 =np.linalg.norm(P1 - P2)
    d2 = np.linalg.norm(P2 - P3)  # 计算两组相邻点间的距离
    out = (P1, 0) if d1 > d2 else (P2, 1)
    startPoint = out[0]
    apx = sort_points_clockwise(approx, start_point=startPoint)
    # print("apx:", apx, "\nstart:", startPoint)
    return approx, apx.astype(np.float32), startPoint, out[1]


def ChooseSpots(matrix1, matrix2):
    """输入两个n*2的numpy矩阵，matrix2为matrix1的子集，保持matrix1顺序不变，删去其中不属于matrix2的行"""
    matrix = []
    matrix1.astype(np.int32)
    matrix2.astype(np.int32)
    # print("1", matrix1, "\n2", matrix2)
    for Spot in matrix1:
        if np.any(np.all(Spot == matrix2, axis=1)):
            matrix.append(Spot)
    # print("matrix", matrix)
    return np.array(matrix, dtype=np.float32)


def DrawChosenSpot(apx, img):
    """画出选中的顶点，可删去"""
    for Spot in apx:
        x, y = Spot.astype(int)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    return img


def PNP_Caculator(img, approx, camera_matrix, dist_coeffs):
    if approx is None:
        return img, False, None, None
    elif len(approx) < 4 or len(approx) > 6:
        print("Number of Approxes Is Wrong", len(approx))
        return img, False, None, None
    else:
        approx, apx, start, order = ApproxOrder(approx)  # 整理顶点顺序

    if len(approx) == 4:
        print("4 approxes")
        objectPoints = np.array([[0, 0, 0], [-10, 0, 0], [-10, 10, 0], [0, 10, 0]], dtype=np.float32)
        # 此处空间直角坐标系选择右手系
        ret, rvec, tvec = cv2.solvePnP(objectPoints, apx, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE)
# 主机调试入口
        # img = DrawChosenSpot(apx, img)
        return img, ret, rvec, tvec
    elif len(approx) == 5:
        print("5 approxes")
        objectPoints = np.array([[0, 0, 0], [-10, 0, 10], [-10, 10, 10], [0, 10, 0]], dtype=np.float32)
        approx = np.delete(approx, 2, axis=0)
        # print("approx", approx)
        apx = ChooseSpots(apx, approx)
        # print("apx51", apx, approx)  # 删去y为中间的那个点
        ret, rvec, tvec = cv2.solvePnP(objectPoints, apx, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE)
# 主机调试入口
        # img = DrawChosenSpot(apx, img)
        return img, ret, rvec, tvec
    elif len(approx) == 6:
        print("6 approxes")
        apx = np.take(apx, [0, 2, 3, 5], axis=0)
        # print("apx6", apx)
        objectPoints = np.array([[0, 0, 0], [-10, 0, 10], [-10, 10, 10], [0, 10, 0]], dtype=np.float32)
        ret, rvec, tvec = cv2.solvePnP(objectPoints, apx, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE)
# 主机调试入口
        # img = DrawChosenSpot(apx, img)
        return img, ret, rvec, tvec


def CaculateCameraCoord(rvec, tvec, wCoord):
    """世界坐标系的点坐标转化为相机坐标系的点坐标"""
    rotate = cv2.Rodrigues(rvec)[0]  # 利用rvec计算旋转矩阵
    wCoord = np.append(wCoord, 1)
    TransMatrix = np.pad(rotate, ((0, 1), (0, 1)), mode="constant", constant_values=0)
    TransMatrix[3][3] = 1
    for i in range(3):
        TransMatrix[i][3] = tvec[i]  # 将平移向量tvec扩充入旋转矩阵
    cCoord = np.dot(TransMatrix, wCoord)
    cCoord = np.delete(cCoord, 3)  # 删掉多出的第四个分量的1
    # print(cCoord)
    return cCoord  # 相机坐标系坐标


def CaculateCube(rvec, tvec):
    """输入旋转、平移向量，选取立方体四个顶点，
    返回立方体体心到相机的距离，以及相机坐标系下立方体的朝向"""
    approx = np.array([[0, 0, 0],
                       [0, 10, 10],
                       [-10, 0, 10],
                       [-10, 10, 0]])  # 取立方体上构成正四面体的四个顶点
    approx = np.array([CaculateCameraCoord(rvec, tvec, approx[i, :]) for i in range(4)])  # 计算其相机坐标系坐标
    center = approx.mean(axis=0)  # 计算立方体中心的相机坐标系坐标
    P1 = CaculateCameraCoord(rvec, tvec, np.array([0, 5, 5]))
    Direction = P1 - center  # 立方体朝向的参考向量
    Direction = Direction/5
    return center, Direction


def ShowDatas(distance, direction):
    direction = np.round(direction, decimals=3)

    w, h = 800, 200
    img = np.zeros((h, w, 3), np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    cv2.namedWindow("datas", cv2.WINDOW_AUTOSIZE)

    cv2.putText(img, f"distance = {distance}", (10, 50), font, 0.6, font_color, line_type)
    cv2.putText(img, f"direction = {direction}", (10, 80), font, font_scale, font_color, line_type)
    cv2.imshow("datas", img)
    return 0


