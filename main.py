import numpy as np
import cv2


cap = cv2.VideoCapture(0)  # 视频对象
high = np.array([50, 255, 255])
low = np.array([0, 100, 60])  # 颜色识别阈值
path = "E:\\WareHouse\\RoboGame\\test_pictures\\cubes"
camera_matrix = np.array([[543.2352805344083, 0.0, 335.6325913504112],
                          [0, 540.0090338378359, 237.52912533756933],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.041107079237683884, 0.09952660331229725,-0.002757530806781375, 0.0024557170640417568,
                        0.1088243671433803], dtype=np.float32)
fourcc = cv2.VideoWriter.fourcc(*"XVID")  # 编码器
out = cv2.VideoWriter("C:\\Users\\yb028028\\Desktop\\my_video.avi", fourcc, 30.00, (640, 480))  # 视频输出对象


def GetMask(img, low, high):
    """new_width = 300
    new_height = 200
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0, None)  # 高斯模糊
    mask = cv2.inRange(img, low, high)  # 生成掩膜

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = list()
    area = [cv2.contourArea(contours[i]) for i in range(len(contours))]  # 选择面积最大的掩膜作为识别到的对象（则一次只能识别一个）
    if len(area) > 0:
        contour.append(contours[area.index(max(area))])
        epsilon = 0.015 * cv2.arcLength(contour[0], True)  # 根据轮廓周长计算逼近精度
        approx = cv2.approxPolyDP(contour[0], epsilon, True)  # 近似提取轮廓的顶点
        #img = cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
        cv2.drawContours(img, approx, -1, (55, 200, 255), 8)  # 顶点可视化
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img, max(area), approx
    else:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img, None, None


def draw_coordsys(img, origin, axes_proj, thickness=3):
    """
    Draw a 3D coordinate system on the image.
    origin: the origin of the coordinate system.
    axes_proj: 3 unit vectors of the coordinate system.
    图象是BGR通道
    """
    origin = np.int32(origin)
    img = cv2.line(img, origin, tuple(axes_proj[0].ravel()), (255, 0, 0), thickness)  # 蓝色：x参考向量
    img = cv2.line(img, origin, tuple(axes_proj[1].ravel()), (0, 255, 0), thickness)  # 绿色：y参考向量
    img = cv2.line(img, origin, tuple(axes_proj[2].ravel()), (0, 0, 255), thickness)  # 红色：z参考向量
    return img


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
        axes = np.float32([[-10, 0, 0], [0, 10, 0], [0, 0, -10]]).reshape(-1, 3)  # 参考向量，认为发出点是(0, 0, 0)
        objectPoints = np.array([[0, 0, 0], [-10, 0, 0], [-10, 10, 0], [0, 10, 0]], dtype=np.float32)
        # 此处空间直角坐标系选择右手系
        ret, rvec, tvec = cv2.solvePnP(objectPoints, apx, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE)
        axes_proj, jac = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)  # 解算参考向量在图像上的位姿
        axes_proj = np.int32(axes_proj).reshape(-1, 2)
        # print("axes_proj4", axes_proj, start)
        img = draw_coordsys(img, apx[0], axes_proj)  # 绘制参考向量
        img = DrawChosenSpot(apx, img)
        return img, ret, rvec, tvec
    elif len(approx) == 5:
        axes = np.float32([[-10, 0, 0], [0, 10, 0], [0, 0, 10]]).reshape(-1, 3)  # 参考向量，认为发出点是(0, 0, 0)
        objectPoints = np.array([[0, 0, 0], [-10, 0, 10], [-10, 10, 10], [0, 10, 0]], dtype=np.float32)
        approx = np.delete(approx, 2, axis=0)
        print("approx", approx)
        apx = ChooseSpots(apx, approx)
        # print("apx51", apx, approx)  # 删去y为中间的那个点
        ret, rvec, tvec = cv2.solvePnP(objectPoints, apx, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE)
        axes_proj, jac = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)  # 解算参考向量在图像上的位姿
        axes_proj = np.int32(axes_proj).reshape(-1, 2)
        # print("axes_proj5", axes_proj, start)
        img = draw_coordsys(img, apx[0], axes_proj)  # 绘制参考向量
        img = DrawChosenSpot(apx, img)
        return img, ret, rvec, tvec
    elif len(approx) == 6:
        axes = np.float32([[-10, 0, 0], [0, 10, 0], [0, 0, 10]]).reshape(-1, 3)  # 参考向量，认为发出点是(0, 0, 0)
        apx = np.take(apx, [0, 2, 3, 5], axis=0)
        # print("apx6", apx)
        objectPoints = np.array([[0, 0, 0], [-10, 0, 10], [-10, 10, 10], [0, 10, 0]], dtype=np.float32)
        ret, rvec, tvec = cv2.solvePnP(objectPoints, apx, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE)
        axes_proj, jac = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)  # 解算参考向量在图像上的位姿
        axes_proj = np.int32(axes_proj).reshape(-1, 2)
        # print("axes_proj6", axes_proj, start)
        img = draw_coordsys(img, apx[0], axes_proj)  # 绘制参考向量
        img = DrawChosenSpot(apx, img)
        return img, ret, rvec, tvec


def CaculateCameraCoord(rvec, tvec, wCoord):
    rotate = cv2.Rodrigues(rvec)[0]
    wCoord = np.append(wCoord, 1)
    TransMatrix = np.pad(rotate, ((0, 1), (0, 1)), mode="constant", constant_values=0)
    TransMatrix[3][3] = 1
    for i in range(3):
        TransMatrix[i][3] = tvec[i]
    cCoord = np.dot(TransMatrix, wCoord)
    cCoord = np.delete(cCoord, 3)
    print(cCoord)
    return cCoord


def CaculateCube(rvec, tvec):
    approx = np.array([[0, 0, 0],
                       [0, 10, 10],
                       [-10, 0, 10],
                       [-10, 10, 0]])  # 取立方体上构成正四面体的四个顶点
    approx = np.array([CaculateCameraCoord(rvec, tvec, approx[i, :]) for i in range(4)])  # 计算其相机坐标系坐标
    center = approx.mean(axis=0)  # 计算立方体中心的相机坐标系坐标
    P1 = CaculateCameraCoord(rvec, tvec, np.array([0, 5, 5]))
    Direction = P1 - center # 立方体朝向的参考向量
    Direction = Direction/5
    return center, Direction


def PlayVideo(cap, low, high):
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        video, area, approx = GetMask(frame, low, high)  # 处理视频
        video, ret, rvec, tvec = PNP_Caculator(video, approx, camera_matrix, dist_coeffs)
        if ret:
            center, direction = CaculateCube(rvec, tvec)
            print(center, direction)
        cv2.imshow("video", video)  # 可视化视频
        out.write(video)
        c = cv2.waitKey(10)
        if c == 27:
            break    # 每5毫秒等一次，直到按下Esc键后退出
        elif c == 32:
            i += 1
            cv2.imwrite(path + "\\cube%d.jpg" % i, video)
            print("Get a Picture", i)
    return 0

PlayVideo(cap, low, high)
