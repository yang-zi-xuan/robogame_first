import numpy as np
import cv2
import pupil_apriltags as apriltag
import os


#相机的位置
class position:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def update(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def get_position(self):
        return [self.x, self.y, self.angle]


def load_known_tags(image_dir):
    known_tags = {}
    at_detector = apriltag.Detector(families='tag36h11')
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            ret, img = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
            tags = at_detector.detect(img)
            for tag in tags:
                known_tags[tag.tag_id] = filename.split('.')[0]
    return known_tags


fx = 554.67362769  # 相机焦距 x
fy = 552.05429286  # 相机焦距 y
cx = 351.81151839  # 主点 x
cy = 227.12553941  # 主点 y

field_length = 4
field_width = 3
tag_distance = 0.64
tag_down_edge_distance = 0.3
tag_side_edge_distance = 0.4

position = position(0, 0, 0)

# 相机参数矩阵
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# AprilTag 实际尺寸（米）
tag_size = 0.4

# 世界坐标系中的AprilTag四个角点
tag_points_world = np.array([
    [-tag_size / 2, -tag_size / 2, 0],
    [tag_size / 2, -tag_size / 2, 0],
    [tag_size / 2, tag_size / 2, 0],
    [-tag_size / 2, tag_size / 2, 0]
])

known_tags_dir = 'templates'  # 存储已知标签图像的文件夹
known_tags = load_known_tags(known_tags_dir)
print(known_tags)

cap = cv2.VideoCapture(0)
at_detector = apriltag.Detector(families='tag36h11')

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置 VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编解码器
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 输出视频文件名，编解码器，帧率，分辨率

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    tags = at_detector.detect(gray)

    if len(tags) > 0:
        for tag in tags:

            # 获取标签的四个角点
            corners = np.array(tag.corners, dtype=np.float32)

            # 绘制正方形框
            corners_int = corners.astype(int)
            cv2.polylines(frame, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)

            # 计算AprilTag在相机图像坐标系下的位置
            tag_points_camera = np.array([
                corners[0], corners[1], corners[2], corners[3]
            ], dtype=np.float32)

            # 使用solvePnP计算旋转矩阵和平移向量
            try:
                success, rvec, tvec = cv2.solvePnP(tag_points_world, tag_points_camera, K, None,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
                if not success:
                    print(f"solvePnP failed for tag id {tag.tag_id}")
                    continue

                # 旋转矩阵转换为旋转向量
                R, _ = cv2.Rodrigues(rvec)

                # 计算相机到标签中心的方向向量
                center = tag.center.astype(int)
                direction_vector_camera = np.dot(R, np.array([0, 0, -1]))

                # 估计距离
                pixel_size = np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1))
                distance = fx * tag_size / pixel_size

                # 计算相机坐标系中标签中心的方向向量
                tag_center_camera = tvec.flatten()

                # 计算标签平面的法向量（方向向量）
                # 假设标签平面的法向量在世界坐标系中是 [0, 0, 1]
                normal_vector_world = np.array([0, 0, 1])
                normal_vector_camera = R @ normal_vector_world
                """这个向量打印出来是需要变号的"""
                normal_vector_camera = -normal_vector_camera

                # 计算水平面的夹角
                angle_1 = np.arctan2(tag_center_camera[0], tag_center_camera[2]) * 180 / np.pi

                #计算方向向量与z轴方向的夹角
                angle_normal_vector = np.arctan2(normal_vector_camera[0], normal_vector_camera[2]) * 180 / np.pi
                angle_2 = angle_normal_vector + 90

                # 确保 angle 是一个标量
                angle_1 = float(angle_1)  #位置夹角
                angle_2 = float(angle_2)  #姿态夹角

                x_list = []
                y_list = []
                angle_2_list = []

                # 获取 AprilTag 的 ID
                tag_id = tag.tag_id
                if tag_id in known_tags:
                    tag_name = known_tags[tag_id]
                    cv2.putText(frame, f'{tag_name}', (int(corners[0][0]), int(corners[0][1]) - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    tag_name = int(tag_name)
                    x_list.append(tag_side_edge_distance + (tag_name - 1) * tag_distance + distance * np.cos(angle_1))
                    y_list.append(distance * np.sin(angle_1))
                    angle_2_list.append(angle_2)

                else:
                    cv2.putText(frame, f'Unknown Tag {tag_id}', (int(corners[0][0]), int(corners[0][1]) - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                x = np.mean(x_list)
                y = np.mean(y_list)
                angle = np.mean(angle_2_list)
                position.update(x, y, angle)

                # 在图像上绘制方向向量
                endpoint_camera = center + 50 * direction_vector_camera[:2]  # 放大显示方向向量
                cv2.line(frame, tuple(center), tuple(endpoint_camera.astype(int)), (0, 0, 255), 2)

                # 在图像上显示距离和夹角信息
                cv2.putText(frame, f'Distance: {distance:.2f} m', (int(corners[0][0]), int(corners[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Distance_Angle: {angle_1:.2f} deg', (int(corners[0][0]), int(corners[0][1]) - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Direction_Angle: {angle_2:.2f} deg', (int(corners[0][0]), int(corners[0][1]) - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            except cv2.error as e:
                print(f"OpenCV error: {e}")

    cv2.imshow('Frame', frame)
    out.write(frame)

    # 按 'ESC' 键退出循环
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
