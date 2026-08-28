import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import mujoco
from pathlib import Path



'''
获取三个物品的实时位置
'''
class SnapshotPrinter(Node):
    def __init__(self):
        super().__init__('snapshot_printer')
        self.sub = self.create_subscription(
            String, '/task_scene_snapshot', self.cb, 10
        )
        self._received = False
        self.total_data = []
    def get_data(self):
        return self.total_data
    def cb(self, msg):
        if self._received:
            return
        self._received = True
        data = json.loads(msg.data)
        data_list = data["movable_objects"]
        data_list[0]["position_xyz"][2] += 0.07
        data_list[1]["position_xyz"][2] += 0.11
        data_list[2]["position_xyz"][2] += 0.1075
        for i in data_list:
            # print(i["body_name"], end='\t')
            t_list = i["position_xyz"]
            for j in range(len(t_list)):
                # print(type(t_list))
                t=t_list[j]
                # print(round(t, 3), end=' ')
                t=round(t, 3)
                t_list[j]=t
            # print("")
            self.total_data.append({
                "name":i["body_name"],
                "position":i["position_xyz"]
            })
        # print("-----------------")
        self.destroy_subscription(self.sub)

'''保存rgb图像'''
class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        self.bridge = CvBridge()
        self.count = 0
        self._saved = False  # ← 加这个

        self.save_dir = './mujoco_env/main/xiaoman1/images'
        os.makedirs(self.save_dir, exist_ok=True)

        topic = self.declare_parameter('topic', '/external_camera/rgb/image_raw').value

        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            Image,
            topic,
            self.callback,
            qos
        )
        self.filename=""

        # self.get_logger().info(f'正在监听话题: {topic}')

    def callback(self, msg):
        if self._saved:   # ← 加这个
            return         # ← 加这个

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        filename = os.path.join(self.save_dir, f'frame_rgb.jpg')
        success = cv2.imwrite(filename, cv_img)

        if success:
            self.count += 1
            self.filename=str(filename)
            # print(f"成功保存，地址为{filename}")

            # self.get_logger().info(f'已保存: {filename}')

        self._saved = True  # ← 加这个
        self.destroy_subscription(self.subscription)  # ← 只停订阅

    def destroy_node(self):
        # self.get_logger().info(f'总共保存了 {self.count} 张图片')
        super().destroy_node()

'''保存深度图以及归一图'''
class DepthSaver(Node):
    def __init__(self):
        super().__init__('depth_saver')
        self.bridge = CvBridge()
        self.count = 0
        self._saved = False  # ← 加这个
        self.save_dir = './mujoco_env/main/xiaoman1/images'
        os.makedirs(self.save_dir, exist_ok=True)
        topic = self.declare_parameter('topic', '/external_camera/depth/image_raw').value
        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.depthfile=""
        self.depthrawfile=""

        self.subscription = self.create_subscription(Image, topic, self.callback, qos)
        # self.get_logger().info(f'正在监听深度话题: {topic}')

    def callback(self, msg):
        if self._saved:   # ← 加这个
            return         # ← 加这个
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'转换失败: {e}')
            return
        # ---- 1. 保存原始深度 ----
        dtype = depth.dtype
        base = f'depth_{self.count:06d}'
        if dtype == np.uint16:
            raw_path = os.path.join(self.save_dir, f'depth_raw.png')
            cv2.imwrite(raw_path, depth)
        elif dtype == np.float32:
            raw_path = os.path.join(self.save_dir, f'depth_raw.npy')
            np.save(raw_path, depth)
        else:
            raw_path = os.path.join(self.save_dir, f'depth_raw.npy')
            np.save(raw_path, depth.astype(np.float32))
        self.depthrawfile = raw_path
        # self.get_logger().info(f'原始深度保存: {raw_path}')

        # ---- 2. 彩色可视化 ----
        depth_float = depth.astype(np.float32)
        depth_clipped = np.clip(depth_float, 500, 1500)
        depth_norm = ((depth_clipped - 500) / (1500 - 500) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        vis_path = os.path.join(self.save_dir, f'depth.jpg')
        self.depthfile = vis_path
        cv2.imwrite(vis_path, depth_colored)
        # self.get_logger().info(f'彩色可视化保存: {vis_path}')

        self.count += 1
        self._saved = True  # ← 加这个
        self.destroy_subscription(self.subscription)  # ← 只停订阅

    def destroy_node(self):
        # self.get_logger().info(f'总共保存了 {self.count} 组深度图')
        super().destroy_node()


'''世界坐标转相机坐标'''
class WordToCam:
    """世界坐标转像素坐标"""

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_MODEL = PROJECT_ROOT / "mujoco_env/robot_model/exp/env_robot_torque_tactile.xml"
    CAMERA_NAME = "rgbd_camera_external"

    def __init__(
        self,
        p_world,
        model_path=None,
        image_path=None,
        output_path=None,
        width=640,
        height=480,
    ):
        self.p_world = np.asarray(p_world, dtype=np.float64)
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL
        self.image_path = image_path
        self.output_path = output_path
        self.width = width
        self.height = height

        # 初始化时加载模型并计算内参
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.cam_id = self.model.cam(self.CAMERA_NAME).id
        fovy_rad = np.deg2rad(self.model.cam_fovy[self.cam_id])
        fy = self.height / (2.0 * np.tan(fovy_rad / 2.0))
        self.fx = fy
        self.fy = fy
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def project(self):
        """投影世界坐标到像素坐标，返回 (u, v, depth, is_in_image)"""
        rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        pos = self.data.cam_xpos[self.cam_id]
        p_cam = rot.T @ (self.p_world - pos)

        # MuJoCo -> OpenCV
        p_cv = np.array([p_cam[0], -p_cam[1], -p_cam[2]])

        if p_cv[2] <= 0.0:
            raise ValueError(f"该世界点位于相机后方，无法投影：p_C={p_cv}")

        u = self.fx * p_cv[0] / p_cv[2] + self.cx
        v = self.fy * p_cv[1] / p_cv[2] + self.cy
        ok = 0.0 <= u < self.width and 0.0 <= v < self.height

        # print(f"投影像素 (u, v): ({u:.3f}, {v:.3f}), 深度: {p_cv[2]:.4f}m, 在图内: {ok}")
        return u, v, p_cv[2], ok

    def draw(self, u, v):
        """在图像上画红点并保存"""
        if self.image_path is None or self.output_path is None:
            raise ValueError("未设置 image_path 或 output_path")

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图片：{self.image_path}")

        point = (round(u), round(v))
        cv2.circle(image, point, 3, (0, 0, 255), -1)
        cv2.circle(image, point, 5, (0, 0, 255), 1)
        cv2.putText(image, f"({point[0]}, {point[1]})", (point[0] + 12, point[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(self.output_path), image)
        # print(f"已保存: {self.output_path}")

'''相机转世界坐标'''
class PixelToWorld:
    """从深度图和像素坐标反投影到世界坐标"""

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_MODEL = PROJECT_ROOT / "mujoco_env/robot_model/exp/env_robot_torque_tactile.xml"
    CAMERA_NAME = "rgbd_camera_external"

    def __init__(
        self,
        depth_path,
        model_path=None,
        width=640,
        height=480,
    ):
        self.depth_path = Path(depth_path)
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL
        self.width = width
        self.height = height

        # 加载深度图
        self.depth_mm = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)
        if self.depth_mm is None:
            raise FileNotFoundError(f"无法读取深度图：{self.depth_path}")
        if self.depth_mm.ndim != 2:
            raise ValueError("深度图必须是单通道图像")

        # 加载模型并计算内参
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.cam_id = self.model.cam(self.CAMERA_NAME).id
        fovy_rad = np.deg2rad(self.model.cam_fovy[self.cam_id])
        fy = self.height / (2.0 * np.tan(fovy_rad / 2.0))
        self.fx = fy
        self.fy = fy
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def back_project(self, u: int, v: int):
        """
        输入像素坐标 (u, v)，返回 (p_world, depth_m)
        """
        h, w = self.depth_mm.shape
        if not (0 <= u < w and 0 <= v < h):
            raise ValueError(f"像素 ({u}, {v}) 超出图像范围：宽={w}, 高={h}")

        # uint16 PNG 单位为 mm -> m
        z_c = float(self.depth_mm[v, u]) / 1000.0
        if not np.isfinite(z_c) or z_c <= 0.0:
            raise ValueError(f"像素 ({u}, {v}) 没有有效深度，原始值={self.depth_mm[v, u]}")

        # OpenCV 相机坐标
        p_camera_cv = np.array([
            (u - self.cx) * z_c / self.fx,
            (v - self.cy) * z_c / self.fy,
            z_c,
        ])

        # MuJoCo 相机坐标
        p_camera_mj = np.diag([1.0, -1.0, -1.0]) @ p_camera_cv

        # 世界坐标
        rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        pos = self.data.cam_xpos[self.cam_id]
        p_world = rot @ p_camera_mj + pos

        # print(f"像素 ({u}, {v}) -> 世界坐标 {p_world}, 深度 {z_c:.4f}m")
        return p_world, z_c





def printf(data):
        print(json.dumps(data, indent=2, ensure_ascii=False))



def main():
    rclpy.init()
    node = SnapshotPrinter()

    # 手动 spin_once，收到消息后就退出循环
    while rclpy.ok() and not node._received:
        rclpy.spin_once(node, timeout_sec=0.1)
    target_data_li = node.get_data()
    node.destroy_node()
    print("获取三个物品世界坐标")
    printf(target_data_li)
    node = ImageSaver()
    # 手动 spin_once 循环，保存完就退出
    while rclpy.ok() and not node._saved:
        rclpy.spin_once(node, timeout_sec=0.1)
    # print(node.filename)
    rgb_file_address = node.filename
    node.destroy_node()
    print(f"rgb保存地址为:\t\t\t {rgb_file_address}")

    node = DepthSaver()
    # 手动 spin_once，收到一张就退出循环
    while rclpy.ok() and not node._saved:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()

    depth_file_address = node.depthfile
    depth_raw_file_address = node.depthrawfile

    print(f"深度图归一化保存地址: \t\t{depth_file_address}")
    print(f"深度图原始图像保存地址: \t{depth_raw_file_address}")

    for index in range(len(target_data_li)):
        target=target_data_li[index]
        val_p_world=target["position"]
        projector = WordToCam(

            p_world = target["position"],

            image_path = rgb_file_address,
            output_path = f'./mujoco_env/main/xiaoman1/images/image_{index}{target["name"]}标注的照片.jpg',
        )
        print(val_p_world)
        # 调用实例方法
        u, v, val_depth, ok = projector.project()
        projector.draw(u, v)
        projector = PixelToWorld(
            depth_path=depth_raw_file_address
        )
        cal_p_world, depth = projector.back_project(int(u), int(v))
        # print(int(u),int(v),round(_,4))
        # print(val_p_world)
        print(target["name"])
        print(f"推算结果: x={round(cal_p_world[0],2)}\ty={round(cal_p_world[1],2)}\tz={round(cal_p_world[2],2)}\tdepth={round(depth,4)}")

        print(f"真实结果: x={round(val_p_world[0],2)}\ty={round(val_p_world[1],2)}\tz={round(val_p_world[2],2)}\tdepth={round(val_depth,4)}")

        print(f"误差范围: x={round(cal_p_world[0]-val_p_world[0],2)}\ty={round(cal_p_world[1]-val_p_world[1],2)}\tz={round(cal_p_world[2]-val_p_world[2],2)}\tdepth={round(val_depth-depth,4)}")

        print('-'*100)

        



        




    rclpy.shutdown()





    


main()