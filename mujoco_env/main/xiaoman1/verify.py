import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import time
import cv2 as cv
from torchvision.models import resnet18, ResNet18_Weights
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from pathlib import Path
import mujoco


# ==================== 世界坐标转像素坐标 ====================
class WordToCam:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_MODEL = PROJECT_ROOT / "mujoco_env/robot_model/exp/env_robot_torque_tactile.xml"
    CAMERA_NAME = "rgbd_camera_external"

    def __init__(self, p_world, model_path=None, width=640, height=480):
        self.p_world = np.asarray(p_world, dtype=np.float64)
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL
        self.width = width
        self.height = height

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
        rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        pos = self.data.cam_xpos[self.cam_id]
        p_cam = rot.T @ (self.p_world - pos)
        p_cv = np.array([p_cam[0], -p_cam[1], -p_cam[2]])

        if p_cv[2] <= 0.0:
            raise ValueError(f"该世界点位于相机后方，无法投影：p_C={p_cv}")

        u = self.fx * p_cv[0] / p_cv[2] + self.cx
        v = self.fy * p_cv[1] / p_cv[2] + self.cy
        ok = 0.0 <= u < self.width and 0.0 <= v < self.height
        return u, v, p_cv[2], ok


# ==================== 网络结构 ====================
class DoubleStreamResNet(nn.Module):
    def __init__(self, num_objects=3, dim_per_obj=9, pretrained=True):
        super().__init__()
        self.num_objects = num_objects
        self.dim_per_obj = dim_per_obj
        self.total_dim = num_objects * dim_per_obj
        
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.rgb_backbone = resnet18(weights=weights)
        self.rgb_backbone.fc = nn.Identity()
        self.depth_backbone = resnet18(weights=weights)
        self.depth_backbone.fc = nn.Identity()
        
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, self.total_dim)
        )
        
        nn.init.xavier_uniform_(self.fusion[-1].weight, gain=0.01)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, rgb, depth):
        depth_3ch = depth.repeat(1, 3, 1, 1)
        fused = torch.cat([self.rgb_backbone(rgb), self.depth_backbone(depth_3ch)], dim=1)
        return self.fusion(fused).view(-1, self.num_objects, self.dim_per_obj)


# ==================== 推理类 ====================
class RGBDInferencer:
    def __init__(self, model_path, device='cuda', mask_roi=(180, 480, 340, 640)):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mask_roi = mask_roi
        
        self.model = DoubleStreamResNet(num_objects=3, dim_per_obj=9, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, rgb_np, depth_np, depth_in_meters=False):
        rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(np.asarray(depth_np)).float().unsqueeze(0)
        
        if not depth_in_meters:
            depth = depth / 1000.0
            
        if self.mask_roi is not None:
            h_start, h_end, w_start, w_end = self.mask_roi
            mask = torch.zeros_like(depth)
            mask[:, h_start:h_end, w_start:w_end] = 1
            rgb = rgb * mask
            depth = depth * mask
        
        return rgb.unsqueeze(0).to(self.device), depth.unsqueeze(0).to(self.device)
    
    def predict(self, rgb_np, depth_np, depth_in_meters=False):
        rgb_tensor, depth_tensor = self.preprocess(rgb_np, depth_np, depth_in_meters)
        with torch.no_grad():
            pred = self.model(rgb_tensor, depth_tensor).cpu().numpy()[0]
        
        results = []
        for i in range(3):
            u_norm, v_norm, d, x_cam, y_cam, z_cam, x_world, y_world, z_world = pred[i]
            results.append({
                'object_id': i,
                'pixel_uv': (round(u_norm * 640.0, 1), round(v_norm * 480.0, 1)),
                'depth_m': round(d, 4),
                'camera_xyz': (round(x_cam, 4), round(y_cam, 4), round(z_cam, 4)),
                'world_xyz': (round(x_world, 4), round(y_world, 4), round(z_world, 4))
            })
        return results


# ==================== ROS2 节点（不存盘，直接存内存） ====================
class RGBDAndSnapshotSaver(Node):
    def __init__(self):
        super().__init__('rgbd_and_snapshot_saver')
        self.bridge = CvBridge()

        self._snapshot_received = False
        self._image_received = False
        self._snapshot_data = None
        
        # ★ 内存中直接存储图像数据，不写磁盘
        self.rgb_data = None
        self.depth_data = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        self.snapshot_sub = self.create_subscription(
            String, '/task_scene_snapshot', self._snapshot_cb, 10
        )

        sub_rgb = message_filters.Subscriber(self, Image, '/external_camera/rgb/image_raw', qos_profile=qos)
        sub_depth = message_filters.Subscriber(self, Image, '/external_camera/depth/image_raw', qos_profile=qos)

        self.ts = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], queue_size=10, slop=0.1)
        self.ts.registerCallback(self._image_cb)

    def _snapshot_cb(self, msg: String):
        if self._snapshot_received:
            return
        self._snapshot_received = True

        data = json.loads(msg.data)
        data_list = data["movable_objects"]
        # 高度偏移补偿
        for item in data_list:
            item["position_xyz"][2] += {"beaker1": 0.07, "graduated_cylinder": 0.11, "erlenmeyer_flask": 0.1075}.get(item["body_name"], 0)

        self._snapshot_data = []
        name_id = {"beaker1": 0, "graduated_cylinder": 1, "erlenmeyer_flask": 2}

        for item in data_list:
            t_list = item["position_xyz"]
            projector = WordToCam(p_world=t_list)
            u, v, val_depth, ok = projector.project()
            z_c = val_depth
            x_c = (u - projector.cx) * z_c / projector.fx
            y_c = (v - projector.cy) * z_c / projector.fy

            self._snapshot_data.append({
                "name": item["body_name"],
                "class_id": name_id[item["body_name"]],
                "pixel_uv_depth": [round(u, 2), round(v, 2), round(val_depth, 5)],
                "camera_xyz": [round(x_c, 6), round(y_c, 6), round(z_c, 6)],
                "world_xyz": [round(t_list[0], 4), round(t_list[1], 4), round(t_list[2], 4)]
            })

        self.destroy_subscription(self.snapshot_sub)

    def _image_cb(self, rgb_msg, depth_msg):
        if self._image_received:
            return
        self._image_received = True

        # ★ 直接存为 numpy 数组，不写文件
        self.rgb_data = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        self.depth_data = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding=depth_msg.encoding)

        # print(f"✅ 图像已加载到内存: RGB {self.rgb_data.shape}, Depth {self.depth_data.shape}")

    @property
    def finished(self):
        return self._snapshot_received and self._image_received


# ==================== 主函数 ====================
def main():
    rclpy.init()
    node = RGBDAndSnapshotSaver()

    # 等待快照和图像都到齐
    while rclpy.ok() and not node.finished:
        rclpy.spin_once(node, timeout_sec=0.1)

    snapshot_data = node._snapshot_data

    # 加载模型并推理（直接从内存取数据）
    model_path = "mujoco_env/main/xiaoman1/module/best_model0.pth"  # ★ 改成你的路径
    inferencer = RGBDInferencer(model_path=model_path, device='cuda')

    # ★ 直接用内存中的 rgb_data 和 depth_data，不读文件
    results = inferencer.predict(node.rgb_data, node.depth_data, depth_in_meters=False)

    # 打印对比结果





    name_li = ["beaker1", "graduated_cylinder", "erlenmeyer_flask"]
    print("\n" + "="*100)
    print(f"{'预测 vs 真实对比':^100s}")
    print("="*100)

    for i in range(len(results)):
        print(f"\n📦 物体 {i+1} ({name_li[results[i]['object_id']]})")
        print(f"   预测世界坐标: x={results[i]['world_xyz'][0]:.3f}m, y={results[i]['world_xyz'][1]:.3f}m, z={results[i]['world_xyz'][2]:.3f}m")
        print(f"   真实世界坐标: x={snapshot_data[i]['world_xyz'][0]:.3f}m, y={snapshot_data[i]['world_xyz'][1]:.3f}m, z={snapshot_data[i]['world_xyz'][2]:.3f}m")
        
        pred_w = np.array(results[i]['world_xyz'])
        true_w = np.array(snapshot_data[i]['world_xyz'])
        error = np.linalg.norm(pred_w - true_w) * 100
        print(f"   🌍 3D 定位误差: {error:.1f} cm")
        print("-" * 100)
    
# ==================== 替换掉的画图代码开始 ====================
    import cv2 as cv
    img_size = 800
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 收集所有坐标（预测+真实），用来自动算缩放比例和原点偏移
    all_x = [0.0]
    all_y = [0.0]
    for i in range(3):
        all_x.extend([results[i]['world_xyz'][0], snapshot_data[i]['world_xyz'][0]])
        all_y.extend([results[i]['world_xyz'][1], snapshot_data[i]['world_xyz'][1]])
        
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    margin = 0.2  # 留白 20%
    span_x = (max_x - min_x) or 1.0
    span_y = (max_y - min_y) or 1.0
    scale = (img_size * (1 - margin*2)) / max(span_x, span_y)
    ox = int(img_size/2 - (max_x+min_x)/2 * scale)
    oy = int(img_size/2 + (max_y+min_y)/2 * scale)  # Y轴向下为正，所以加

    # 画原点和轴
    cv.circle(img, (ox, oy), 5, (0,0,0), -1)
    cv.line(img, (ox, 0), (ox, img_size), (200,200,200), 1)
    cv.line(img, (0, oy), (img_size, oy), (200,200,200), 1)
    cv.putText(img, "Origin", (ox+10, oy-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


        
    for i in range(3):
        pred_x, pred_y = results[i]['world_xyz'][0], results[i]['world_xyz'][1]
        true_x, true_y = snapshot_data[i]['world_xyz'][0], snapshot_data[i]['world_xyz'][1]
        name = name_li[results[i]['object_id']]
        
        px_pred = int(ox + pred_x * scale)
        py_pred = int(oy - pred_y * scale)
        px_true = int(ox + true_x * scale)
        py_true = int(oy - true_y * scale)
        
        # 真实位置（绿色空心圆）
        cv.circle(img, (px_true, py_true), 12, (0, 255, 0), 2)
        # 预测位置（红色实心三角）
        cv.drawMarker(img, (px_pred, py_pred), (0, 0, 255), cv.MARKER_TRIANGLE_UP, 15, 2)
        # 误差连线
        cv.line(img, (px_true, py_true), (px_pred, py_pred), (100,100,100), 1, cv.LINE_AA)
        
        cv.putText(img, f"{name} P", (px_pred+10, py_pred), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv.putText(img, f"{name} GT", (px_true+10, py_true), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,150,0), 1)

    cv.imwrite("simple_2d.png", img)
    print("saved: simple_2d.png")





    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()