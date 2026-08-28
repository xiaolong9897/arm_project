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
import numpy as np
import cv2  # 可选，用于读取图片
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
import json
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import mujoco
# import torch
from pathlib import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import glob
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

# ==================== Network (保持不变，确保 dim_per_obj=9) ====================
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
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.total_dim)
        )
        
        nn.init.xavier_uniform_(self.fusion[-1].weight, gain=0.01)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, rgb, depth):
        depth_3ch = depth.repeat(1, 3, 1, 1)
        rgb_feat = self.rgb_backbone(rgb)
        depth_feat = self.depth_backbone(depth_3ch)
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        out = self.fusion(fused)
        out = out.view(-1, self.num_objects, self.dim_per_obj)
        return out
class RGBDAndSnapshotSaver(Node):
    def __init__(self):
        super().__init__('rgbd_and_snapshot_saver')
        self.bridge = CvBridge()

        # 状态标志
        self._snapshot_received = False
        self._image_received = False
        self._snapshot_data = None

        self.save_dir = os.path.expanduser('~/study/Arm_Project/rgbd_saved_external_verify')
        os.makedirs(os.path.join(self.save_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'lables'), exist_ok=True)
        # os.makedirs(os.path.join(self.save_dir, 'lables_rgb'), exist_ok=True)
        existing = glob.glob(os.path.join(self.save_dir, 'lables', '*.json'))
        if existing:
            # 文件名形如 00000.json -> 取 '.' 前的主名
            ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in existing]
            name_id = max(ids) + 1
        else:
            name_id = 0


        # name_id=0
        # rgb npy 保存地址
        self.rgb_npy_address =os.path.join(self.save_dir, 'rgb', f'{name_id:05d}_rgb.npy')

        # rgb png 保存地址
        self.rgb_png_address =os.path.join(self.save_dir, 'rgb', f'{name_id:05d}_rgb.png')

        # depth npy 保存地址
        self.depth_npy_address =os.path.join(self.save_dir, 'depth', f'{name_id:05d}_depth.npy')

        # depth png 保存地址
        self.depth_png_address =os.path.join(self.save_dir, 'depth', f'{name_id:05d}_depth.png')
        # 标签保存位置
        self.lables_address =os.path.join(self.save_dir, 'lables', f'{name_id:05d}.json')




        # 标注的照片地址
        self.rgb_lables_png_address =os.path.join(self.save_dir, 'lables_rgb', f'{name_id:05d}_rgb_标注.png')


        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        # ===== 订阅 /task_scene_snapshot =====
        self.snapshot_sub = self.create_subscription(
            String, '/task_scene_snapshot', self._snapshot_cb, 10
        )

        # ===== 订阅 RGB-D（时间同步）=====
        sub_rgb = message_filters.Subscriber(
            self, Image, '/external_camera/rgb/image_raw', qos_profile=qos
        )
        sub_depth = message_filters.Subscriber(
            self, Image, '/external_camera/depth/image_raw', qos_profile=qos
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self._image_cb)

        # self.get_logger().info("Waiting for snapshot AND image pair...")

    # ---------- 快照回调 ----------
    def _snapshot_cb(self, msg: String):
        if self._snapshot_received:
            return
        self._snapshot_received = True

        data = json.loads(msg.data)
        data_list = data["movable_objects"]
        data_list[0]["position_xyz"][2] += 0.07
        data_list[1]["position_xyz"][2] += 0.11
        data_list[2]["position_xyz"][2] += 0.1075

        self._snapshot_data = []
        
        name_id={
            "beaker1":              0,
            "graduated_cylinder":   1,
            "erlenmeyer_flask":     2
        }
        
        for item in data_list:
            t_list = item["position_xyz"]
            for j in range(len(t_list)):
                t_list[j] = round(t_list[j], 3)
            # print(t_list)
            projector = WordToCam(
                p_world = t_list,
                image_path = self.rgb_png_address,
                output_path = self.rgb_lables_png_address,
            )
            u, v, val_depth, ok = projector.project()
            # projector.draw(u, v)

            # print(u, v, val_depth, ok)
            pixel_li=[round(u,2),round(v,2),round(val_depth,5)]

            z_c = val_depth
            x_c = (u - projector.cx) * z_c / projector.fx
            y_c = (v - projector.cy) * z_c / projector.fy
            camera_xyz = [
            round(float(x_c), 6),
            round(float(y_c), 6),
            round(float(z_c), 6),
             ]
            # print("camera_xyz =", camera_xyz)

            pixel_uv_depth=[u ,v ,val_depth]
            # print(pixel_uv_depth)
            self._snapshot_data.append({
                "name": item["body_name"],
                "class_id":name_id[item["body_name"]],
                "pixel_uv_depth":pixel_uv_depth,
                "camera_xyz": camera_xyz,
                "world_xyz": item["position_xyz"]
            })

        # self.get_logger().info(f"Snapshot received: {len(self._snapshot_data)} objects")
        self.destroy_subscription(self.snapshot_sub)

    # ---------- 图像回调 ----------
    def _image_cb(self, rgb_msg, depth_msg):
        if self._image_received:
            return
        self._image_received = True

        # 保存 RGB
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        # np.save(os.path.join(self.save_dir, 'rgb', 'frame_rgb.npy'), cv_rgb)
        np.save(self.rgb_npy_address, cv_rgb)
        # cv2.imwrite(os.path.join(self.save_dir, 'rgb', 'frame_rgb.png'), cv_rgb)
        cv2.imwrite(self.rgb_png_address, cv_rgb)

        # 保存 Depth
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding=depth_msg.encoding)
        # np.save(os.path.join(self.save_dir, 'depth', 'frame_depth.npy'), cv_depth)
        np.save(self.depth_npy_address, cv_depth)
        depth_vis = self._depth_to_colormap(cv_depth)
        # cv2.imwrite(os.path.join(self.save_dir, 'depth', 'frame_depth.png'), depth_vis)
        cv2.imwrite(self.depth_png_address, depth_vis)

        # self.get_logger().info(
        #     f"Image pair saved. RGB: {cv_rgb.dtype}, Depth: {cv_depth.dtype}"
        # )

    # ---------- 深度可视化（500-1500mm）----------
    def _depth_to_colormap(self, depth, min_mm=500, max_mm=1500):
        if depth.dtype in (np.float32, np.float64):
            d = depth.astype(np.float32) * 1000.0
        else:
            d = depth.astype(np.float32)
        d[~np.isfinite(d)] = 0.0

        mask = (d >= min_mm) & (d <= max_mm)
        norm = np.zeros(d.shape, dtype=np.uint8)
        if mask.any():
            d_masked = d[mask]
            lo, hi = d_masked.min(), d_masked.max()
            if hi > lo:
                norm[mask] = ((d_masked - lo) / (hi - lo) * 255.0).astype(np.uint8)
            else:
                norm[mask] = 128
        colormap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        colormap[~mask] = 0
        return colormap

    # ---------- 是否全部完成 ----------
    @property
    def finished(self):
        return self._snapshot_received and self._image_received

# ==================== 推理类 ====================
class RGBDInferencer:
    def __init__(self, model_path, device='cuda', mask_roi=(180, 480, 340, 640)):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mask_roi = mask_roi
        
        # 1. 加载模型（★ dim_per_obj 改为 9）
        self.model = DoubleStreamResNet(num_objects=3, dim_per_obj=9, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        # print(f"✅ 模型已从 {model_path} 加载")
    
    def preprocess(self, rgb_np, depth_np, depth_in_meters=False):
        """
        rgb_np: numpy array, shape (480, 640, 3), 值域 0-255
        depth_np: numpy array, shape (480, 640), 单位毫米(mm)或米(m)
        depth_in_meters: 如果 depth_np 单位已经是米，设为 True
        """
        # RGB: HWC -> CHW -> 归一化到 0-1
        rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        # Depth: HW -> 1HW
        depth = torch.from_numpy(np.asarray(depth_np)).float().unsqueeze(0)
        
        # 单位转换：如果不是米，则毫米转米
        if not depth_in_meters:
            depth = depth / 1000.0
            
        # 应用 mask（必须和训练时一致）
        if self.mask_roi is not None:
            h_start, h_end, w_start, w_end = self.mask_roi
            mask = torch.zeros_like(depth)
            mask[:, h_start:h_end, w_start:w_end] = 1
            rgb = rgb * mask
            depth = depth * mask
        
        # 增加 batch 维度并送到 device
        rgb = rgb.unsqueeze(0).to(self.device)
        depth = depth.unsqueeze(0).to(self.device)
        return rgb, depth
    
    def predict(self, rgb_np, depth_np, depth_in_meters=False):
        rgb_tensor, depth_tensor = self.preprocess(rgb_np, depth_np, depth_in_meters)
        
        with torch.no_grad():
            pred = self.model(rgb_tensor, depth_tensor)  # [1, 3, 9]
            pred = pred.cpu().numpy()[0]  # -> [3, 9]
        
        # 后处理：解归一化（★ 修改：解析全部 9 维）
        results = []
        for i in range(3):
            # ★ 解包 9 个预测值
            u_norm, v_norm, d, x_cam, y_cam, z_cam, x_world, y_world, z_world = pred[i]
            
            # 像素坐标还原
            u_pixel = u_norm * 640.0
            v_pixel = v_norm * 480.0
            
            results.append({
                'object_id': i,
                'pixel_uv': (round(u_pixel, 1), round(v_pixel, 1)),
                'depth_m': round(d, 4),
                'camera_xyz': (round(x_cam, 4), round(y_cam, 4), round(z_cam, 4)),
                # ★ 新增：world_xyz
                'world_xyz': (round(x_world, 4), round(y_world, 4), round(z_world, 4))
            })
        
        return results



def main():
    rclpy.init()
    node = RGBDAndSnapshotSaver()

    # spin_once 等待，和你之前的风格一模一样
    while rclpy.ok() and not node.finished:
        rclpy.spin_once(node, timeout_sec=0.1)

    # 退出循环后保存快照 JSON
    if node._snapshot_data is not None:
        # snapshot_path = os.path.join(node.save_dir, 'snapshot.json')
        # print(node._snapshot_data)
        # print(json.dumps(node._snapshot_data, indent=2, ensure_ascii=False))
        with open(node.lables_address, 'w') as f:
            json.dump(node._snapshot_data, f, indent=2)
        # print(f"Snapshot saved to {snapshot_path}")

    node.destroy_node()
    rclpy.shutdown()
    return node.lables_address,node._snapshot_data


if __name__ == '__main__':
    dir,label_datas=main()
    # print(dir)
    # print(label_datas)
    # print(json.dumps(label_datas, indent=2, ensure_ascii=False))
    model1_path="mujoco_env/main/xiaoman1/module/best_model5.pth"
    inferencer = RGBDInferencer(
        model_path=model1_path,  # ★ 修改为你的实际路径
        device='cuda'
    )
    # dir = "/home/xiaoman/study/Arm_Project/rgbd_saved_external_verify/lables/00001.json"
      
    # print(dir)
    # print(json.dumps(data_dic, indent=2, ensure_ascii=False))
    file_id = os.path.splitext(os.path.basename(dir))[0]
    # print(file_id)
    rgb_path = os.path.join("./rgbd_saved_external_verify/rgb", f"{file_id}_rgb.npy")
    depth_path = os.path.join("./rgbd_saved_external_verify/depth", f"{file_id}_depth.npy")
    lables_path = os.path.join("./rgbd_saved_external_verify/lables", f"{file_id}.json")
    rgb_data = np.load(rgb_path).astype(np.float32)
    depth_data = np.load(depth_path).astype(np.float32)
    results = inferencer.predict(rgb_data, depth_data, depth_in_meters=False)
    # print(results)
    # print(type(results))
    # print(json.dumps(results, indent=2, ensure_ascii=False))
    for i in range(len(results)):
            # print(results[i])
            # print(label_datas[i])
            # name_li=["beaker1","beaker1","erlenmeyer_flask"]
            # print(json.dumps(results[i], indent=2, ensure_ascii=False))
            # print(f"预测种类：{name_li[results[i]['object_id']]}")
            # print(f"真实种类：{label_datas[i]['name']}")
            # print(f"   预测像素坐标: u={results[i]['pixel_uv'][0]:.1f}, v={results[i]['pixel_uv'][1]:.1f}")
            # print(f"   真实像素坐标: u={label_datas[i]['pixel_uv_depth'][0]:.1f}, v={label_datas[i]['pixel_uv_depth'][1]:.1f}")
            # print(f"   预测深度: {results[i]['depth_m']:.3f} m")
            # print(f"   真实深度: {label_datas[i]['pixel_uv_depth'][2]:.3f} m")
            # print(f"   预测相机坐标: x={results[i]['camera_xyz'][0]:.3f}m, y={results[i]['camera_xyz'][1]:.3f}m, z={results[i]['camera_xyz'][2]:.3f}m")
            # print(f"   真实相机坐标: x={label_datas[i]['camera_xyz'][0]:.3f}m, y={label_datas[i]['camera_xyz'][1]:.3f}m, z={label_datas[i]['camera_xyz'][2]:.3f}m")
            print(f"   预测世界坐标: x={results[i]['world_xyz'][0]:.3f}m, y={results[i]['world_xyz'][1]:.3f}m, z={results[i]['world_xyz'][2]:.3f}m")
            print(f"   真实世界坐标: x={label_datas[i]['world_xyz'][0]:.3f}m, y={label_datas[i]['world_xyz'][1]:.3f}m, z={label_datas[i]['world_xyz'][2]:.3f}m")
            print("-"*100)




    





