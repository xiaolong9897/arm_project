#!/usr/bin/env python3
"""像素坐标 + 深度图 -> 世界坐标（类封装版）"""

import cv2
import mujoco
import numpy as np
from pathlib import Path


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

        print(f"像素 ({u}, {v}) -> 世界坐标 {p_world}, 深度 {z_c:.4f}m")
        return p_world, z_c


# ========== CLI ==========
if __name__ == "__main__":
    projector = PixelToWorld(
        depth_path="./mujoco_env/main/xiaoman1/images/depth_raw.png",
    )

    p_world, depth = projector.back_project(u=401, v=280)
    print(f"结果: {p_world} {depth}")