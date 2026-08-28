#!/usr/bin/env python3
"""世界坐标 -> 像素坐标投影"""

import cv2
import mujoco
import numpy as np
from pathlib import Path


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

        print(f"投影像素 (u, v): ({u:.3f}, {v:.3f}), 深度: {p_cv[2]:.4f}m, 在图内: {ok}")
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
        print(f"已保存: {self.output_path}")


# ========== CLI ==========
if __name__ == "__main__":
    # 创建实例，一次性传入所有参数
    projector = WordToCam(
        p_world=[0.994, 0.19, 1.07],
        image_path="./mujoco_env/main/xiaoman1/frame_image.jpg",
        output_path="./mujoco_env/main/xiaoman1/frame_image_projected12.jpg",
    )

    # 调用实例方法
    u, v, _, ok = projector.project()
    projector.draw(u, v)
    