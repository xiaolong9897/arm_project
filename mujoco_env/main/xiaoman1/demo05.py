#!/usr/bin/env python3
"""
ROS2 深度图订阅保存节点（只保存一张就退出）
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class DepthSaver(Node):
    def __init__(self):
        super().__init__('depth_saver')

        self.bridge = CvBridge()
        self.count = 0
        self._saved = False  # ← 加这个

        self.save_dir = './saved_depth'
        os.makedirs(self.save_dir, exist_ok=True)

        topic = self.declare_parameter('topic', '/external_camera/depth/image_raw').value

        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(Image, topic, self.callback, qos)
        self.get_logger().info(f'正在监听深度话题: {topic}')

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
        self.get_logger().info(f'原始深度保存: {raw_path}')

        # ---- 2. 彩色可视化 ----
        depth_float = depth.astype(np.float32)
        depth_clipped = np.clip(depth_float, 500, 1500)
        depth_norm = ((depth_clipped - 500) / (1500 - 500) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        vis_path = os.path.join(self.save_dir, f'depth_vis.jpg')
        cv2.imwrite(vis_path, depth_colored)
        self.get_logger().info(f'彩色可视化保存: {vis_path}')

        self.count += 1
        self._saved = True  # ← 加这个
        self.destroy_subscription(self.subscription)  # ← 只停订阅

    def destroy_node(self):
        self.get_logger().info(f'总共保存了 {self.count} 组深度图')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthSaver()

    # 手动 spin_once，收到一张就退出循环
    while rclpy.ok() and not node._saved:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()