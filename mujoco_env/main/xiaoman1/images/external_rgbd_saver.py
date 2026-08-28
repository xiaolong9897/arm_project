import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class ExternalRGBDSaver(Node):
    def __init__(self):
        super().__init__('external_rgbd_saver')
        self.bridge = CvBridge()
        self.counter = 0

        self.save_dir = os.path.expanduser('~/study/Arm_Project/rgbd_saved_external')
        os.makedirs(os.path.join(self.save_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'depth'), exist_ok=True)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        sub_rgb = message_filters.Subscriber(
            self, Image, '/external_camera/rgb/image_raw', qos_profile=qos_profile
        )
        sub_depth = message_filters.Subscriber(
            self, Image, '/external_camera/depth/image_raw', qos_profile=qos_profile
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info("External RGB-D Saver Started. Saving .npy + visual PNG [500-1500 mm] ...")

    def callback(self, rgb_msg, depth_msg):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding=depth_msg.encoding)

        idx = f'{self.counter:04d}'

        # ===== RGB：原始 .npy + 可视化 .png =====
        np.save(os.path.join(self.save_dir, 'rgb', f'{idx}.npy'), cv_rgb)
        cv2.imwrite(os.path.join(self.save_dir, 'rgb', f'{idx}.png'), cv_rgb)

        # ===== Depth：原始 .npy（完整保留）+ 可视化伪彩色 PNG（500-1500 区间）=====
        np.save(os.path.join(self.save_dir, 'depth', f'{idx}.npy'), cv_depth)
        depth_vis = self._depth_to_colormap(cv_depth, min_mm=500, max_mm=1500)
        cv2.imwrite(os.path.join(self.save_dir, 'depth', f'{idx}.png'), depth_vis)

        self.get_logger().info(
            f'Saved frame {self.counter} | rgb={cv_rgb.dtype} depth={cv_depth.dtype}'
        )
        self.counter += 1

    def _depth_to_colormap(self, depth, min_mm=500, max_mm=1500):
        """深度可视化：仅 min_mm~max_mm 区间映射为 Jet 伪彩色，其余黑色"""
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

def main(args=None):
    rclpy.init(args=args)
    node = ExternalRGBDSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()