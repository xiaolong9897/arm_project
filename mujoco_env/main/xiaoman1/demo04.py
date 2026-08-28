#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        self.bridge = CvBridge()
        self.count = 0
        self._saved = False  # ← 加这个

        self.save_dir = './saved_images'
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

        self.get_logger().info(f'正在监听话题: {topic}')

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
            self.get_logger().info(f'已保存: {filename}')

        self._saved = True  # ← 加这个
        self.destroy_subscription(self.subscription)  # ← 只停订阅

    def destroy_node(self):
        self.get_logger().info(f'总共保存了 {self.count} 张图片')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()

    # 手动 spin_once 循环，保存完就退出
    while rclpy.ok() and not node._saved:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()