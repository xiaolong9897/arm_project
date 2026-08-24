#!/usr/bin/python3
"""
ROS2 bag 图像后处理压缩工具。

功能：
- 读取 rosbag2/MCAP 目录
- 将指定的 sensor_msgs/Image RGB 话题压缩为 sensor_msgs/CompressedImage
- 写入新的 bag 目录
- 未压缩的话题原样透传

示例：
    /usr/bin/python3 rosbag_image_compressor.py input_bag output_bag \
        --compress-topic /ee_camera/rgb/image_raw:/ee_camera/rgb/image_raw/compressed \
        --compress-topic /external_camera/rgb/image_raw:/external_camera/rgb/image_raw/compressed
"""

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Image


def _image_msg_to_cv_bgr(image_msg: Image) -> np.ndarray:
    """将 ROS Image 转为 OpenCV BGR 图像。"""
    height = int(image_msg.height)
    width = int(image_msg.width)
    step = int(image_msg.step)
    encoding = str(image_msg.encoding).lower()

    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"暂不支持压缩的图像编码: {image_msg.encoding}")

    raw = np.frombuffer(image_msg.data, dtype=np.uint8)
    rows = raw.reshape(height, step)
    pixel_bytes = rows[:, : width * 3]
    image = pixel_bytes.reshape(height, width, 3)

    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def compress_image_topics_in_bag(
    input_bag_dir: Path,
    output_bag_dir: Path,
    topic_mapping: Dict[str, str],
    jpeg_quality: int = 85,
) -> None:
    """压缩 bag 中指定 RGB 图像话题。"""
    if not input_bag_dir.exists():
        raise FileNotFoundError(f"输入 bag 目录不存在: {input_bag_dir}")
    if output_bag_dir.exists():
        raise FileExistsError(f"输出 bag 目录已存在: {output_bag_dir}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    passthrough_topics = {}

    for topic_name, topic_type in topic_types.items():
        if topic_name in topic_mapping:
            compressed_topic = topic_mapping[topic_name]
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=compressed_topic,
                    type="sensor_msgs/msg/CompressedImage",
                    serialization_format="cdr",
                    offered_qos_profiles="",
                )
            )
        else:
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=topic_name,
                    type=topic_type,
                    serialization_format="cdr",
                    offered_qos_profiles="",
                )
            )
            passthrough_topics[topic_name] = topic_type

    image_type = get_message("sensor_msgs/msg/Image")

    processed_count = 0
    compressed_count = 0
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        processed_count += 1

        if topic_name not in topic_mapping:
            writer.write(topic_name, data, timestamp)
            continue

        image_msg = deserialize_message(data, image_type)
        bgr = _image_msg_to_cv_bgr(image_msg)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError(f"JPEG 编码失败: {topic_name}")

        compressed_msg = CompressedImage()
        compressed_msg.header = image_msg.header
        compressed_msg.format = "jpeg"
        compressed_msg.data = encoded.tobytes()

        writer.write(
            topic_mapping[topic_name],
            serialize_message(compressed_msg),
            timestamp,
        )
        compressed_count += 1

        if processed_count % 1000 == 0:
            print(
                f"已处理 {processed_count} 条消息，"
                f"压缩图像 {compressed_count} 条...",
                flush=True,
            )

    print(f"✅ 压缩完成: 总消息 {processed_count}，压缩图像 {compressed_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS2 bag 图像后处理压缩工具")
    parser.add_argument("input_bag_dir", type=Path, help="输入 bag 目录")
    parser.add_argument("output_bag_dir", type=Path, help="输出 bag 目录")
    parser.add_argument(
        "--compress-topic",
        action="append",
        default=[],
        help="压缩规则，格式: /raw/topic:/compressed/topic",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG 质量 (1-100)")
    args = parser.parse_args()

    topic_mapping: Dict[str, str] = {}
    for rule in args.compress_topic:
        if ":" not in rule:
            raise ValueError(f"无效压缩规则: {rule}")
        src, dst = rule.split(":", 1)
        topic_mapping[src] = dst

    if not topic_mapping:
        raise ValueError("至少需要一个 --compress-topic 规则")

    compress_image_topics_in_bag(
        input_bag_dir=args.input_bag_dir,
        output_bag_dir=args.output_bag_dir,
        topic_mapping=topic_mapping,
        jpeg_quality=args.jpeg_quality,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
