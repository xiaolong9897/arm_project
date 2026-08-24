#!/usr/bin/python3
"""
ROS2 bag 统一后处理工具。

功能：
- 一次遍历完成话题重命名
- 一次遍历完成 RGB 图像压缩
- 输出新的 bag 目录（自动生成正确 metadata.yaml）

说明：
- 该工具运行在 ROS2 环境下，使用 rosbag2_py 读写 bag。
- `cdr` 只是消息序列化格式；图像是否压缩取决于消息类型。
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
    """将 ROS Image 转为 OpenCV BGR。"""
    height = int(image_msg.height)
    width = int(image_msg.width)
    step = int(image_msg.step)
    encoding = str(image_msg.encoding).lower()

    raw = np.frombuffer(image_msg.data, dtype=np.uint8)
    rows = raw.reshape(height, step)

    if encoding in {"rgb8", "bgr8"}:
        image = rows[:, : width * 3].reshape(height, width, 3)
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    if encoding == "mono8":
        gray = rows[:, :width].reshape(height, width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"暂不支持压缩的图像编码: {image_msg.encoding}")


def postprocess_bag(
    input_bag_dir: Path,
    output_bag_dir: Path,
    topic_rename_map: Dict[str, str],
    image_compress_map: Dict[str, str],
    image_codec: str = "png",
    jpeg_quality: int = 85,
) -> Dict[str, int]:
    """统一执行 bag 后处理。"""
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

    stats = {
        "total_messages": 0,
        "renamed_messages": 0,
        "compressed_messages": 0,
    }

    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}

    for src_topic, topic_type in topic_types.items():
        renamed_topic = topic_rename_map.get(src_topic, src_topic)

        if src_topic in image_compress_map:
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=image_compress_map[src_topic],
                    type="sensor_msgs/msg/CompressedImage",
                    serialization_format="cdr",
                    offered_qos_profiles="",
                )
            )
            continue

        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=renamed_topic,
                type=topic_type,
                serialization_format="cdr",
                offered_qos_profiles="",
            )
        )

    image_type = get_message("sensor_msgs/msg/Image")

    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        stats["total_messages"] += 1

        if topic_name in image_compress_map:
            image_msg = deserialize_message(data, image_type)
            bgr = _image_msg_to_cv_bgr(image_msg)

            if image_codec == "png":
                ext = ".png"
                encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
                format_name = "png"
            else:
                ext = ".jpg"
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                format_name = "jpeg"

            ok, encoded = cv2.imencode(ext, bgr, encode_params)
            if not ok:
                raise RuntimeError(f"图像编码失败: {topic_name}")

            compressed_msg = CompressedImage()
            compressed_msg.header = image_msg.header
            compressed_msg.format = format_name
            compressed_msg.data = encoded.tobytes()

            writer.write(
                image_compress_map[topic_name],
                serialize_message(compressed_msg),
                timestamp,
            )
            stats["compressed_messages"] += 1
            continue

        dst_topic = topic_rename_map.get(topic_name, topic_name)
        if dst_topic != topic_name:
            stats["renamed_messages"] += 1
        writer.write(dst_topic, data, timestamp)

        if stats["total_messages"] % 1000 == 0:
            print(
                f"已处理 {stats['total_messages']} 条消息，"
                f"重命名 {stats['renamed_messages']} 条，"
                f"压缩图像 {stats['compressed_messages']} 条...",
                flush=True,
            )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS2 bag 统一后处理工具")
    parser.add_argument("input_bag_dir", type=Path, help="输入 bag 目录")
    parser.add_argument("output_bag_dir", type=Path, help="输出 bag 目录")
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        help="重命名规则，格式: /old_topic:/new_topic",
    )
    parser.add_argument(
        "--compress-topic",
        action="append",
        default=[],
        help="图像压缩规则，格式: /raw/topic:/compressed/topic",
    )
    parser.add_argument(
        "--image-codec",
        choices=["png", "jpeg"],
        default="png",
        help="压缩图像编码。png 为无损，jpeg 为有损。",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG 质量 (1-100)")
    args = parser.parse_args()

    topic_rename_map: Dict[str, str] = {}
    for rule in args.rename:
        if ":" not in rule:
            raise ValueError(f"无效重命名规则: {rule}")
        src, dst = rule.split(":", 1)
        topic_rename_map[src] = dst

    image_compress_map: Dict[str, str] = {}
    for rule in args.compress_topic:
        if ":" not in rule:
            raise ValueError(f"无效图像压缩规则: {rule}")
        src, dst = rule.split(":", 1)
        image_compress_map[src] = dst

    stats = postprocess_bag(
        input_bag_dir=args.input_bag_dir,
        output_bag_dir=args.output_bag_dir,
        topic_rename_map=topic_rename_map,
        image_compress_map=image_compress_map,
        image_codec=args.image_codec,
        jpeg_quality=args.jpeg_quality,
    )
    print("=" * 70)
    print("✅ 后处理完成")
    print("=" * 70)
    print(f"总消息数: {stats['total_messages']}")
    print(f"重命名消息数: {stats['renamed_messages']}")
    print(f"压缩图像消息数: {stats['compressed_messages']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
