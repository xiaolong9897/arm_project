#!/usr/bin/python3
"""
从归档视频重建 ROS2 MCAP bag（仅 RGB 图像 topic）。

输入目录要求（由 mcap_export_archive.py 生成）：
  archive_dir/
    topics/
      ee_camera__rgb__image_raw/
        video.mp4
        frames.jsonl.gz
      external_camera__rgb__image_raw/
        video.mp4
        frames.jsonl.gz

输出：
  output_bag_dir/
    metadata.yaml
    *.mcap

说明：
- 默认读取 frames.jsonl.gz 的 header_time_ns（缺失时用 bag_time_ns）作为写入时间戳。
- 发布消息类型为 sensor_msgs/msg/Image，encoding=bgr8。
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image


def _read_frames_index(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"frames index not found: {path}")
    records: List[Dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"no records in: {path}")
    return records


def _choose_timestamp_ns(rec: Dict) -> int:
    if rec.get("header_time_ns") is not None:
        return int(rec["header_time_ns"])
    if rec.get("bag_time_ns") is not None:
        return int(rec["bag_time_ns"])
    raise RuntimeError("frame record missing both header_time_ns and bag_time_ns")


def _build_image_msg_bgr8(frame_bgr, stamp_ns: int, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    msg.header.frame_id = frame_id or ""
    msg.height = int(frame_bgr.shape[0])
    msg.width = int(frame_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(msg.width * 3)
    msg.data = frame_bgr.tobytes()
    return msg


def _iter_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _prepare_topic_specs(archive_dir: Path) -> List[Tuple[str, Path, Path]]:
    specs = [
        (
            "/ee_camera/rgb/image_raw",
            archive_dir / "topics" / "ee_camera__rgb__image_raw" / "video.mp4",
            archive_dir / "topics" / "ee_camera__rgb__image_raw" / "frames.jsonl.gz",
        ),
        (
            "/external_camera/rgb/image_raw",
            archive_dir / "topics" / "external_camera__rgb__image_raw" / "video.mp4",
            archive_dir / "topics" / "external_camera__rgb__image_raw" / "frames.jsonl.gz",
        ),
    ]
    for topic, video, index in specs:
        if not video.exists():
            raise FileNotFoundError(f"{topic} video not found: {video}")
        if not index.exists():
            raise FileNotFoundError(f"{topic} frames index not found: {index}")
    return specs


def rebuild_rgb_bag(archive_dir: Path, output_bag_dir: Path, overwrite: bool = False) -> Dict[str, int]:
    if output_bag_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output bag dir exists: {output_bag_dir}")
        import shutil
        shutil.rmtree(output_bag_dir)

    topic_specs = _prepare_topic_specs(archive_dir)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    for topic_name, _, _ in topic_specs:
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic_name,
                type="sensor_msgs/msg/Image",
                serialization_format="cdr",
                offered_qos_profiles="",
            )
        )

    stats = {"total_messages": 0}

    for topic_name, video_path, frames_path in topic_specs:
        frame_records = _read_frames_index(frames_path)
        frame_iter = _iter_video_frames(video_path)

        written = 0
        for idx, rec in enumerate(frame_records):
            try:
                frame_bgr = next(frame_iter)
            except StopIteration:
                break

            stamp_ns = _choose_timestamp_ns(rec)
            frame_id = str(rec.get("header_frame_id") or "")
            msg = _build_image_msg_bgr8(frame_bgr, stamp_ns, frame_id)
            writer.write(topic_name, serialize_message(msg), int(stamp_ns))
            written += 1
            stats["total_messages"] += 1

            if stats["total_messages"] % 1000 == 0:
                print(f"written {stats['total_messages']} image messages...", flush=True)

        # 统计不一致
        # OpenCV 可能比 index 少/多帧（极少见），这里显式打印便于排查。
        index_count = len(frame_records)
        stats[f"{topic_name}_index_frames"] = index_count
        stats[f"{topic_name}_written_frames"] = written

        # consume rest frames quickly to compare if video has extras
        extras = 0
        for _ in frame_iter:
            extras += 1
        stats[f"{topic_name}_extra_video_frames"] = extras

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild ROS2 MCAP bag from archive RGB videos")
    parser.add_argument("archive_dir", type=Path, help="archive dir from mcap_export_archive.py")
    parser.add_argument("output_bag_dir", type=Path, help="output rosbag directory (mcap)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output directory if exists")
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    output_bag_dir = args.output_bag_dir.resolve()

    stats = rebuild_rgb_bag(archive_dir, output_bag_dir, overwrite=args.overwrite)

    print("=" * 70)
    print("Rebuild finished")
    print("=" * 70)
    print(f"archive: {archive_dir}")
    print(f"output:  {output_bag_dir}")
    print(f"total messages: {stats['total_messages']}")
    for key in sorted(k for k in stats.keys() if k != "total_messages"):
        print(f"{key}: {stats[key]}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
