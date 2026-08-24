#!/usr/bin/python3
"""
将 ROS2 rosbag/MCAP 导出为紧凑归档格式。

设计目标：
1. 图像话题导出为视频，显著减少“逐帧图片文件”带来的目录碎片与空间浪费。
2. 保留逐帧精确时间戳与 ROS 头信息，便于后续恢复为 ROS2 topic 数据流。
3. 普通话题导出为压缩 JSONL 索引，兼顾可读性与体积。

输出结构示例：
archive_xxx/
  manifest.json
  topics/
    ee_camera__rgb__image_raw__compressed/
      video.mkv
      frames.jsonl.gz
    joint_states_sim/
      messages.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Image


SYSTEM_PYTHON = "/usr/bin/python3"


def _enforce_system_python() -> None:
    """强制使用系统 Python，避免 rosbag2_py / ROS2 环境错位。"""
    current_python = os.path.realpath(sys.executable)
    expected_python = os.path.realpath(SYSTEM_PYTHON)
    if current_python != expected_python:
        raise SystemExit(
            "\n❌ 当前解释器不是系统 Python。\n"
            f"   当前: {current_python}\n"
            f"   要求: {expected_python}\n"
            f"   请改用: {SYSTEM_PYTHON} mujoco_env/tools/ros2_recorder/mcap_export_archive.py\n"
        )


_enforce_system_python()


def _normalize_input_bag_path(input_path: Path) -> Path:
    """
    将输入路径规范为 rosbag 目录。

    - 如果传入的是 bag 目录，直接返回。
    - 如果传入的是 .mcap 文件，则返回其所在目录。
    """
    if input_path.is_dir():
        metadata_path = input_path / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"目录中缺少 metadata.yaml: {input_path}")
        return input_path

    if input_path.is_file() and input_path.suffix.lower() in {".mcap", ".mca"}:
        bag_dir = input_path.parent
        metadata_path = bag_dir / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"MCAP 所在目录缺少 metadata.yaml: {bag_dir}")
        return bag_dir

    raise FileNotFoundError(f"无法识别输入路径: {input_path}")


def _sanitize_topic_name(topic_name: str) -> str:
    """将 topic 名转换为稳定的目录名。"""
    sanitized = topic_name.strip("/")
    if not sanitized:
        return "root_topic"
    return sanitized.replace("/", "__")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _header_stamp_to_ns(header: Any) -> Optional[int]:
    """将 ROS Header 的 stamp 转换为纳秒。"""
    if header is None or not hasattr(header, "stamp"):
        return None
    sec = int(getattr(header.stamp, "sec", 0))
    nanosec = int(getattr(header.stamp, "nanosec", 0))
    return sec * 1_000_000_000 + nanosec


def _image_msg_to_bgr_and_meta(image_msg: Image) -> tuple[np.ndarray, Dict[str, Any]]:
    """将 sensor_msgs/Image 转为 BGR 图像，并返回图像元数据。"""
    height = int(image_msg.height)
    width = int(image_msg.width)
    step = int(image_msg.step)
    encoding = str(image_msg.encoding).lower()

    raw = np.frombuffer(image_msg.data, dtype=np.uint8)
    rows = raw.reshape(height, step)

    if encoding == "bgr8":
        bgr = rows[:, : width * 3].reshape(height, width, 3).copy()
    elif encoding == "rgb8":
        rgb = rows[:, : width * 3].reshape(height, width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif encoding == "mono8":
        gray = rows[:, :width].reshape(height, width)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif encoding == "16uc1":
        depth_u16 = rows[:, : width * 2].view(np.uint16).reshape(height, width)
        bgr = _depth_u16_to_bgr(depth_u16)
    elif encoding == "32fc1":
        depth_f32 = rows[:, : width * 4].view(np.float32).reshape(height, width)
        bgr = _depth_f32_to_bgr(depth_f32)
    else:
        raise ValueError(f"暂不支持导出的视频图像编码: {image_msg.encoding}")

    meta = {
        "source_encoding": str(image_msg.encoding),
        "height": height,
        "width": width,
        "step": step,
        "channels": int(bgr.shape[2]) if bgr.ndim == 3 else 1,
    }
    return bgr, meta


def _depth_u16_to_bgr(depth_u16: np.ndarray) -> np.ndarray:
    """
    将 16UC1 深度图转换为便于导出视频的 BGR 可视化图。

    公式：
        d_norm = normalize(d_valid)
        d_u8 = uint8(255 * d_norm)
    其中仅对有效深度做归一化，0 视为无效值。
    """
    depth = np.asarray(depth_u16, dtype=np.uint16)
    valid = depth > 0
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        valid_depth = depth[valid].astype(np.float32)
        d_min = float(valid_depth.min())
        d_max = float(valid_depth.max())
        if d_max > d_min:
            vis_valid = np.clip((valid_depth - d_min) * 255.0 / (d_max - d_min), 0.0, 255.0).astype(np.uint8)
        else:
            vis_valid = np.full(valid_depth.shape, 255, dtype=np.uint8)
        vis[valid] = vis_valid
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def _depth_f32_to_bgr(depth_f32: np.ndarray) -> np.ndarray:
    """
    将 32FC1 米制深度图转换为便于导出视频的 BGR 可视化图。

    公式：
        d_mm = 1000 * d_m
        然后沿用 uint16 可视化归一化逻辑。
    """
    depth = np.asarray(depth_f32, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth_u16 = np.zeros(depth.shape, dtype=np.uint16)
    if np.any(valid):
        depth_u16[valid] = np.clip(np.rint(depth[valid] * 1000.0), 0.0, 65535.0).astype(np.uint16)
    return _depth_u16_to_bgr(depth_u16)


def _compressed_image_msg_to_bgr_and_meta(image_msg: CompressedImage) -> tuple[np.ndarray, Dict[str, Any]]:
    """将 sensor_msgs/CompressedImage 解码为 BGR 图像，并返回元数据。"""
    buffer = np.frombuffer(image_msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("CompressedImage 解码失败")

    meta = {
        "source_format": str(image_msg.format),
        "height": int(bgr.shape[0]),
        "width": int(bgr.shape[1]),
        "step": int(bgr.shape[1] * bgr.shape[2]),
        "channels": int(bgr.shape[2]),
    }
    return bgr, meta


def _to_builtin_jsonable(value: Any) -> Any:
    """将 numpy / bytes 等对象转换为可 JSON 序列化的基础类型。"""
    if isinstance(value, dict):
        return {str(k): _to_builtin_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return {
            "__type__": "bytes_omitted",
            "length": len(value),
        }
    return value


@dataclass
class VideoFormatSpec:
    name: str
    container: str
    extension: str
    codec: str
    lossless: bool
    description: str


VIDEO_FORMAT_SPECS: Dict[str, VideoFormatSpec] = {
    "mkv_ffv1": VideoFormatSpec(
        name="mkv_ffv1",
        container="MKV",
        extension=".mkv",
        codec="FFV1",
        lossless=True,
        description="严格无损，推荐归档母版",
    ),
    "mp4_x264_lossless": VideoFormatSpec(
        name="mp4_x264_lossless",
        container="MP4",
        extension=".mp4",
        codec="libx264-crf0",
        lossless=False,
        description="尽可能无损的 MP4，便于兼容播放器",
    ),
}


class FFmpegVideoWriter:
    """
    基于 ffmpeg stdin 管道的视频写入器。

    公式说明：
    - 逐帧输入的是 BGR 原始像素，shape = (H, W, 3)
    - 每帧字节数 = H * W * 3
    - ffmpeg 读取时使用 `-f rawvideo -pix_fmt bgr24 -s WxH -r FPS`
    """

    def __init__(self, output_path: Path, width: int, height: int, fps: float, spec: VideoFormatSpec):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.spec = spec
        self.process: Optional[subprocess.Popen] = None
        self._start()

    def _start(self) -> None:
        _ensure_parent(self.output_path)
        base_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps}",
            "-i",
            "-",
            "-an",
        ]

        if self.spec.name == "mkv_ffv1":
            codec_cmd = [
                "-c:v",
                "ffv1",
                "-level",
                "3",
            ]
        elif self.spec.name == "mp4_x264_lossless":
            codec_cmd = [
                "-c:v",
                "libx264",
                "-preset",
                "veryslow",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv444p",
                "-movflags",
                "+faststart",
            ]
        else:
            raise ValueError(f"未知视频格式: {self.spec.name}")

        cmd = base_cmd + codec_cmd + [str(self.output_path)]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, bgr: np.ndarray) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("ffmpeg 视频写入器未启动")
        if bgr.dtype != np.uint8:
            raise ValueError("视频输入必须为 uint8")
        if bgr.shape != (self.height, self.width, 3):
            raise ValueError(
                f"视频帧尺寸不匹配: 期望 {(self.height, self.width, 3)}，实际 {bgr.shape}"
            )
        self.process.stdin.write(bgr.tobytes())

    def close(self) -> None:
        if self.process is None:
            return
        stderr_output = b""
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            stderr_output = self.process.stderr.read() if self.process.stderr is not None else b""
            return_code = self.process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg 编码失败，退出码 {return_code}:\n{stderr_output.decode('utf-8', errors='ignore')}"
                )
        finally:
            if self.process.stderr is not None:
                self.process.stderr.close()
            self.process = None


@dataclass
class ImageTopicArchive:
    topic_name: str
    topic_type: str
    topic_dir: Path
    video_filename: str
    metadata_filename: str
    video_fps: float
    video_format_spec: VideoFormatSpec
    writer: Optional[FFmpegVideoWriter] = None
    frame_count: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    source_message_type: Optional[str] = None
    first_bag_time_ns: Optional[int] = None
    last_bag_time_ns: Optional[int] = None
    source_was_already_compressed: bool = False

    def _metadata_path(self) -> Path:
        return self.topic_dir / self.metadata_filename

    def append_frame(
        self,
        bgr: np.ndarray,
        bag_time_ns: int,
        header_time_ns: Optional[int],
        header_frame_id: str,
        source_meta: Dict[str, Any],
    ) -> None:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("导出视频的图像必须为 BGR 三通道")

        height, width = int(bgr.shape[0]), int(bgr.shape[1])
        if self.writer is None:
            self.width = width
            self.height = height
            self.writer = FFmpegVideoWriter(
                output_path=self.topic_dir / self.video_filename,
                width=width,
                height=height,
                fps=self.video_fps,
                spec=self.video_format_spec,
            )
        else:
            if self.width != width or self.height != height:
                raise RuntimeError(
                    f"图像尺寸发生变化，当前脚本不支持单个视频内分辨率切换: "
                    f"{self.topic_name} {self.width}x{self.height} -> {width}x{height}"
                )

        self.writer.write(bgr)
        self.frame_count += 1
        if self.first_bag_time_ns is None:
            self.first_bag_time_ns = int(bag_time_ns)
        self.last_bag_time_ns = int(bag_time_ns)

        frame_record = {
            "frame_index": self.frame_count - 1,
            "bag_time_ns": int(bag_time_ns),
            "header_time_ns": int(header_time_ns) if header_time_ns is not None else None,
            "header_frame_id": header_frame_id,
            "height": height,
            "width": width,
            "video_file": self.video_filename,
            "source_meta": _to_builtin_jsonable(source_meta),
        }

        with gzip.open(self._metadata_path(), "at", encoding="utf-8") as fp:
            fp.write(json.dumps(frame_record, ensure_ascii=False) + "\n")

    def close(self) -> Dict[str, Any]:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        return {
            "kind": "image_video_archive",
            "topic_name": self.topic_name,
            "topic_type": self.topic_type,
            "storage": {
                "video_file": self.video_filename,
                "video_codec": self.video_format_spec.codec,
                "container": self.video_format_spec.container,
                "pixel_format": "bgr8",
                "metadata_file": self.metadata_filename,
                "metadata_format": "jsonl.gz",
            },
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "first_bag_time_ns": self.first_bag_time_ns,
            "last_bag_time_ns": self.last_bag_time_ns,
            "source_message_type": self.source_message_type,
            "source_was_already_compressed": self.source_was_already_compressed,
            "restore_note": (
                f"可依据 {self.metadata_filename} 中的时间戳与 frame_id，逐帧解码 {self.video_filename}，"
                "恢复为 sensor_msgs/Image(bgr8) 数据流。"
            ),
            "quality_note": (
                "如果源消息本身是 sensor_msgs/CompressedImage（常见为 JPEG），"
                "则该视频不会再引入新的明显压缩损失，但无法恢复到 JPEG 压缩前的原始像素。"
            ),
        }


@dataclass
class GenericTopicArchive:
    topic_name: str
    topic_type: str
    topic_dir: Path
    messages_filename: str
    message_count: int = 0
    first_bag_time_ns: Optional[int] = None
    last_bag_time_ns: Optional[int] = None

    def _messages_path(self) -> Path:
        return self.topic_dir / self.messages_filename

    def append_message(self, message: Any, bag_time_ns: int) -> None:
        if self.first_bag_time_ns is None:
            self.first_bag_time_ns = int(bag_time_ns)
        self.last_bag_time_ns = int(bag_time_ns)
        self.message_count += 1

        header_time_ns = None
        header_frame_id = None
        if hasattr(message, "header"):
            header_time_ns = _header_stamp_to_ns(message.header)
            header_frame_id = str(getattr(message.header, "frame_id", ""))

        record = {
            "bag_time_ns": int(bag_time_ns),
            "header_time_ns": int(header_time_ns) if header_time_ns is not None else None,
            "header_frame_id": header_frame_id,
            "message": _to_builtin_jsonable(message_to_ordereddict(message)),
        }
        with gzip.open(self._messages_path(), "at", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> Dict[str, Any]:
        return {
            "kind": "generic_topic_archive",
            "topic_name": self.topic_name,
            "topic_type": self.topic_type,
            "storage": {
                "messages_file": self.messages_filename,
                "messages_format": "jsonl.gz",
            },
            "message_count": self.message_count,
            "first_bag_time_ns": self.first_bag_time_ns,
            "last_bag_time_ns": self.last_bag_time_ns,
        }


def export_bag_archive(
    input_path: Path,
    output_dir: Path,
    video_fps: float = 30.0,
    overwrite: bool = False,
    video_format: str = "mp4_x264_lossless",
) -> Dict[str, Any]:
    """将 rosbag/MCAP 导出为紧凑归档。"""
    input_bag_dir = _normalize_input_bag_path(input_path)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {output_dir}")
        shutil.rmtree(output_dir)

    if video_format not in VIDEO_FORMAT_SPECS:
        raise ValueError(f"不支持的视频格式: {video_format}")
    video_spec = VIDEO_FORMAT_SPECS[video_format]

    (output_dir / "topics").mkdir(parents=True, exist_ok=True)

    initial_scene_path = input_bag_dir / "initial_scene.json"
    if initial_scene_path.exists():
        shutil.copy2(initial_scene_path, output_dir / "initial_scene.json")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    type_cache = {topic: get_message(msg_type) for topic, msg_type in topic_types.items()}

    image_archives: Dict[str, ImageTopicArchive] = {}
    generic_archives: Dict[str, GenericTopicArchive] = {}

    manifest: Dict[str, Any] = {
        "source_bag_dir": str(input_bag_dir),
        "source_argument": str(input_path),
        "format_version": 1,
        "exporter": "mcap_export_archive.py",
        "video_policy": {
            "container": video_spec.container,
            "codec": video_spec.codec,
            "fps": float(video_fps),
            "lossless": bool(video_spec.lossless),
            "description": video_spec.description,
            "timestamp_policy": "exact_timestamps_saved_in_frames_jsonl_gz",
        },
        "topics": [],
        "stats": {
            "total_messages": 0,
            "image_messages": 0,
            "generic_messages": 0,
        },
        "sidecar_files": ["initial_scene.json"] if initial_scene_path.exists() else [],
    }

    while reader.has_next():
        topic_name, data, bag_time_ns = reader.read_next()
        manifest["stats"]["total_messages"] += 1

        topic_type = topic_types[topic_name]
        message_type = type_cache[topic_name]
        message = deserialize_message(data, message_type)

        topic_dir = output_dir / "topics" / _sanitize_topic_name(topic_name)
        topic_dir.mkdir(parents=True, exist_ok=True)

        if topic_type == "sensor_msgs/msg/Image":
            archive = image_archives.get(topic_name)
            if archive is None:
                archive = ImageTopicArchive(
                    topic_name=topic_name,
                    topic_type=topic_type,
                    topic_dir=topic_dir,
                    video_filename=f"video{video_spec.extension}",
                    metadata_filename="frames.jsonl.gz",
                    video_fps=video_fps,
                    video_format_spec=video_spec,
                    source_message_type=topic_type,
                    source_was_already_compressed=False,
                )
                image_archives[topic_name] = archive

            bgr, source_meta = _image_msg_to_bgr_and_meta(message)
            archive.append_frame(
                bgr=bgr,
                bag_time_ns=int(bag_time_ns),
                header_time_ns=_header_stamp_to_ns(message.header),
                header_frame_id=str(getattr(message.header, "frame_id", "")),
                source_meta=source_meta,
            )
            manifest["stats"]["image_messages"] += 1
            continue

        if topic_type == "sensor_msgs/msg/CompressedImage":
            archive = image_archives.get(topic_name)
            if archive is None:
                archive = ImageTopicArchive(
                    topic_name=topic_name,
                    topic_type=topic_type,
                    topic_dir=topic_dir,
                    video_filename=f"video{video_spec.extension}",
                    metadata_filename="frames.jsonl.gz",
                    video_fps=video_fps,
                    video_format_spec=video_spec,
                    source_message_type=topic_type,
                    source_was_already_compressed=True,
                )
                image_archives[topic_name] = archive

            bgr, source_meta = _compressed_image_msg_to_bgr_and_meta(message)
            archive.append_frame(
                bgr=bgr,
                bag_time_ns=int(bag_time_ns),
                header_time_ns=_header_stamp_to_ns(message.header),
                header_frame_id=str(getattr(message.header, "frame_id", "")),
                source_meta=source_meta,
            )
            manifest["stats"]["image_messages"] += 1
            continue

        archive = generic_archives.get(topic_name)
        if archive is None:
            archive = GenericTopicArchive(
                topic_name=topic_name,
                topic_type=topic_type,
                topic_dir=topic_dir,
                messages_filename="messages.jsonl.gz",
            )
            generic_archives[topic_name] = archive

        archive.append_message(message, int(bag_time_ns))
        manifest["stats"]["generic_messages"] += 1

        if manifest["stats"]["total_messages"] % 1000 == 0:
            print(
                f"已处理 {manifest['stats']['total_messages']} 条消息，"
                f"图像 {manifest['stats']['image_messages']} 条，"
                f"普通消息 {manifest['stats']['generic_messages']} 条...",
                flush=True,
            )

    for archive in image_archives.values():
        manifest["topics"].append(archive.close())
    for archive in generic_archives.values():
        manifest["topics"].append(archive.close())

    manifest["topics"].sort(key=lambda item: item["topic_name"])

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="将 MCAP/rosbag 导出为紧凑归档格式")
    parser.add_argument("input_path", type=Path, help="输入 rosbag 目录，或目录中的 .mcap 文件")
    parser.add_argument("output_dir", type=Path, help="输出归档目录")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="视频封装 FPS。精确时间戳不会丢失，而是单独写入 frames.jsonl.gz。",
    )
    parser.add_argument(
        "--video-format",
        choices=list(VIDEO_FORMAT_SPECS.keys()),
        default="mp4_x264_lossless",
        help=(
            "图像导出格式。"
            "mkv_ffv1=严格无损；"
            "mp4_x264_lossless=尽可能无损的 MP4。"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出目录")
    args = parser.parse_args()

    manifest = export_bag_archive(
        input_path=args.input_path,
        output_dir=args.output_dir,
        video_fps=args.video_fps,
        overwrite=args.overwrite,
        video_format=args.video_format,
    )

    print("=" * 70)
    print("✅ MCAP 后处理导出完成")
    print("=" * 70)
    print(f"输入: {manifest['source_bag_dir']}")
    print(f"输出: {args.output_dir}")
    print(f"视频容器: {manifest['video_policy']['container']}")
    print(f"视频编码: {manifest['video_policy']['codec']}")
    print(f"总消息数: {manifest['stats']['total_messages']}")
    print(f"图像消息数: {manifest['stats']['image_messages']}")
    print(f"普通消息数: {manifest['stats']['generic_messages']}")
    print(f"Topic 数量: {len(manifest['topics'])}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
