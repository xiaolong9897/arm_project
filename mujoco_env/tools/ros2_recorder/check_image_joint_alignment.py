#!/usr/bin/python3
"""Direct alignment check between image topic and joint topic from MCAP/rosbag."""

from __future__ import annotations

import argparse
import os
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

SYSTEM_PYTHON = "/usr/bin/python3"


def _enforce_system_python() -> None:
    if os.path.realpath(sys.executable) != os.path.realpath(SYSTEM_PYTHON):
        raise SystemExit(
            f"Use system python: {SYSTEM_PYTHON} mujoco_env/tools/ros2_recorder/check_image_joint_alignment.py"
        )


def _normalize_bag_dir(input_path: Path) -> Path:
    if input_path.is_dir():
        if not (input_path / "metadata.yaml").exists():
            raise FileNotFoundError(f"metadata.yaml not found: {input_path}")
        return input_path
    if input_path.is_file() and input_path.suffix.lower() == ".mcap":
        bag_dir = input_path.parent
        if not (bag_dir / "metadata.yaml").exists():
            raise FileNotFoundError(f"metadata.yaml not found: {bag_dir}")
        return bag_dir
    raise FileNotFoundError(f"unsupported input: {input_path}")


def _header_time_ns(msg: Any) -> Optional[int]:
    if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
        return None
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _choose_ts_ns(msg: Any, bag_time_ns: int) -> int:
    h = _header_time_ns(msg)
    return int(h) if h is not None else int(bag_time_ns)


def _nearest_index(sorted_ns: List[int], target: int) -> int:
    i = bisect_left(sorted_ns, target)
    if i <= 0:
        return 0
    if i >= len(sorted_ns):
        return len(sorted_ns) - 1
    prev_i = i - 1
    if abs(sorted_ns[i] - target) < abs(target - sorted_ns[prev_i]):
        return i
    return prev_i


def _quality(frame_ns: List[int], joint_ns: List[int], slop_ms: float) -> Dict[str, float]:
    slop_ns = int(slop_ms * 1e6)
    dts = []
    valid_dts = []
    for t in frame_ns:
        j = _nearest_index(joint_ns, t)
        dt_ns = joint_ns[j] - t
        dt_ms = dt_ns / 1e6
        dts.append(abs(dt_ms))
        if abs(dt_ns) <= slop_ns:
            valid_dts.append(abs(dt_ms))

    total = len(frame_ns)
    matched = len(valid_dts)
    drop_rate = 1.0 - (matched / total) if total > 0 else 1.0

    dts_sorted = sorted(dts)
    valid_sorted = sorted(valid_dts)

    p95_all = dts_sorted[min(len(dts_sorted) - 1, int(0.95 * (len(dts_sorted) - 1)))] if dts_sorted else None
    p95_valid = valid_sorted[min(len(valid_sorted) - 1, int(0.95 * (len(valid_sorted) - 1)))] if valid_sorted else None

    return {
        "total_frames": total,
        "matched_frames": matched,
        "drop_rate": drop_rate,
        "mean_abs_dt_ms_all": (sum(dts) / len(dts)) if dts else None,
        "p95_abs_dt_ms_all": p95_all,
        "max_abs_dt_ms_all": (dts_sorted[-1] if dts_sorted else None),
        "mean_abs_dt_ms_valid": (sum(valid_dts) / len(valid_dts)) if valid_dts else None,
        "p95_abs_dt_ms_valid": p95_valid,
    }


def main() -> int:
    _enforce_system_python()

    parser = argparse.ArgumentParser(description="Check image/joint timestamp alignment from MCAP")
    parser.add_argument("input_path", type=Path, help="bag dir or .mcap")
    parser.add_argument("--image-topic", default="/ee_camera/rgb/image_raw")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--joint-topic-fallback", default="/joint_states_sim")
    parser.add_argument("--slop-ms", type=float, default=33.0)
    parser.add_argument("--pass-p95-ms", type=float, default=25.0)
    parser.add_argument("--pass-drop-rate", type=float, default=0.05)
    args = parser.parse_args()

    bag_dir = _normalize_bag_dir(args.input_path.resolve())

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.image_topic not in topic_types:
        raise SystemExit(f"image topic not found: {args.image_topic}")

    joint_topic = args.joint_topic
    if joint_topic not in topic_types:
        if args.joint_topic_fallback in topic_types:
            joint_topic = args.joint_topic_fallback
        else:
            raise SystemExit(f"joint topic not found: {args.joint_topic} or {args.joint_topic_fallback}")

    needed = {args.image_topic, joint_topic}
    type_cache = {name: get_message(topic_types[name]) for name in needed}

    image_ts: List[int] = []
    joint_ts: List[int] = []

    while reader.has_next():
        topic_name, data, bag_time_ns = reader.read_next()
        if topic_name not in needed:
            continue
        msg = deserialize_message(data, type_cache[topic_name])
        ts = _choose_ts_ns(msg, int(bag_time_ns))
        if topic_name == args.image_topic:
            image_ts.append(ts)
        else:
            joint_ts.append(ts)

    if not image_ts:
        raise SystemExit("no image messages found")
    if not joint_ts:
        raise SystemExit("no joint messages found")

    image_ts.sort()
    joint_ts.sort()

    r = _quality(image_ts, joint_ts, slop_ms=args.slop_ms)
    p95 = r["p95_abs_dt_ms_valid"] if r["p95_abs_dt_ms_valid"] is not None else 1e9
    ok = (r["drop_rate"] <= args.pass_drop_rate) and (p95 <= args.pass_p95_ms)

    print("=" * 70)
    print("Image/Joint Alignment Check")
    print("=" * 70)
    print(f"bag_dir:     {bag_dir}")
    print(f"image_topic: {args.image_topic}")
    print(f"joint_topic: {joint_topic}")
    print(f"slop_ms:     {args.slop_ms}")
    print("-")
    print(f"total_frames:            {r['total_frames']}")
    print(f"matched_frames:          {r['matched_frames']}")
    print(f"drop_rate:               {r['drop_rate']:.4f}")
    print(f"mean|dt| all (ms):       {r['mean_abs_dt_ms_all']}")
    print(f"p95 |dt| all (ms):       {r['p95_abs_dt_ms_all']}")
    print(f"max |dt| all (ms):       {r['max_abs_dt_ms_all']}")
    print(f"mean|dt| valid (ms):     {r['mean_abs_dt_ms_valid']}")
    print(f"p95 |dt| valid (ms):     {r['p95_abs_dt_ms_valid']}")
    print("-")
    print(f"PASS: {ok}  (rule: drop_rate<={args.pass_drop_rate}, p95_valid<={args.pass_p95_ms}ms)")
    print("=" * 70)

    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
