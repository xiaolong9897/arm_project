#!/usr/bin/env python3
"""Pipeline entry: load archive -> sync -> aligned session."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run alignment pipeline")
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--camera-topic", default="/ee_camera/rgb/image_raw")
    parser.add_argument("--sync-mode", choices=["nearest", "approx"], default="approx")
    parser.add_argument("--slop-ms", type=float, default=33.0)
    parser.add_argument(
        "--joint-topic-candidates",
        nargs="+",
        default=["/joint_states", "/joint_states_sim"],
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve().parent / "build_aligned_session.py"
    cmd = [
        sys.executable,
        str(script_path),
        str(args.archive_dir),
        str(args.output_dir),
        "--camera-topic",
        args.camera_topic,
        "--sync-mode",
        args.sync_mode,
        "--slop-ms",
        str(args.slop_ms),
        "--joint-topic-candidates",
        *args.joint_topic_candidates,
    ]

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
