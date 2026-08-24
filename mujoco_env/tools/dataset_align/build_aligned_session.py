#!/usr/bin/env python3
"""Build aligned session from exported archive (manifest + frames/messages)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from load_archive import (
    ArchiveLoadError,
    extract_joint_position,
    find_first_topic_entry,
    load_manifest,
    pick_time_ns,
    read_jsonl_gz,
    topic_file_path,
)
from sync_filters import build_quality_report, sync_approx, sync_nearest


def _load_frame_records(archive_dir: Path, topic_name: str, topic_entry: Dict) -> List[Dict]:
    meta_file = topic_entry["storage"]["metadata_file"]
    path = topic_file_path(archive_dir, topic_name, meta_file)
    return read_jsonl_gz(path)


def _load_joint_records(archive_dir: Path, topic_name: str, topic_entry: Dict) -> List[Dict]:
    msg_file = topic_entry["storage"]["messages_file"]
    path = topic_file_path(archive_dir, topic_name, msg_file)
    return read_jsonl_gz(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build aligned session from exported archive")
    parser.add_argument("archive_dir", type=Path, help="Archive directory with manifest.json")
    parser.add_argument("output_dir", type=Path, help="Output aligned session directory")
    parser.add_argument("--camera-topic", default="/ee_camera/rgb/image_raw", help="Camera topic name")
    parser.add_argument(
        "--joint-topic-candidates",
        nargs="+",
        default=["/joint_states", "/joint_states_sim"],
        help="Candidate joint topics in priority order",
    )
    parser.add_argument("--sync-mode", choices=["nearest", "approx"], default="approx")
    parser.add_argument("--slop-ms", type=float, default=33.0, help="Approx sync window in milliseconds")
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = load_manifest(archive_dir)

        cam_topic, cam_entry = find_first_topic_entry(manifest, [args.camera_topic])
        joint_topic, joint_entry = find_first_topic_entry(manifest, args.joint_topic_candidates)

        frame_records = _load_frame_records(archive_dir, cam_topic, cam_entry)
        joint_records = _load_joint_records(archive_dir, joint_topic, joint_entry)

        frame_times_ns = [pick_time_ns(r) for r in frame_records]
        joint_times_ns = [pick_time_ns(r) for r in joint_records]

        if args.sync_mode == "nearest":
            matches = sync_nearest(frame_times_ns, joint_times_ns)
        else:
            matches = sync_approx(frame_times_ns, joint_times_ns, slop_ms=args.slop_ms)

        aligned_rows: List[Dict] = []
        for m in matches:
            frame_record = frame_records[m["frame_index"]]
            joint_record = joint_records[m["joint_index"]]
            aligned_rows.append(
                {
                    "frame_index": int(m["frame_index"]),
                    "frame_time_ns": int(m["frame_time_ns"]),
                    "joint_index": int(m["joint_index"]),
                    "joint_time_ns": int(m["joint_time_ns"]),
                    "dt_ms": float(m["dt_ms"]),
                    "valid": bool(m["valid"]),
                    "joint_position": extract_joint_position(joint_record),
                    "camera_topic": cam_topic,
                    "joint_topic": joint_topic,
                    "video_file": cam_entry["storage"]["video_file"],
                    "video_frame_index": int(frame_record.get("frame_index", m["frame_index"])),
                }
            )

        report = build_quality_report(matches)
        report["sync_mode"] = args.sync_mode
        report["slop_ms"] = float(args.slop_ms)
        report["camera_topic"] = cam_topic
        report["joint_topic"] = joint_topic

        with open(output_dir / "aligned_frames.jsonl", "w", encoding="utf-8") as f:
            for row in aligned_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        session_meta = {
            "source_archive": str(archive_dir),
            "manifest_source_bag": manifest.get("source_bag_dir"),
            "camera_topic": cam_topic,
            "joint_topic": joint_topic,
            "video_topic_dir": cam_topic.strip("/").replace("/", "__") or "root_topic",
            "video_file": cam_entry["storage"]["video_file"],
            "aligned_count": len(aligned_rows),
            "quality": report,
        }
        with open(output_dir / "session_meta.json", "w", encoding="utf-8") as f:
            json.dump(session_meta, f, ensure_ascii=False, indent=2)

        with open(output_dir / "sync_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("=" * 70)
        print("Aligned session generated")
        print("=" * 70)
        print(f"Archive: {archive_dir}")
        print(f"Output:  {output_dir}")
        print(f"Camera:  {cam_topic}")
        print(f"Joint:   {joint_topic}")
        print(f"Total frames:   {report['total_frames']}")
        print(f"Matched frames: {report['matched_frames']}")
        print(f"Drop rate:      {report['drop_rate']:.4f}")
        print(f"Mean |dt| ms:   {report['mean_abs_dt_ms']}")
        print(f"P95  |dt| ms:   {report['p95_abs_dt_ms']}")
        print("=" * 70)
        return 0

    except ArchiveLoadError as e:
        print(f"Archive load error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
