#!/usr/bin/env python3
"""Message-filter-like sync methods for frame and joint timelines."""

from __future__ import annotations

from bisect import bisect_left
from typing import Dict, List, Optional


def _nearest_index(sorted_time_ns: List[int], target_ns: int) -> int:
    i = bisect_left(sorted_time_ns, target_ns)
    if i == 0:
        return 0
    if i >= len(sorted_time_ns):
        return len(sorted_time_ns) - 1
    prev_i = i - 1
    if abs(sorted_time_ns[i] - target_ns) < abs(target_ns - sorted_time_ns[prev_i]):
        return i
    return prev_i


def sync_nearest(frame_times_ns: List[int], joint_times_ns: List[int]) -> List[Dict]:
    out: List[Dict] = []
    for frame_idx, frame_t in enumerate(frame_times_ns):
        j = _nearest_index(joint_times_ns, frame_t)
        joint_t = joint_times_ns[j]
        dt_ms = (joint_t - frame_t) / 1e6
        out.append(
            {
                "frame_index": frame_idx,
                "frame_time_ns": int(frame_t),
                "joint_index": j,
                "joint_time_ns": int(joint_t),
                "dt_ms": float(dt_ms),
                "valid": True,
            }
        )
    return out


def sync_approx(frame_times_ns: List[int], joint_times_ns: List[int], slop_ms: float) -> List[Dict]:
    slop_ns = int(slop_ms * 1e6)
    out: List[Dict] = []
    for frame_idx, frame_t in enumerate(frame_times_ns):
        j = _nearest_index(joint_times_ns, frame_t)
        joint_t = joint_times_ns[j]
        dt_ns = joint_t - frame_t
        valid = abs(dt_ns) <= slop_ns
        out.append(
            {
                "frame_index": frame_idx,
                "frame_time_ns": int(frame_t),
                "joint_index": j,
                "joint_time_ns": int(joint_t),
                "dt_ms": float(dt_ns / 1e6),
                "valid": bool(valid),
            }
        )
    return out


def build_quality_report(matches: List[Dict]) -> Dict:
    total = len(matches)
    if total == 0:
        return {
            "total_frames": 0,
            "matched_frames": 0,
            "drop_rate": 1.0,
            "mean_abs_dt_ms": None,
            "p95_abs_dt_ms": None,
            "max_abs_dt_ms": None,
        }

    valid = [m for m in matches if m["valid"]]
    matched = len(valid)
    drop_rate = 1.0 - (matched / total)

    if matched == 0:
        return {
            "total_frames": total,
            "matched_frames": 0,
            "drop_rate": drop_rate,
            "mean_abs_dt_ms": None,
            "p95_abs_dt_ms": None,
            "max_abs_dt_ms": None,
        }

    abs_dt = sorted(abs(m["dt_ms"]) for m in valid)
    mean_abs = sum(abs_dt) / matched
    p95_idx = min(len(abs_dt) - 1, int(0.95 * (len(abs_dt) - 1)))

    return {
        "total_frames": total,
        "matched_frames": matched,
        "drop_rate": float(drop_rate),
        "mean_abs_dt_ms": float(mean_abs),
        "p95_abs_dt_ms": float(abs_dt[p95_idx]),
        "max_abs_dt_ms": float(abs_dt[-1]),
    }
