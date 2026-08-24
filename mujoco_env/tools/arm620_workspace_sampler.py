#!/usr/bin/env python3
"""
Sample the ARM620 reachable workspace with forward kinematics.

This is a Monte Carlo approximation: it samples joint angles within limits,
computes end-effector xyz in base_link, and saves the reachable point cloud.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


JOINT_LIMITS = np.array(
    [
        [-2.9670, 2.9670],
        [-1.5708, 1.5708],
        [-1.5708, 1.5708],
        [-2.9670, 2.9670],
        [-1.5708, 1.5708],
        [-2.9670, 2.9670],
    ],
    dtype=np.float64,
)


FIXED_JOINT_XYZ_RPY = [
    ([0.0, 0.0, 0.084], [0.0, 0.0, 0.0]),
    ([0.0, 0.0, 0.068718], [1.5708, 0.0, -1.5708]),
    ([0.0, 0.30025, 0.0], [-3.1416, 0.0, 0.0]),
    ([0.0, -0.15558, 3.5e-05], [1.5708, -1.5708, 0.0]),
    ([-3.5e-05, 1.5362e-05, 0.064223], [1.5708, 0.0, 0.0]),
    ([-0.00047523, 0.095552, 0.0], [-1.5709, 0.0, 0.0]),
]


JOINT_AXES_LOCAL = [
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
]


def rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def trans(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def rpy_to_transform(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def axis_angle_transform(axis: list[float], theta: float) -> np.ndarray:
    if axis[0] == 1.0:
        return rot_x(theta)
    if axis[0] == -1.0:
        return rot_x(-theta)
    if axis[1] == 1.0:
        return rot_y(theta)
    if axis[1] == -1.0:
        return rot_y(-theta)
    if axis[2] == 1.0:
        return rot_z(theta)
    if axis[2] == -1.0:
        return rot_z(-theta)
    return np.eye(4, dtype=np.float64)


def forward_kinematics(q: np.ndarray, tool_offset: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    for i in range(6):
        xyz, rpy = FIXED_JOINT_XYZ_RPY[i]
        transform = transform @ trans(*xyz) @ rpy_to_transform(*rpy)
        transform = transform @ axis_angle_transform(JOINT_AXES_LOCAL[i], float(q[i]))
    transform = transform @ trans(0.0, 0.0, tool_offset)
    return transform


def sample_workspace(samples: int, seed: int, tool_offset: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(samples, 6))
    points = np.empty((samples, 3), dtype=np.float64)

    for i in range(samples):
        points[i] = forward_kinematics(q[i], tool_offset)[:3, 3]

    return q, points


def print_summary(points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius_xy = np.linalg.norm(points[:, :2], axis=1)
    radius_xyz = np.linalg.norm(points, axis=1)

    print("workspace in base_link frame, unit: meter")
    print(f"x range: [{mins[0]:+.4f}, {maxs[0]:+.4f}], width={maxs[0] - mins[0]:.4f}")
    print(f"y range: [{mins[1]:+.4f}, {maxs[1]:+.4f}], width={maxs[1] - mins[1]:.4f}")
    print(f"z range: [{mins[2]:+.4f}, {maxs[2]:+.4f}], height={maxs[2] - mins[2]:.4f}")
    print(f"box center: [{center[0]:+.4f}, {center[1]:+.4f}, {center[2]:+.4f}]")
    print(f"xy radius range: [{radius_xy.min():.4f}, {radius_xy.max():.4f}]")
    print(f"xyz distance range: [{radius_xyz.min():.4f}, {radius_xyz.max():.4f}]")


def save_csv(path: Path, q: np.ndarray, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([q, points])
    header = "joint1,joint2,joint3,joint4,joint5,joint6,x,y,z"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_plot(path: Path, points: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip plot")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    stride = max(1, len(points) // 30000)
    pts = points[::stride]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.25)
    ax.set_xlabel("x base_link (m)")
    ax.set_ylabel("y base_link (m)")
    ax.set_zlabel("z base_link (m)")
    ax.set_title("ARM620 sampled reachable workspace")
    ax.set_box_aspect(
        [
            points[:, 0].ptp(),
            points[:, 1].ptp(),
            max(points[:, 2].ptp(), 1e-6),
        ]
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample ARM620 reachable workspace.")
    parser.add_argument("--samples", type=int, default=100000, help="Random samples. Default: 100000")
    parser.add_argument("--seed", type=int, default=1, help="Random seed. Default: 1")
    parser.add_argument("--tool-offset", type=float, default=0.13, help="Tool offset along local Z. Default: 0.13")
    parser.add_argument(
        "--output",
        default="mujoco_env/main/xiaoman1/arm620_workspace_points.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        default="",
        help="Optional png path for a 3D scatter plot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    q, points = sample_workspace(args.samples, args.seed, args.tool_offset)
    print_summary(points)
    save_csv(Path(args.output), q, points)
    print(f"saved csv: {args.output}")
    if args.plot:
        save_plot(Path(args.plot), points)
        print(f"saved plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
