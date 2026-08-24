#!/usr/bin/python3
"""
Replay a recorded teaching archive inside MuJoCo.

The recorder exports archives under:
    rosbag_archive/teaching_YYYYMMDD_HHMMSS/

This script reads the archived JointState JSONL data and drives the MuJoCo
robot with the same joint commands/timing, so a recorded teaching motion can be
played back without starting the full ROS2 recording stack.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PYTHON = "/usr/bin/python3"


def _enforce_system_python() -> None:
    """Keep this script consistent with the rest of the ROS/MuJoCo project."""
    current_python = os.path.realpath(sys.executable)
    expected_python = os.path.realpath(SYSTEM_PYTHON)
    if current_python != expected_python:
        raise SystemExit(
            "\n❌ replay.py 必须使用系统 Python 运行。\n"
            f"   当前: {current_python}\n"
            f"   要求: {expected_python}\n"
            f"   请改用: {SYSTEM_PYTHON} mujoco_env/main/replay.py\n"
        )


_enforce_system_python()

if not os.environ.get("ROS_LOG_DIR"):
    ros_log_dir = Path(tempfile.gettempdir()) / "arm_project_ros_logs"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = str(ros_log_dir)

sys.path.append(str(Path(__file__).resolve().parents[1] / "envs"))
from env import ArmEnv  # noqa: E402


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
DEFAULT_MODEL = PROJECT_ROOT / "mujoco_env/robot_model/exp/env_robot_torque_tactile.xml"
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "rosbag_archive"
DEFAULT_BAG_ROOT = PROJECT_ROOT / "rosbag_data"


@dataclass(frozen=True)
class ReplaySample:
    """One timestamped joint command/state from the archived JSONL topic."""

    time_ns: int
    arm_joints: np.ndarray
    gripper: Optional[float]


def _topic_dir_name(topic_name: str) -> str:
    return topic_name.strip("/").replace("/", "__") or "root_topic"


def _iter_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_latest_archive(archive_root: Path) -> Path:
    """Return the newest teaching archive containing a manifest.json."""
    candidates = [
        path
        for path in archive_root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"没有找到可回放的归档目录: {archive_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_archive(path: Optional[Path], archive_root: Path) -> Path:
    if path is None:
        return find_latest_archive(archive_root)
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"归档路径不存在: {path}")
    if path.is_file() and path.suffix == ".zip":
        raise ValueError("当前 replay.py 读取已解压归档目录，请传入 teaching_xxx 目录而不是 .zip")
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"归档目录缺少 manifest.json: {path}")
    return path


def choose_joint_topic(archive_dir: Path, source: str) -> str:
    """Choose the archived topic to replay."""
    topics_dir = archive_dir / "topics"
    available = {
        topic
        for topic in ("joint_cmd", "joint_states", "joint_states_R")
        if (topics_dir / topic / "messages.jsonl.gz").exists()
    }

    if source == "auto":
        # Prefer commands: they are what the user sent during teaching and they
        # include the gripper command. Fall back to states if commands are absent.
        for candidate in ("joint_cmd", "joint_states", "joint_states_R"):
            if candidate in available:
                return candidate
    elif source in available:
        return source

    raise FileNotFoundError(
        f"归档中找不到可用 joint topic。请求={source}, 可用={sorted(available)}"
    )


def load_replay_samples(archive_dir: Path, source: str) -> tuple[str, List[ReplaySample]]:
    topic = choose_joint_topic(archive_dir, source)
    topic_path = archive_dir / "topics" / _topic_dir_name(f"/{topic}") / "messages.jsonl.gz"
    samples: List[ReplaySample] = []

    for row in _iter_jsonl_gz(topic_path):
        msg = row.get("message", {})
        names = list(msg.get("name", []))
        positions = list(msg.get("position", []))
        if not names or not positions:
            continue

        name_to_pos = {name: float(pos) for name, pos in zip(names, positions)}
        if not all(name in name_to_pos for name in ARM_JOINT_NAMES):
            continue

        joints = np.array([name_to_pos[name] for name in ARM_JOINT_NAMES], dtype=np.float64)

        # /joint_cmd is the external ROS command topic. main.py maps J3/J4/J6 by
        # sign before applying it to MuJoCo, so replay must do the same mapping.
        if topic == "joint_cmd":
            joints[2] *= -1.0
            joints[3] *= -1.0
            joints[5] *= -1.0

        time_ns = int(row.get("header_time_ns") or row.get("bag_time_ns"))
        gripper = name_to_pos.get("gripper")
        samples.append(ReplaySample(time_ns=time_ns, arm_joints=joints, gripper=gripper))

    samples.sort(key=lambda sample: sample.time_ns)
    if not samples:
        raise RuntimeError(f"没有从 {topic_path} 读取到有效 JointState 数据")
    return topic, samples


def load_scene_state(archive_dir: Path) -> Optional[dict]:
    """Load the recorded initial scene state sidecar, if this archive contains one."""
    scene_path = archive_dir / "initial_scene.json"
    if not scene_path.exists():
        return None

    with open(scene_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    scene_state = payload.get("scene_state")
    return scene_state if isinstance(scene_state, dict) else None


def _set_arm_qpos_direct(env: ArmEnv, joints: np.ndarray) -> None:
    """Set the robot arm pose exactly, useful for the first replay frame."""
    for name, value in zip(ARM_JOINT_NAMES, joints):
        env.robot.data.joint(name).qpos[0] = float(value)
    env.robot.joint_targets = joints.copy()
    env.robot.update_joint_state()
    mujoco.mj_forward(env.robot.model, env.robot.data)


def _apply_gripper_command(env: ArmEnv, gripper: Optional[float]) -> None:
    if gripper is None:
        return
    # Recorded commands use 0.0~1.0. Robot.set_gripper_command expects 0~255.
    if 0.0 <= gripper <= 1.0:
        env.robot.set_gripper_command(float(gripper) * 255.0)
    else:
        env.robot.set_gripper_command(float(gripper))


def replay_archive(
    archive_dir: Path,
    model_path: Path,
    source: str,
    speed: float,
    headless: bool,
    realtime: bool,
    settle_seconds: float,
    render_fps: float,
) -> None:
    topic, samples = load_replay_samples(archive_dir, source)
    scene_state = load_scene_state(archive_dir)
    duration_s = max(0.0, (samples[-1].time_ns - samples[0].time_ns) / 1e9)

    print("=" * 72)
    print("MuJoCo teaching replay")
    print("=" * 72)
    print(f"归档目录: {archive_dir}")
    print(f"回放 topic: /{topic}")
    print(f"样本数量: {len(samples)}")
    print(f"录制时长: {duration_s:.3f}s")
    print(f"模型文件: {model_path}")
    print(f"速度倍率: {speed:.2f}x")
    print(f"渲染帧率: {render_fps:.1f} FPS")
    print(f"场景恢复: {'已找到 initial_scene.json' if scene_state else '未找到，使用 reset 随机场景'}")
    print("=" * 72)

    env = ArmEnv(
        model_path=str(model_path),
        render_mode=None if headless else "human",
        enable_visualization=not headless,
        enable_depth_render=False,
        enable_robot_cameras=False,
    )

    try:
        # replay.py does not run the camera publishing pipeline; disable the
        # generic camera heartbeat warning so it does not look like a deadlock.
        if hasattr(env.robot, "_heartbeat_enabled"):
            env.robot._heartbeat_enabled = False

        env.reset()
        if scene_state is not None:
            env.apply_scene_state(scene_state)
            print("已恢复录制时场景状态")
        _set_arm_qpos_direct(env, samples[0].arm_joints)
        _apply_gripper_command(env, samples[0].gripper)

        wall_start = time.perf_counter()
        render_period = 1.0 / max(float(render_fps), 1e-6)
        next_render_wall = wall_start
        last_time_ns = samples[0].time_ns

        for index, sample in enumerate(samples):
            if index > 0:
                dt_s = max(0.0, (sample.time_ns - last_time_ns) / 1e9)
                sim_dt = float(env.robot.dt)
                step_count = max(1, int(round(dt_s / max(sim_dt, 1e-6))))
            else:
                step_count = 1

            env.robot.update_joint_state()
            env.robot.apply_joint_control(sample.arm_joints)
            _apply_gripper_command(env, sample.gripper)

            for _ in range(step_count):
                env.robot.step()

                now = time.perf_counter()
                if not headless and now >= next_render_wall:
                    env.render()
                    next_render_wall = now + render_period

            last_time_ns = sample.time_ns

            if realtime and speed > 0.0:
                target_wall = (sample.time_ns - samples[0].time_ns) / 1e9 / speed
                remaining = wall_start + target_wall - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

            if index % 500 == 0 or index == len(samples) - 1:
                progress = 100.0 * (index + 1) / len(samples)
                print(f"\r回放进度: {progress:6.2f}% ({index + 1}/{len(samples)})", end="", flush=True)

        print("\n回放完成")

        if settle_seconds > 0.0:
            settle_steps = int(settle_seconds / max(float(env.robot.dt), 1e-6))
            current = samples[-1].arm_joints
            for _ in range(settle_steps):
                env.robot.update_joint_state()
                env.robot.apply_joint_control(current)
                _apply_gripper_command(env, samples[-1].gripper)
                env.robot.step()
                if not headless:
                    env.render()
            print(f"已额外保持末帧 {settle_seconds:.2f}s")

    finally:
        env.close()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_rmtree(path: Path, root: Path) -> bool:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if not path.exists():
        return False
    if not path.is_dir():
        raise ValueError(f"拒绝删除非目录路径: {path}")
    if path == root or not _is_relative_to(path, root):
        raise ValueError(f"拒绝删除根目录外路径: {path} (root={root})")
    shutil.rmtree(path)
    return True


def delete_recording_pair(archive_dir: Path, archive_root: Path, bag_root: Path) -> list[Path]:
    """Delete the reviewed archive and the same-named raw rosbag directory."""
    archive_dir = archive_dir.expanduser().resolve()
    archive_root = archive_root.expanduser().resolve()
    bag_root = bag_root.expanduser().resolve()

    if not archive_dir.name.startswith("teaching_"):
        raise ValueError(f"拒绝删除非 teaching_* 目录: {archive_dir}")

    deleted: list[Path] = []
    if _safe_rmtree(archive_dir, archive_root):
        deleted.append(archive_dir)

    raw_bag_dir = bag_root / archive_dir.name
    if _safe_rmtree(raw_bag_dir, bag_root):
        deleted.append(raw_bag_dir)

    return deleted


def prompt_review_decision(archive_dir: Path) -> str:
    print("\n" + "=" * 72)
    print("数据审核")
    print("=" * 72)
    print(f"当前数据: {archive_dir}")
    print("按 Enter/k 保留；按 d 删除；按 r 再回放一次；按 q 退出并保留。")
    while True:
        decision = input("审核结果 [keep/delete/replay/quit]: ").strip().lower()
        if decision in ("", "k", "keep", "y", "yes", "pass", "p"):
            return "keep"
        if decision in ("d", "delete", "del", "n", "no", "fail", "f"):
            return "delete"
        if decision in ("r", "replay", "again"):
            return "replay"
        if decision in ("q", "quit", "exit"):
            return "quit"
        print("请输入 k/Enter 保留，d 删除，r 重放，q 退出。")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 MuJoCo 中回放 rosbag_archive 里的 teaching 动作")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="归档根目录，默认 PROJECT_ROOT/rosbag_archive",
    )
    parser.add_argument(
        "--bag-root",
        type=Path,
        default=DEFAULT_BAG_ROOT,
        help="原始 rosbag 根目录；审核删除时会删除同名 teaching_* 目录",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="指定某个 teaching_xxx 归档目录；不传则自动使用最新归档",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo XML 模型路径")
    parser.add_argument(
        "--source",
        choices=["auto", "joint_cmd", "joint_states", "joint_states_R"],
        default="auto",
        help="回放数据源。auto 优先使用 joint_cmd，缺失时使用 joint_states",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率，例如 2.0 表示两倍速")
    parser.add_argument("--headless", action="store_true", help="无 GUI 回放")
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="不按真实墙钟等待，只按记录时间间隔推进仿真",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="回放结束后保持最后一个动作的时间",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=30.0,
        help="GUI 回放渲染帧率；默认 30 FPS，避免高频 joint sample 拖慢回放",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="回放结束后进入审核：k/Enter 保留，d 删除，r 重放",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    archive_dir = resolve_archive(args.archive, args.archive_root)
    while True:
        replay_archive(
            archive_dir=archive_dir,
            model_path=args.model.expanduser().resolve(),
            source=args.source,
            speed=max(float(args.speed), 1e-6),
            headless=bool(args.headless),
            realtime=not args.no_realtime,
            settle_seconds=max(0.0, float(args.settle_seconds)),
            render_fps=max(float(args.render_fps), 1.0),
        )

        if not args.review:
            return

        decision = prompt_review_decision(archive_dir)
        if decision == "replay":
            continue
        if decision == "delete":
            deleted = delete_recording_pair(
                archive_dir=archive_dir,
                archive_root=args.archive_root,
                bag_root=args.bag_root,
            )
            if deleted:
                print("已删除不合格数据:")
                for path in deleted:
                    print(f"  - {path}")
            else:
                print("没有找到可删除的数据目录")
            return
        if decision == "keep":
            print(f"已保留合格数据: {archive_dir}")
            return
        if decision == "quit":
            print(f"退出审核，数据已保留: {archive_dir}")
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断回放")
