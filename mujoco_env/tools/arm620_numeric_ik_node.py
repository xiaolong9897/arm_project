#!/usr/bin/env python3
"""
ARM620 数值 IK ROS2 节点。

订阅:
  - /ee_target: geometry_msgs/PoseStamped，目标末端姿态，frame_id 期望为 base_link
  - /joint_states_sim 或 /joint_states: sensor_msgs/JointState，当前关节状态

发布:
  - /joint_target: sensor_msgs/JointState，6 轴关节目标
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
JOINT_LIMITS = np.array(
    [
        [-2.967, 2.967],
        [-1.5708, 1.5708],
        [-1.5708, 1.5708],
        [-2.967, 2.967],
        [-1.5708, 1.5708],
        [-2.967, 2.967],
    ],
    dtype=np.float64,
)


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat_wxyz))
    if norm < 1e-10:
        return np.eye(3)
    q = quat_wxyz / norm
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_to_matrix(quat_wxyz)
    transform[:3, 3] = pos
    return transform


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    return R.from_rotvec(axis * angle).as_matrix()


def _make_transform(pos, quat) -> np.ndarray:
    return _pose_matrix(np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))


@dataclass
class FKResult:
    transform: np.ndarray
    joint_origins: list[np.ndarray]
    joint_axes_world: list[np.ndarray]


class Arm620Kinematics:
    """
    按当前 MJCF 运动链实现 FK/Jacobian。

    末端位姿链:
      base -> Link1 -> ... -> Link6 -> robotiq_2f85_v4_base -> tools_link
    Jacobian 公式:
      Jv_i = a_i x (p_ee - p_i)
      Jw_i = a_i
    其中 a_i 是第 i 个转轴在 base_link 坐标系下的方向，p_i 是该关节原点。
    """

    def __init__(self) -> None:
        self.fixed_after_joint = [
            _make_transform([0, 0, 0.084], [1, 0, 0, 0]),
            _make_transform([0, 0, 0.068718], [0.499998, 0.5, -0.500002, -0.5]),
            _make_transform([0, 0.30025, 0], [-3.67321e-06, -1, 0, 0]),
            _make_transform([0, -0.15558, 3.5e-05], [0.499998, 0.5, -0.5, 0.500002]),
            _make_transform([-3.5e-05, 1.5362e-05, 0.064223], [0.707105, 0.707108, 0, 0]),
            _make_transform([-0.00047523, 0.095552, 0], [0.70707, -0.707143, 0, 0]),
        ]
        self.joint_axes_local = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        self.tool_offset = _make_transform([0, 0, 0.13], [1, 0, 0, 0])

    def forward(self, q: np.ndarray) -> FKResult:
        transform = np.eye(4, dtype=np.float64)
        joint_origins: list[np.ndarray] = []
        joint_axes_world: list[np.ndarray] = []

        for i in range(6):
            transform = transform @ self.fixed_after_joint[i]
            joint_origins.append(transform[:3, 3].copy())
            axis_world = transform[:3, :3] @ self.joint_axes_local[i]
            axis_world = axis_world / max(np.linalg.norm(axis_world), 1e-12)
            joint_axes_world.append(axis_world)

            joint_rot = np.eye(4, dtype=np.float64)
            joint_rot[:3, :3] = _axis_angle_rotation(self.joint_axes_local[i], float(q[i]))
            transform = transform @ joint_rot

        transform = transform @ self.tool_offset
        return FKResult(transform=transform, joint_origins=joint_origins, joint_axes_world=joint_axes_world)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        fk = self.forward(q)
        p_ee = fk.transform[:3, 3]
        jac = np.zeros((6, 6), dtype=np.float64)
        for i, (origin, axis) in enumerate(zip(fk.joint_origins, fk.joint_axes_world)):
            jac[:3, i] = np.cross(axis, p_ee - origin)
            jac[3:, i] = axis
        return jac


class NumericIKSolver:
    def __init__(
        self,
        max_iters: int = 80,
        damping: float = 0.04,
        step_scale: float = 0.7,
        position_tolerance: float = 0.003,
        orientation_tolerance: float = 0.04,
    ) -> None:
        self.kin = Arm620Kinematics()
        self.max_iters = int(max_iters)
        self.damping = float(damping)
        self.step_scale = float(step_scale)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)

    def solve(self, target_transform: np.ndarray, seed: np.ndarray) -> tuple[bool, np.ndarray, float, float]:
        q = np.clip(seed.astype(np.float64).copy(), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        weights = np.diag([1.0, 1.0, 1.0, 0.35, 0.35, 0.35])

        for _ in range(self.max_iters):
            fk = self.kin.forward(q)
            current = fk.transform
            pos_error = target_transform[:3, 3] - current[:3, 3]
            # 姿态误差使用 R_err = R_target * R_current^T，再取旋转向量。
            rot_error = R.from_matrix(target_transform[:3, :3] @ current[:3, :3].T).as_rotvec()

            pos_norm = float(np.linalg.norm(pos_error))
            ori_norm = float(np.linalg.norm(rot_error))
            if pos_norm <= self.position_tolerance and ori_norm <= self.orientation_tolerance:
                return True, q, pos_norm, ori_norm

            error = weights @ np.concatenate([pos_error, rot_error])
            jac = weights @ self.kin.jacobian(q)
            lhs = jac @ jac.T + (self.damping ** 2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(lhs, error)
            max_abs = float(np.max(np.abs(dq)))
            if max_abs > 0.12:
                dq *= 0.12 / max_abs
            q = np.clip(q + self.step_scale * dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        fk = self.kin.forward(q)
        pos_norm = float(np.linalg.norm(target_transform[:3, 3] - fk.transform[:3, 3]))
        ori_norm = float(np.linalg.norm(R.from_matrix(target_transform[:3, :3] @ fk.transform[:3, :3].T).as_rotvec()))
        return False, q, pos_norm, ori_norm


class Arm620NumericIKNode(Node):
    def __init__(self, args) -> None:
        super().__init__("arm620_numeric_ik_node")
        self.args = args
        self.solver = NumericIKSolver(
            max_iters=args.max_iters,
            damping=args.damping,
            step_scale=args.step_scale,
            position_tolerance=args.position_tolerance,
            orientation_tolerance=args.orientation_tolerance,
        )
        self.current_q = np.zeros(6, dtype=np.float64)
        self.last_solution = self.current_q.copy()
        self.last_publish_time = 0.0

        self.target_sub = self.create_subscription(PoseStamped, args.target_topic, self._target_callback, 20)
        self.joint_sub = self.create_subscription(JointState, args.joint_state_topic, self._joint_state_callback, 50)
        if args.fallback_joint_state_topic:
            self.fallback_joint_sub = self.create_subscription(
                JointState,
                args.fallback_joint_state_topic,
                self._joint_state_callback,
                50,
            )
        self.joint_pub = self.create_publisher(JointState, args.output_topic, 20)

        self.get_logger().info(
            f"Numeric IK ready: {args.target_topic} -> {args.output_topic}, "
            f"joint state: {args.joint_state_topic}"
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        if len(msg.position) < 6:
            return
        if msg.name:
            values = []
            for name in JOINT_NAMES:
                if name not in msg.name:
                    return
                values.append(msg.position[msg.name.index(name)])
            q = np.asarray(values, dtype=np.float64)
        else:
            q = np.asarray(msg.position[:6], dtype=np.float64)

        # 主仿真订阅 /joint_target 时会把 J3/J4 取反；这里读取仿真反馈保持模型内部真实角度。
        self.current_q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    def _target_callback(self, msg: PoseStamped) -> None:
        now = time.time()
        if self.args.max_publish_hz > 0 and now - self.last_publish_time < 1.0 / self.args.max_publish_hz:
            return

        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        quat_wxyz = np.array(
            [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z],
            dtype=np.float64,
        )
        target = _pose_matrix(pos, quat_wxyz)

        seed = self.current_q if self.args.seed_from_feedback else self.last_solution
        success, q_model, pos_err, ori_err = self.solver.solve(target, seed)
        self.last_solution = q_model.copy()

        q_cmd = q_model.copy()
        # main.py 的 /joint_target callback 会将 J3/J4 再取反，因此发布前反向补偿。
        if self.args.compensate_main_remap:
            q_cmd[2] *= -1.0
            q_cmd[3] *= -1.0

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = JOINT_NAMES
        out.position = [float(v) for v in q_cmd]
        out.velocity = [0.0] * 6
        self.joint_pub.publish(out)
        self.last_publish_time = now

        if success:
            self.get_logger().debug(f"IK ok pos_err={pos_err:.4f} ori_err={ori_err:.4f}")
        else:
            self.get_logger().warn(
                f"IK not fully converged, publishing best effort: pos_err={pos_err:.4f}, ori_err={ori_err:.4f}",
                throttle_duration_sec=1.0,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="ARM620 numeric IK ROS2 node")
    parser.add_argument("--target-topic", default="/ee_target")
    parser.add_argument("--joint-state-topic", default="/joint_states_sim")
    parser.add_argument("--fallback-joint-state-topic", default="/joint_states")
    parser.add_argument("--output-topic", default="/joint_target")
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--step-scale", type=float, default=0.7)
    parser.add_argument("--position-tolerance", type=float, default=0.003)
    parser.add_argument("--orientation-tolerance", type=float, default=0.04)
    parser.add_argument("--max-publish-hz", type=float, default=30.0)
    parser.add_argument("--seed-from-feedback", action="store_true", default=True)
    parser.add_argument("--no-main-remap-compensation", dest="compensate_main_remap", action="store_false")
    parser.set_defaults(compensate_main_remap=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = Arm620NumericIKNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
