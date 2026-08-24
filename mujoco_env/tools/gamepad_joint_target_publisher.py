#!/usr/bin/env python3
"""
Publish /joint_target from an Xbox controller.

Default mapping:
  left stick X  -> joint1
  left stick Y  -> joint2
  right stick Y -> joint3
  right stick X -> joint4
  joint5 stays at the initial value
  LT / RT       -> joint5
  LB / RB       -> joint6
  View + Menu   -> reset to initial joints

Run:
  source /opt/ros/humble/setup.bash
  python3 mujoco_env/tools/gamepad_joint_target_publisher.py
"""

from __future__ import annotations

import argparse
import math
import os
import select
import struct
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_LEFT_TRIGGER = 2
AXIS_RIGHT_X = 3
AXIS_RIGHT_Y = 4
AXIS_RIGHT_TRIGGER = 5

BUTTON_LB = 4
BUTTON_RB = 5
BUTTON_VIEW = 6
BUTTON_MENU = 7

# Initial position in degrees: [0, 0, -90, 0, -90, 0].
# /joint_target position uses radians.
INITIAL_JOINTS = [0.0, 0.0, math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 1.0]
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
JOINT_LIMITS = [
    (-2.9670, 2.9670),
    (-1.5708, 1.5708),
    (-1.5708, 1.5708),
    (-2.9670, 2.9670),
    (-1.5708, 1.5708),
    (-2.9670, 2.9670),
]


@dataclass
class GamepadState:
    axes: list[int]
    buttons: list[int]


def normalize_axis(value: int, deadzone: float) -> float:
    if value < 0:
        out = value / 32768.0
    else:
        out = value / 32767.0
    out = max(-1.0, min(1.0, out))
    return 0.0 if abs(out) < deadzone else out


def normalize_trigger(value: int, deadzone: float) -> float:
    # Xbox triggers are usually -32767 when released and +32767 when fully pressed.
    out = (value + 32767.0) / 65534.0
    out = max(0.0, min(1.0, out))
    return 0.0 if out < deadzone else out


def read_gamepad_events(fd: int, state: GamepadState) -> None:
    while True:
        readable, _, _ = select.select([fd], [], [], 0.0)
        if not readable:
            return

        try:
            data = os.read(fd, 8)
        except BlockingIOError:
            return

        if len(data) != 8:
            return

        _time_ms, value, event_type, number = struct.unpack("IhBB", data)
        event_type = event_type & ~JS_EVENT_INIT

        if event_type == JS_EVENT_AXIS and number < len(state.axes):
            state.axes[number] = value
        elif event_type == JS_EVENT_BUTTON and number < len(state.buttons):
            state.buttons[number] = value


class GamepadJointTargetPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("gamepad_joint_target_publisher")
        self.args = args
        self.publisher = self.create_publisher(JointState, args.topic, 10)
        self.joints = list(INITIAL_JOINTS)
        self.state = GamepadState(axes=[0] * 8, buttons=[0] * 11)
        self.fd = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK)
        self.last_time = time.monotonic()
        self.timer = self.create_timer(1.0 / args.rate, self.update)
        self.reset_latched = False

        self.get_logger().info(f"reading gamepad: {args.device}")
        self.get_logger().info(f"publishing: {args.topic}")
        self.get_logger().info(f"initial joints: {self.joints}")

    def destroy_node(self) -> bool:
        try:
            os.close(self.fd)
        except OSError:
            pass
        return super().destroy_node()

    def update(self) -> None:
        now = time.monotonic()
        dt = max(0.0, min(now - self.last_time, 0.1))
        self.last_time = now

        read_gamepad_events(self.fd, self.state)

        if self.state.buttons[BUTTON_VIEW] and self.state.buttons[BUTTON_MENU]:
            if not self.reset_latched:
                self.joints = list(INITIAL_JOINTS)
                self.get_logger().info("reset joints to initial position")
            self.reset_latched = True
        else:
            self.reset_latched = False

            left_x = normalize_axis(self.state.axes[AXIS_LEFT_X], self.args.deadzone)
            left_y = normalize_axis(self.state.axes[AXIS_LEFT_Y], self.args.deadzone)
            right_x = normalize_axis(self.state.axes[AXIS_RIGHT_X], self.args.deadzone)
            right_y = normalize_axis(self.state.axes[AXIS_RIGHT_Y], self.args.deadzone)
            lt = normalize_trigger(self.state.axes[AXIS_LEFT_TRIGGER], self.args.trigger_deadzone)
            rt = normalize_trigger(self.state.axes[AXIS_RIGHT_TRIGGER], self.args.trigger_deadzone)

            self.joints[0] += left_x * self.args.joint_speed * dt
            self.joints[1] += -left_y * self.args.joint_speed * dt
            self.joints[2] += -right_y * self.args.joint_speed * dt
            self.joints[3] += right_x * self.args.joint_speed * dt

            joint5_delta = (rt - lt) * self.args.joint5_speed * dt
            self.joints[4] += joint5_delta

            joint6_delta = self.state.buttons[BUTTON_RB] * self.args.button_speed * dt
            joint6_delta -= self.state.buttons[BUTTON_LB] * self.args.button_speed * dt
            self.joints[5] += joint6_delta

            for i, (lower, upper) in enumerate(JOINT_LIMITS):
                self.joints[i] = max(lower, min(upper, self.joints[i]))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = self.joints
        self.publisher.publish(msg)

        if self.args.print:
            print(
                "joint_target: "
                + " ".join(f"j{i + 1}={value:+.3f}" for i, value in enumerate(self.joints)),
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish /joint_target from an Xbox controller.")
    parser.add_argument("--device", default="/dev/input/js0", help="Joystick device. Default: /dev/input/js0")
    parser.add_argument("--topic", default="/joint_target", help="ROS2 topic. Default: /joint_target")
    parser.add_argument("--rate", type=float, default=50.0, help="Publish rate in Hz. Default: 50")
    parser.add_argument("--deadzone", type=float, default=0.08, help="Stick deadzone. Default: 0.08")
    parser.add_argument("--trigger-deadzone", type=float, default=0.05, help="Trigger deadzone. Default: 0.05")
    parser.add_argument("--joint-speed", type=float, default=0.8, help="Joint 1-4 speed rad/s. Default: 0.8")
    parser.add_argument("--joint5-speed", type=float, default=0.8, help="LT/RT joint5 speed rad/s. Default: 0.8")
    parser.add_argument("--button-speed", type=float, default=0.5, help="LB/RB joint6 speed rad/s. Default: 0.5")
    parser.add_argument("--print", action="store_true", help="Print published joint values.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = GamepadJointTargetPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
