#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ -f /opt/ros/humble/setup.bash ]; then
  # 加载系统 ROS2 环境，确保 rclpy 与系统 Python 一致
  . /opt/ros/humble/setup.bash
fi

cd "$PROJECT_ROOT/mujoco_env"
exec /usr/bin/python3 main/main.py "$@"
