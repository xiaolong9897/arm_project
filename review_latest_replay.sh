#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ -f /opt/ros/humble/setup.bash ]; then
  . /opt/ros/humble/setup.bash
fi

cd "$PROJECT_ROOT"
exec /usr/bin/python3 mujoco_env/main/replay.py --review "$@"
