#!/usr/bin/python3
"""
机械臂直接控制测试入口。

说明:
1. 当前项目已移除 RL / IK / 插值轨迹测试链路
2. 该脚本仅复用主程序的直接控制入口
3. 推荐统一从 main.py 维护运行逻辑
"""

import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(CURRENT_DIR, "../main")
sys.path.append(MAIN_DIR)

from main import test_integrated_trajectory_control


if __name__ == "__main__":
    test_integrated_trajectory_control(
        enable_visualization=True,
        enable_ros_control=True,
        enable_rosbag=False,
        enable_topic_rename=True,
        enable_image_publish=True,
        enable_depth_render=True,
        enable_thermal=True,
        enable_tactile_ui=False,
        enable_task_snapshot_info=False,
        enable_teaching_recording=True,
        control_loop_hz=15.0,
        thermal_render_hz=15.0,
    )
