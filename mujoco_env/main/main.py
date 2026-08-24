#!/usr/bin/python3
"""
机械臂仿真主程序

功能流程:
1. 直接关节控制与仿真执行
2. ROS2 Joint State发布 (实时发布关节状态到 /joint_states 话题)
3. 图像显示与自动视频保存

使用方法:
    # 使用系统Python以支持ROS2功能


    source /opt/ros/humble/setup.bash
    /usr/bin/python3 main/main.py
    
ROS2功能:
    - 自动发布关节状态到 /joint_states 话题 (500Hz)
    - 订阅关节控制命令从 /joint_target 话题
    - 直接控制模式:
      * ROS2控制: 收到命令时立即响应
    - 查看关节状态:
      source /opt/ros/humble/setup.bash
      ros2 topic echo /joint_states
    - 控制机械臂:
      ros2 topic pub /joint_target sensor_msgs/msg/JointState \
        '{name: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], 
          position: [0.8, 0.32, 1.07, 0.04, -1.57, 0.02]}'


      ros2 topic pub /joint_target sensor_msgs/msg/JointState \
        '{name: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], 
          position: [2.57, 0.57, 1.57, 0.00, -1.57, 0.00]}'





视频功能:
    - 自动保存机器人运行视频 (60FPS)
    - 文件名格式: robot_YYYYMMDD_HHMMSS.mp4
    - 按 'q' 键退出并保存视频
"""


import numpy as np
import sys
import os

import time
import mujoco
import logging
import threading
import argparse
import json
import cv2
import subprocess
from pathlib import Path
from queue import Queue
from typing import Optional, Tuple, Callable, Dict, Any
from datetime import datetime
from mujoco.renderer import Renderer

from gripper_status_bits import (
    GRIPPER_STATUS_IDLE_BIT,
    GripperStatusEncoder,
    set_status_bit,
)

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../sensors'))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PYTHON = "/usr/bin/python3"
PROFILE_LOGGER = logging.getLogger("arm_project.profile")


def _enforce_system_python() -> None:
    """
    强制主程序运行在系统 Python 下，避免 conda 环境里的 ROS2 依赖冲突。
    """
    current_python = os.path.realpath(sys.executable)
    expected_python = os.path.realpath(SYSTEM_PYTHON)
    if current_python != expected_python:
        raise SystemExit(
            "\n❌ 检测到当前解释器不是系统 Python。\n"
            f"   当前: {current_python}\n"
            f"   要求: {expected_python}\n"
            "   请改用以下方式启动:\n"
            "   1. ./run_system_python.sh\n"
            f"   2. {SYSTEM_PYTHON} mujoco_env/main/main.py\n"
        )


_enforce_system_python()

from env import ArmEnv

# 导入 ROS2 RGBD 发布器（使用标准库 cv_bridge）
try:
    from rgbd_ros2_publisher import RGBD_ROS2_Publisher
    ros2_publisher_available = True
except Exception:
    RGBD_ROS2_Publisher = None
    ros2_publisher_available = False
    print("⚠ RGBD_ROS2_Publisher not available")

# 导入热成像传感器
try:
    from thermal_sensor import ThermalSensor
    thermal_sensor_available = True
except ImportError:
    ThermalSensor = None
    thermal_sensor_available = False
    print("⚠ ThermalSensor not available")

from RGBD import MuJoCo_RGBD_Sensor

# 导入共享内存相关
try:
    from multiprocessing import shared_memory
    import pickle
    shared_memory_available = True
except ImportError:
    shared_memory_available = False
    print("⚠ shared_memory not available")

# 触觉可视化（直接读 MuJoCo 传感器，独立于 ROS2）
try:
    from tactile_visualizer import GripperTactileVisualizer
    tactile_visualizer_available = True
except Exception:
    GripperTactileVisualizer = None
    tactile_visualizer_available = False
    print("⚠ GripperTactileVisualizer not available")


class ROS2PublisherThread:
    """
    独立的 ROS2 发布线程
    通过共享内存获取图像数据并发布到 ROS2
    """

    def __init__(
        self,
        camera_configs: list,
        camera_info_publish_hz: float = 1.0,
    ):
        """
        初始化 ROS2 发布线程

        Args:
            camera_configs: 相机配置列表，每个配置包含 camera_name 和 node_name
                           例如: [{'camera_name': 'ee_camera', 'node_name': 'ee_camera_publisher'}]
        """
        self.camera_configs = camera_configs
        self.camera_info_publish_hz = float(camera_info_publish_hz)
        self.publishers = {}
        self.is_running = False
        self.thread = None

        # 共享内存字典 {camera_name: {'rgb': shm, 'depth': shm, 'intrinsics': shm, 'ready': bool}}
        self.shared_memory = {}
        self.memory_lock = threading.Lock()

        # 图像数据缓存
        self.image_cache = {}

        # 触觉数据缓存
        self.tactile_cache = {
            'left_vec': None,
            'right_vec': None,
            'left_pad': None,
            'right_pad': None,
            'left_tan': None,
            'right_tan': None,
            'ready': False,
            'timestamp_sec': 0.0,
        }

        # 触觉 topic 发布器（逐点向量+切向）
        self._tactile_enabled = False
        self._tactile_ctx = None
        self._tactile_node = None
        self._tactile_array_msg_type = None
        self._tactile_dim_msg_type = None
        self._tactile_left_vector_pub = None
        self._tactile_right_vector_pub = None
        self._tactile_left_pad_pub = None
        self._tactile_right_pad_pub = None
        self._tactile_left_tangential_pub = None
        self._tactile_right_tangential_pub = None

    def initialize_publishers(self):
        """初始化所有 ROS2 发布器"""
        image_publish_ok = False

        try:
            if ros2_publisher_available:
                for config in self.camera_configs:
                    camera_name = config['camera_name']
                    node_name = config['node_name']

                    publisher = RGBD_ROS2_Publisher(
                        camera_name=camera_name,
                        node_name=node_name,
                        camera_info_publish_hz=self.camera_info_publish_hz,
                    )
                    self.publishers[camera_name] = publisher

                    # 初始化共享内存槽位
                    self.shared_memory[camera_name] = {
                        'rgb': None,
                        'depth': None,
                        'intrinsics': None,
                        'ready': False
                    }
                image_publish_ok = len(self.publishers) > 0
            else:
                print("⚠ RGBD ROS2 发布器不可用，将仅尝试触觉数据发布")

            # 初始化触觉发布器（不依赖 cv_bridge）
            if ros2_available():
                try:
                    import rclpy
                    from rclpy.node import Node
                    from std_msgs.msg import Float32MultiArray, MultiArrayDimension

                    self._tactile_ctx = rclpy.Context()
                    rclpy.init(context=self._tactile_ctx)
                    node_name = f"tactile_array_publisher_{int(time.time() * 1000) % 10000}"
                    self._tactile_node = Node(node_name, context=self._tactile_ctx)
                    self._tactile_array_msg_type = Float32MultiArray
                    self._tactile_dim_msg_type = MultiArrayDimension

                    self._tactile_left_vector_pub = self._tactile_node.create_publisher(
                        Float32MultiArray, '/gripper_tactile/left/vector', 10
                    )
                    self._tactile_right_vector_pub = self._tactile_node.create_publisher(
                        Float32MultiArray, '/gripper_tactile/right/vector', 10
                    )
                    self._tactile_left_tangential_pub = self._tactile_node.create_publisher(
                        Float32MultiArray, '/gripper_tactile/left/tangential', 10
                    )
                    self._tactile_right_tangential_pub = self._tactile_node.create_publisher(
                        Float32MultiArray, '/gripper_tactile/right/tangential', 10
                    )
                    self._tactile_left_pad_pub = self._tactile_node.create_publisher(
                        Float32MultiArray, '/gripper_tactile/left/pad', 10
                    )
                    self._tactile_right_pad_pub = self._tactile_node.create_publisher(
                        Float32MultiArray, '/gripper_tactile/right/pad', 10
                    )
                    self._tactile_enabled = True
                    print(f"✓ Tactile ROS2 publishers initialized (node: {node_name})")
                    print("  - /gripper_tactile/left/vector, /gripper_tactile/right/vector (10x5x3)")
                    print("  - /gripper_tactile/left/tangential, /gripper_tactile/right/tangential (10x5)")
                    print("  - /gripper_tactile/left/pad, /gripper_tactile/right/pad (10x5)")
                except Exception as tactile_err:
                    self._tactile_enabled = False
                    print(f"⚠ Tactile ROS2 publisher init failed: {tactile_err}")

            if image_publish_ok:
                print(f"✓ ROS2 发布线程初始化完成 ({len(self.publishers)} 个相机)")
            elif not self._tactile_enabled:
                print("❌ ROS2 发布器不可用")

            return image_publish_ok or self._tactile_enabled

        except Exception as e:
            print(f"❌ ROS2 发布器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_image_data(self, camera_name: str, rgb, depth, intrinsics, timestamp_sec):
        """
        更新图像数据到共享缓存

        Args:
            camera_name: 相机名称
            rgb: RGB 图像
            depth: 深度图像
            intrinsics: 相机内参
            timestamp_sec: 时间戳
        """
        with self.memory_lock:
            self.image_cache[camera_name] = {
                'rgb': rgb.copy() if rgb is not None else None,
                'depth': depth.copy() if depth is not None else None,
                'intrinsics': intrinsics,
                'timestamp_sec': timestamp_sec,
                'ready': True
            }

    def publish_image_now(self, camera_name: str, rgb, depth, intrinsics, timestamp_sec):
        """
        直接发布图像，不经过中间缓存。

        用于“渲染完成后立即发布”的路径，减少数组 copy 和线程间搬运。
        """
        publisher = self.publishers.get(camera_name)
        if publisher is None or rgb is None:
            return False
        publisher.publish_rgbd(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            timestamp_sec=timestamp_sec,
        )
        return True

    def _build_multiarray(self, array: np.ndarray, labels: list[str]):
        """Build Float32MultiArray with explicit layout dimensions."""
        msg = self._tactile_array_msg_type()
        arr = np.asarray(array, dtype=np.float32)
        stride = int(arr.size)
        msg.layout.dim = []
        for idx, (label, size) in enumerate(zip(labels, arr.shape)):
            d = self._tactile_dim_msg_type()
            d.label = label
            d.size = int(size)
            if idx + 1 < arr.ndim:
                stride = int(np.prod(arr.shape[idx + 1:]))
            else:
                stride = 1
            d.stride = stride
            msg.layout.dim.append(d)
        msg.layout.data_offset = 0
        msg.data = arr.reshape(-1).tolist()
        return msg

    def update_tactile_data(self, tactile_components: dict, timestamp_sec: float):
        """Update tactile cache with per-taxel vectors and tangential forces."""
        if tactile_components is None:
            return

        left = tactile_components.get('left', {})
        right = tactile_components.get('right', {})
        left_fx = left.get('fx_grid')
        left_fy = left.get('fy_grid')
        left_fz = left.get('fz_signed_grid')
        right_fx = right.get('fx_grid')
        right_fy = right.get('fy_grid')
        right_fz = right.get('fz_signed_grid')
        left_pad = left.get('normal_grid')
        right_pad = right.get('normal_grid')
        left_tan = left.get('tangential_grid')
        right_tan = right.get('tangential_grid')

        if any(v is None for v in [left_fx, left_fy, left_fz, right_fx, right_fy, right_fz, left_pad, right_pad, left_tan, right_tan]):
            return

        left_vec = np.stack([left_fx, left_fy, left_fz], axis=-1).astype(np.float32, copy=False)
        right_vec = np.stack([right_fx, right_fy, right_fz], axis=-1).astype(np.float32, copy=False)

        with self.memory_lock:
            self.tactile_cache = {
                'left_vec': left_vec.copy(),
                'right_vec': right_vec.copy(),
                'left_pad': np.asarray(left_pad, dtype=np.float32).copy(),
                'right_pad': np.asarray(right_pad, dtype=np.float32).copy(),
                'left_tan': np.asarray(left_tan, dtype=np.float32).copy(),
                'right_tan': np.asarray(right_tan, dtype=np.float32).copy(),
                'ready': True,
                'timestamp_sec': float(timestamp_sec),
            }

    def _publish_loop(self):
        """ROS2 发布循环（在独立线程中运行）"""
        print("📡 ROS2 发布线程已启动")

        frame_count = 0
        tactile_count = 0

        while self.is_running:
            try:
                # 发布触觉逐点力数据
                if self._tactile_enabled:
                    with self.memory_lock:
                        tactile_ready = self.tactile_cache.get('ready', False)
                        if tactile_ready:
                            left_vec = self.tactile_cache.get('left_vec')
                            right_vec = self.tactile_cache.get('right_vec')
                            left_pad = self.tactile_cache.get('left_pad')
                            right_pad = self.tactile_cache.get('right_pad')
                            left_tan = self.tactile_cache.get('left_tan')
                            right_tan = self.tactile_cache.get('right_tan')
                            self.tactile_cache['ready'] = False
                        else:
                            left_vec = None
                            right_vec = None
                            left_pad = None
                            right_pad = None
                            left_tan = None
                            right_tan = None

                    if left_vec is not None and right_vec is not None and left_pad is not None and right_pad is not None:
                        try:
                            left_vec_msg = self._build_multiarray(left_vec, ['rows', 'cols', 'xyz'])
                            right_vec_msg = self._build_multiarray(right_vec, ['rows', 'cols', 'xyz'])
                            left_pad_msg = self._build_multiarray(left_pad, ['rows', 'cols'])
                            right_pad_msg = self._build_multiarray(right_pad, ['rows', 'cols'])
                            left_tan_msg = self._build_multiarray(left_tan, ['rows', 'cols'])
                            right_tan_msg = self._build_multiarray(right_tan, ['rows', 'cols'])
                            self._tactile_left_vector_pub.publish(left_vec_msg)
                            self._tactile_right_vector_pub.publish(right_vec_msg)
                            self._tactile_left_pad_pub.publish(left_pad_msg)
                            self._tactile_right_pad_pub.publish(right_pad_msg)
                            self._tactile_left_tangential_pub.publish(left_tan_msg)
                            self._tactile_right_tangential_pub.publish(right_tan_msg)
                            tactile_count += 1
                        except Exception as e:
                            if tactile_count % 100 == 0:
                                print(f"⚠ tactile 发布失败: {e}")

                # 短暂休眠，避免CPU占用过高
                time.sleep(0.001)  # 1ms

            except Exception as e:
                print(f"❌ ROS2 发布循环错误: {e}")
                import traceback
                traceback.print_exc()

        print("📡 ROS2 发布线程已停止")

    def start(self):
        """启动 ROS2 发布线程"""
        if self.is_running:
            print("⚠ ROS2 发布线程已在运行")
            return False

        if not self.initialize_publishers():
            return False

        self.is_running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()

        print("✓ ROS2 发布线程已启动")
        return True

    def stop(self):
        """停止 ROS2 发布线程"""
        if not self.is_running:
            return

        print("🛑 正在停止 ROS2 发布线程...")
        self.is_running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("⚠ ROS2 发布线程未能在超时时间内结束")

        # 清理发布器
        for publisher in self.publishers.values():
            try:
                publisher.shutdown()
            except:
                pass

        if self._tactile_node is not None:
            try:
                self._tactile_node.destroy_node()
            except Exception:
                pass
        if self._tactile_ctx is not None:
            try:
                self._tactile_ctx.shutdown()
            except Exception:
                pass

        print("✓ ROS2 发布线程已停止")


class ReplicaRenderThread:
    """使用独立 MjData 副本的离屏渲染线程。"""

    def __init__(
        self,
        robot,
        enable_depth_render: bool = True,
        enable_thermal: bool = True,
        thermal_sensor_initializer: Optional[Callable[[Any], None]] = None,
        frame_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        rgbd_publishers: Optional[Dict[str, Any]] = None,
        enable_profile_logs: bool = False,
    ):
        self.robot = robot
        self.enable_depth_render = bool(enable_depth_render)
        self.enable_thermal = bool(enable_thermal and thermal_sensor_available and ThermalSensor is not None)
        self.thermal_sensor_initializer = thermal_sensor_initializer
        self.frame_callback = frame_callback
        self.rgbd_publishers = rgbd_publishers or {}
        self.enable_profile_logs = bool(enable_profile_logs)

        self.thread = None
        self.is_running = False
        self.init_event = threading.Event()
        self.init_error = None
        self.frame_lock = threading.Lock()

        self.render_data = mujoco.MjData(self.robot.model)
        self.ee_sensor = None
        self.external_sensor = None
        self.thermal_sensor = None
        self.renderers = []
        self.body_temperatures: Dict[int, float] = {}
        self.render_model = self.robot.model

        self.state_spec = (
            int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
            | int(mujoco.mjtState.mjSTATE_MOCAP_POS)
            | int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
            | int(mujoco.mjtState.mjSTATE_USERDATA)
        )
        self.state_buffer = np.zeros(
            mujoco.mj_stateSize(self.robot.model, self.state_spec),
            dtype=np.float64,
        )

        self.latest_frames = {
            "ee_intrinsics": None,
            "external_intrinsics": None,
            "timestamp_sec": 0.0,
            "frame_id": 0,
            "thermal_frame_id": 0,
        }
        self._profile_window = 120
        self._profile_stats = {
            "sync_state_ms": [],
            "state_forward_ms": [],
            "ee_render_ms": [],
            "external_render_ms": [],
            "thermal_render_ms": [],
            "callback_ms": [],
            "frame_total_ms": [],
        }

    def _profile_record(self, key: str, value_ms: float):
        if key in self._profile_stats:
            self._profile_stats[key].append(float(value_ms))

    def _profile_flush_if_needed(self, frame_id: int):
        if (not self.enable_profile_logs) or frame_id <= 0 or (frame_id % self._profile_window) != 0:
            return

        def _fmt(values):
            if not values:
                return "avg=0.00 max=0.00"
            arr = np.asarray(values, dtype=np.float64)
            return f"avg={arr.mean():.2f} max={arr.max():.2f}"

        PROFILE_LOGGER.info(
            "\n[RenderProfile] frame=%s\n"
            "  sync_state_ms:    %s\n"
            "  state_forward_ms: %s\n"
            "  ee_render_ms:     %s\n"
            "  external_render_ms:%s\n"
            "  thermal_render_ms:%s\n"
            "  callback_ms:      %s\n"
            "  frame_total_ms:   %s",
            frame_id,
            _fmt(self._profile_stats['sync_state_ms']),
            _fmt(self._profile_stats['state_forward_ms']),
            _fmt(self._profile_stats['ee_render_ms']),
            _fmt(self._profile_stats['external_render_ms']),
            _fmt(self._profile_stats['thermal_render_ms']),
            _fmt(self._profile_stats['callback_ms']),
            _fmt(self._profile_stats['frame_total_ms']),
        )

        for values in self._profile_stats.values():
            values.clear()

    def _init_renderers(self):
        model_path = getattr(self.robot, "model_path", None)
        if model_path:
            self.render_model = mujoco.MjModel.from_xml_path(model_path)
            self.render_data = mujoco.MjData(self.render_model)
        else:
            self.render_model = self.robot.model
            self.render_data = mujoco.MjData(self.render_model)

        self.state_buffer = np.zeros(
            mujoco.mj_stateSize(self.render_model, self.state_spec),
            dtype=np.float64,
        )

        ee_rgb_renderer = Renderer(self.render_model, height=480, width=640)
        ee_depth_renderer = None
        if self.enable_depth_render:
            ee_depth_renderer = Renderer(self.render_model, height=480, width=640)

        ext_rgb_renderer = Renderer(self.render_model, height=480, width=640)
        ext_depth_renderer = None
        if self.enable_depth_render:
            ext_depth_renderer = Renderer(self.render_model, height=480, width=640)

        self.renderers = [ee_rgb_renderer, ext_rgb_renderer]
        if ee_depth_renderer is not None:
            self.renderers.append(ee_depth_renderer)
        if ext_depth_renderer is not None:
            self.renderers.append(ext_depth_renderer)

        self.ee_sensor = MuJoCo_RGBD_Sensor(
            model=self.render_model,
            data=self.render_data,
            rgb_renderer=ee_rgb_renderer,
            depth_renderer=ee_depth_renderer,
            cam_name="rgbd_camera_ee",
            fps=30,
        )
        self.external_sensor = MuJoCo_RGBD_Sensor(
            model=self.render_model,
            data=self.render_data,
            rgb_renderer=ext_rgb_renderer,
            depth_renderer=ext_depth_renderer,
            cam_name="rgbd_camera_external",
            fps=30,
        )

        if self.enable_thermal:
            self.thermal_sensor = ThermalSensor(
                model=self.render_model,
                data=self.render_data,
                cam_name="rgbd_camera_ee",
                width=640,
                height=480,
                enable_thermal_blur=False,
                blur_kernel_size=7,
                blur_sigma=0.4,
                enable_distance_attenuation=True,
                attenuation_coefficient=0.05,
                enable_noise=False,
                noise_stddev=0.3,
                enable_internal_gradient=False,
                edge_temperature_ratio=0.7,
            )
            if self.thermal_sensor_initializer is not None:
                self.thermal_sensor_initializer(self.thermal_sensor)
            self.body_temperatures = dict(self.thermal_sensor.body_temperatures)
            self.renderers.append(self.thermal_sensor.renderer)

        with self.frame_lock:
            self.latest_frames["ee_intrinsics"] = self.ee_sensor.intr.copy()
            self.latest_frames["external_intrinsics"] = self.external_sensor.intr.copy()

    def _sync_state_from_sim(self):
        acquired = self.robot._step_lock.acquire(timeout=0.01)
        if not acquired:
            return None
        try:
            mujoco.mj_getState(self.robot.model, self.robot.data, self.state_buffer, self.state_spec)
            sim_timestamp_sec = float(self.robot.sim_time)
        finally:
            self.robot._step_lock.release()
        return sim_timestamp_sec

    def _render_loop(self):
        try:
            self._init_renderers()
        except Exception as e:
            self.init_error = e
            self.init_event.set()
            return

        frame_id = 0
        thermal_frame_id = 0
        self.init_event.set()

        while self.is_running:
            frame_wall_start = time.perf_counter()
            if self.robot._render_pause_event.is_set():
                time.sleep(0.0005)
                continue

            sync_start = time.perf_counter()
            sim_timestamp_sec = self._sync_state_from_sim()
            self._profile_record("sync_state_ms", (time.perf_counter() - sync_start) * 1000.0)
            if sim_timestamp_sec is None:
                time.sleep(0.0005)
                continue

            forward_start = time.perf_counter()
            mujoco.mj_setState(self.render_model, self.render_data, self.state_buffer, self.state_spec)
            mujoco.mj_forward(self.render_model, self.render_data)
            self._profile_record("state_forward_ms", (time.perf_counter() - forward_start) * 1000.0)

            ee_start = time.perf_counter()
            ee_rgb, ee_depth = self.ee_sensor.render_rgbd()
            self._profile_record("ee_render_ms", (time.perf_counter() - ee_start) * 1000.0)
            ext_start = time.perf_counter()
            ext_rgb, ext_depth = self.external_sensor.render_rgbd()
            self._profile_record("external_render_ms", (time.perf_counter() - ext_start) * 1000.0)

            self.robot._camera_render_count += 2
            self.robot._last_camera_render_time = time.time()

            thermal_rgb = None
            if self.thermal_sensor is not None:
                thermal_start = time.perf_counter()
                temperature_map, _ = self.thermal_sensor.render_thermal_image()
                rgb_image = self.thermal_sensor.render_rgb()
                thermal_rgb = self.thermal_sensor.postprocess_thermal_frame(
                    temperature_map=temperature_map,
                    rgb_image=rgb_image,
                    temp_min=0.0,
                    temp_max=100.0,
                    blur_kernel_size=25,
                    blur_sigma=5.0,
                    thermal_weight=0.85,
                )
                self._profile_record("thermal_render_ms", (time.perf_counter() - thermal_start) * 1000.0)
                thermal_frame_id += 1
                self.robot._camera_render_count += 1
                self.robot._last_camera_render_time = time.time()
            else:
                self._profile_record("thermal_render_ms", 0.0)

            frame_id += 1

            if self.frame_callback is not None:
                callback_start = time.perf_counter()
                try:
                    self.frame_callback(
                        {
                            "ee_rgb": ee_rgb,
                            "ee_depth": ee_depth,
                            "ee_intrinsics": self.ee_sensor.intr,
                            "external_rgb": ext_rgb,
                            "external_depth": ext_depth,
                            "external_intrinsics": self.external_sensor.intr,
                            "thermal_rgb": thermal_rgb,
                            "timestamp_sec": sim_timestamp_sec,
                            "frame_id": frame_id,
                            "thermal_frame_id": thermal_frame_id,
                        }
                    )
                except Exception as e:
                    print(f"⚠ 渲染帧回调失败: {e}")
                self._profile_record("callback_ms", (time.perf_counter() - callback_start) * 1000.0)
            else:
                self._profile_record("callback_ms", 0.0)

            with self.frame_lock:
                self.latest_frames["timestamp_sec"] = sim_timestamp_sec
                self.latest_frames["frame_id"] = frame_id
                self.latest_frames["thermal_frame_id"] = thermal_frame_id

            self._profile_record("frame_total_ms", (time.perf_counter() - frame_wall_start) * 1000.0)
            self._profile_flush_if_needed(frame_id)

        self._close_renderers()

    def _close_renderers(self):
        for renderer in self.renderers:
            try:
                if renderer is not None:
                    renderer.close()
            except Exception:
                pass
        self.renderers = []
        for publisher in self.rgbd_publishers.values():
            try:
                publisher.shutdown()
            except Exception:
                pass
        self.rgbd_publishers = {}

    def start(self) -> bool:
        if self.is_running:
            return True
        self.is_running = True
        self.thread = threading.Thread(target=self._render_loop, name="replica_render_loop", daemon=True)
        self.thread.start()
        self.init_event.wait(timeout=5.0)
        if self.init_error is not None:
            self.is_running = False
            print(f"❌ 渲染副本线程初始化失败: {self.init_error}")
            return False
        if not self.init_event.is_set():
            self.is_running = False
            print("❌ 渲染副本线程初始化超时")
            return False
        print("✓ Replica offscreen render thread started")
        print("  - render data: dedicated MjData replica")
        print(f"  - depth rendering: {'enabled' if self.enable_depth_render else 'disabled'}")
        print(f"  - thermal rendering: {'enabled' if self.thermal_sensor is not None else 'disabled'}")
        return True

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            self.thread = None
        self._close_renderers()

    def get_latest_frames(self) -> Dict[str, Any]:
        with self.frame_lock:
            return {
                "ee_intrinsics": None if self.latest_frames["ee_intrinsics"] is None else self.latest_frames["ee_intrinsics"].copy(),
                "external_intrinsics": None if self.latest_frames["external_intrinsics"] is None else self.latest_frames["external_intrinsics"].copy(),
                "timestamp_sec": float(self.latest_frames["timestamp_sec"]),
                "frame_id": int(self.latest_frames["frame_id"]),
                "thermal_frame_id": int(self.latest_frames["thermal_frame_id"]),
            }


class DirectFrameDispatcher:
    """
    渲染后处理分发器。

    作用:
    1. RGBD 渲染完成后立即直接发布到 ROS2
    2. Thermal 仍由独立线程渲染后直接发布
    3. 为 Teaching 录制按需复制图像
    """

    def __init__(
        self,
        ros2_publisher_thread=None,
        thermal_publisher=None,
        controller=None,
        thermal_publish_hz: float = 15.0,
        image_publish_hz: float = 30.0,
        enable_profile_logs: bool = False,
    ):
        self.ros2_publisher_thread = ros2_publisher_thread
        self.thermal_publisher = thermal_publisher
        self.controller = controller
        self.thermal_publish_hz = float(thermal_publish_hz) if thermal_publish_hz is not None else 0.0
        self.image_publish_hz = float(image_publish_hz) if image_publish_hz is not None else 0.0
        self.enable_profile_logs = bool(enable_profile_logs)

        self.last_thermal_frame_id = -1
        self.last_thermal_publish_wall_time = 0.0
        self.bridge_frame_count = 0
        self.last_dispatch_profile_frame_id = -1
        self._profile_window = 120
        self._profile_stats = {
            "thermal_publish_ms": [],
            "ee_publish_ms": [],
            "external_publish_ms": [],
            "teaching_queue_ms": [],
            "dispatch_total_ms": [],
        }

    def _profile_record(self, key: str, value_ms: float):
        if key in self._profile_stats:
            self._profile_stats[key].append(float(value_ms))

    def _profile_flush_if_needed(self, frame_id: int):
        if (not self.enable_profile_logs) or frame_id <= 0 or (frame_id % self._profile_window) != 0:
            return
        if frame_id == self.last_dispatch_profile_frame_id:
            return

        def _fmt(values):
            if not values:
                return "avg=0.00 max=0.00"
            arr = np.asarray(values, dtype=np.float64)
            return f"avg={arr.mean():.2f} max={arr.max():.2f}"

        PROFILE_LOGGER.info(
            "[DispatchProfile] frame=%s\n"
            "  thermal_publish_ms:%s\n"
            "  ee_publish_ms:     %s\n"
            "  external_publish_ms:%s\n"
            "  teaching_queue_ms: %s\n"
            "  dispatch_total_ms: %s",
            frame_id,
            _fmt(self._profile_stats['thermal_publish_ms']),
            _fmt(self._profile_stats['ee_publish_ms']),
            _fmt(self._profile_stats['external_publish_ms']),
            _fmt(self._profile_stats['teaching_queue_ms']),
            _fmt(self._profile_stats['dispatch_total_ms']),
        )

        self.last_dispatch_profile_frame_id = frame_id
        for values in self._profile_stats.values():
            values.clear()

    def start(self):
        print("✓ RGBD direct publish mode enabled (publish-on-render)")
        return True

    def stop(self):
        return None

    def handle_frame(self, render_frames: Dict[str, Any]):
        dispatch_start = time.perf_counter()
        ee_rgb = render_frames["ee_rgb"]
        ee_depth = render_frames["ee_depth"]
        ext_rgb = render_frames["external_rgb"]
        ext_depth = render_frames["external_depth"]
        thermal_rgb = render_frames["thermal_rgb"]
        ee_intrinsics = render_frames["ee_intrinsics"]
        ext_intrinsics = render_frames["external_intrinsics"]
        sim_timestamp_sec = float(render_frames["timestamp_sec"])
        thermal_frame_id = int(render_frames["thermal_frame_id"])
        frame_id = int(render_frames["frame_id"])

        self._profile_record("thermal_publish_ms", 0.0)

        if self.ros2_publisher_thread is not None:
            if ee_rgb is not None:
                ee_pub_start = time.perf_counter()
                self.ros2_publisher_thread.publish_image_now(
                    camera_name='ee_camera',
                    rgb=ee_rgb,
                    depth=ee_depth,
                    intrinsics=ee_intrinsics,
                    timestamp_sec=sim_timestamp_sec
                )
                self._profile_record("ee_publish_ms", (time.perf_counter() - ee_pub_start) * 1000.0)
            else:
                self._profile_record("ee_publish_ms", 0.0)

            if ext_rgb is not None:
                ext_pub_start = time.perf_counter()
                self.ros2_publisher_thread.publish_image_now(
                    camera_name='external_camera',
                    rgb=ext_rgb,
                    depth=ext_depth,
                    intrinsics=ext_intrinsics,
                    timestamp_sec=sim_timestamp_sec
                )
                self._profile_record("external_publish_ms", (time.perf_counter() - ext_pub_start) * 1000.0)
            else:
                self._profile_record("external_publish_ms", 0.0)
        else:
            self._profile_record("ee_publish_ms", 0.0)
            self._profile_record("external_publish_ms", 0.0)

        self._profile_record("teaching_queue_ms", 0.0)

        self.bridge_frame_count += 1
        self._profile_record("dispatch_total_ms", (time.perf_counter() - dispatch_start) * 1000.0)
        self._profile_flush_if_needed(frame_id)


class ThermalRenderThread:
    """独立热成像渲染线程。"""

    def __init__(
        self,
        robot,
        thermal_publish_hz: float = 15.0,
        thermal_sensor_initializer: Optional[Callable[[Any], None]] = None,
        thermal_publisher=None,
        enable_profile_logs: bool = False,
    ):
        self.robot = robot
        self.thermal_publish_hz = max(0.1, float(thermal_publish_hz))
        self.thermal_sensor_initializer = thermal_sensor_initializer
        self.thermal_publisher = thermal_publisher
        self.enable_profile_logs = bool(enable_profile_logs)

        self.thread = None
        self.is_running = False
        self.init_event = threading.Event()
        self.init_error = None

        self.render_data = mujoco.MjData(self.robot.model)
        self.thermal_sensor = None
        self.body_temperatures: Dict[int, float] = {}

        self.state_spec = (
            int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
            | int(mujoco.mjtState.mjSTATE_MOCAP_POS)
            | int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
            | int(mujoco.mjtState.mjSTATE_USERDATA)
        )
        self.state_buffer = np.zeros(
            mujoco.mj_stateSize(self.robot.model, self.state_spec),
            dtype=np.float64,
        )

        self._profile_window = 120
        self._profile_stats = {
            "sync_state_ms": [],
            "state_forward_ms": [],
            "thermal_seg_render_ms": [],
            "thermal_depth_render_ms": [],
            "thermal_rgb_render_ms": [],
            "thermal_postprocess_ms": [],
            "thermal_render_ms": [],
            "thermal_publish_ms": [],
            "frame_total_ms": [],
        }

    def _profile_record(self, key: str, value_ms: float):
        if key in self._profile_stats:
            self._profile_stats[key].append(float(value_ms))

    def _profile_flush_if_needed(self, frame_id: int):
        if (not self.enable_profile_logs) or frame_id <= 0 or (frame_id % self._profile_window) != 0:
            return

        def _fmt(values):
            if not values:
                return "avg=0.00 max=0.00"
            arr = np.asarray(values, dtype=np.float64)
            return f"avg={arr.mean():.2f} max={arr.max():.2f}"

        PROFILE_LOGGER.info(
            "[ThermalProfile] frame=%s\n"
            "  sync_state_ms:    %s\n"
            "  state_forward_ms: %s\n"
            "  thermal_seg_render_ms:%s\n"
            "  thermal_depth_render_ms:%s\n"
            "  thermal_rgb_render_ms:%s\n"
            "  thermal_postprocess_ms:%s\n"
            "  thermal_render_ms:%s\n"
            "  thermal_publish_ms:%s\n"
            "  frame_total_ms:   %s",
            frame_id,
            _fmt(self._profile_stats['sync_state_ms']),
            _fmt(self._profile_stats['state_forward_ms']),
            _fmt(self._profile_stats['thermal_seg_render_ms']),
            _fmt(self._profile_stats['thermal_depth_render_ms']),
            _fmt(self._profile_stats['thermal_rgb_render_ms']),
            _fmt(self._profile_stats['thermal_postprocess_ms']),
            _fmt(self._profile_stats['thermal_render_ms']),
            _fmt(self._profile_stats['thermal_publish_ms']),
            _fmt(self._profile_stats['frame_total_ms']),
        )
        for values in self._profile_stats.values():
            values.clear()

    def _init_sensor(self):
        self.thermal_sensor = ThermalSensor(
            model=self.robot.model,
            data=self.render_data,
            cam_name="rgbd_camera_ee",
            width=640,
            height=480,
            enable_thermal_blur=False,
            blur_kernel_size=7,
            blur_sigma=0.4,
            enable_distance_attenuation=True,
            attenuation_coefficient=0.05,
            enable_noise=False,
            noise_stddev=0.3,
            enable_internal_gradient=False,
            edge_temperature_ratio=0.7,
        )
        if self.thermal_sensor_initializer is not None:
            self.thermal_sensor_initializer(self.thermal_sensor)
        self.body_temperatures = dict(self.thermal_sensor.body_temperatures)

    def _sync_state_from_sim(self):
        acquired = self.robot._step_lock.acquire(timeout=0.01)
        if not acquired:
            return None
        try:
            mujoco.mj_getState(self.robot.model, self.robot.data, self.state_buffer, self.state_spec)
            sim_timestamp_sec = float(self.robot.sim_time)
        finally:
            self.robot._step_lock.release()
        return sim_timestamp_sec

    def _loop(self):
        try:
            self._init_sensor()
        except Exception as e:
            self.init_error = e
            self.init_event.set()
            return

        frame_id = 0
        self.init_event.set()
        period_s = 1.0 / self.thermal_publish_hz
        next_publish_time = time.perf_counter()

        while self.is_running:
            now = time.perf_counter()
            if now < next_publish_time:
                time.sleep(next_publish_time - now)
            frame_start = time.perf_counter()
            if self.robot._render_pause_event.is_set():
                time.sleep(0.001)
                continue

            sync_start = time.perf_counter()
            sim_timestamp_sec = self._sync_state_from_sim()
            self._profile_record("sync_state_ms", (time.perf_counter() - sync_start) * 1000.0)
            if sim_timestamp_sec is None:
                time.sleep(0.0005)
                continue

            forward_start = time.perf_counter()
            mujoco.mj_setState(self.robot.model, self.render_data, self.state_buffer, self.state_spec)
            mujoco.mj_forward(self.robot.model, self.render_data)
            self._profile_record("state_forward_ms", (time.perf_counter() - forward_start) * 1000.0)

            render_start = time.perf_counter()
            seg_start = time.perf_counter()
            self.thermal_sensor.renderer.update_scene(self.render_data, camera=self.thermal_sensor.camera_id)
            self.thermal_sensor._scene_cache_valid = True
            self.thermal_sensor._scene_cache_camera_id = self.thermal_sensor.camera_id
            self.thermal_sensor.renderer.enable_segmentation_rendering()
            seg_image = self.thermal_sensor.renderer.render()
            self.thermal_sensor.renderer.disable_segmentation_rendering()
            self._profile_record("thermal_seg_render_ms", (time.perf_counter() - seg_start) * 1000.0)

            depth_start = time.perf_counter()
            self.thermal_sensor.renderer.enable_depth_rendering()
            depth_image = self.thermal_sensor.renderer.render()
            self.thermal_sensor.renderer.disable_depth_rendering()
            self._profile_record("thermal_depth_render_ms", (time.perf_counter() - depth_start) * 1000.0)

            temp_map = np.full((self.thermal_sensor.height, self.thermal_sensor.width), self.thermal_sensor.ambient_temperature, dtype=np.float32)
            body_id_map = np.full((self.thermal_sensor.height, self.thermal_sensor.width), -1, dtype=np.int32)
            geom_id_image = seg_image[:, :, 0].astype(np.int32)
            valid_mask = (geom_id_image >= 0) & (geom_id_image < self.robot.model.ngeom)
            if valid_mask.any():
                body_id_map[valid_mask] = self.robot.model.geom_bodyid[geom_id_image[valid_mask]]
                unique_geom_ids = np.unique(geom_id_image[valid_mask])
                geom_temp_lut = np.full(self.robot.model.ngeom, self.thermal_sensor.ambient_temperature, dtype=np.float32)
                for geom_id in unique_geom_ids:
                    body_id = int(self.robot.model.geom_bodyid[geom_id])
                    geom_temp_lut[geom_id] = self.thermal_sensor._compute_internal_temperature(body_id, geom_id, (0, 0))
                temp_map[valid_mask] = geom_temp_lut[geom_id_image[valid_mask]]
                if self.thermal_sensor.enable_distance_attenuation:
                    dist_mask = valid_mask & (depth_image > 0)
                    temp_diff = temp_map[dist_mask] - self.thermal_sensor.ambient_temperature
                    temp_map[dist_mask] = (
                        self.thermal_sensor.ambient_temperature
                        + temp_diff * np.exp(-self.thermal_sensor.attenuation_coefficient * depth_image[dist_mask])
                    )
                if self.thermal_sensor.enable_noise:
                    noise = np.random.normal(0, self.thermal_sensor.noise_stddev, (self.thermal_sensor.height, self.thermal_sensor.width))
                    temp_map += noise
                if self.thermal_sensor.enable_thermal_blur:
                    temp_map = self.thermal_sensor._apply_thermal_blur(temp_map, body_id_map)

            rgb_start = time.perf_counter()
            rgb_image = self.thermal_sensor.render_rgb()
            self._profile_record("thermal_rgb_render_ms", (time.perf_counter() - rgb_start) * 1000.0)

            post_start = time.perf_counter()
            thermal_rgb = self.thermal_sensor.postprocess_thermal_frame(
                temperature_map=temp_map,
                rgb_image=rgb_image,
                temp_min=0.0,
                temp_max=100.0,
                blur_kernel_size=25,
                blur_sigma=5.0,
                thermal_weight=0.85,
            )
            self._profile_record("thermal_postprocess_ms", (time.perf_counter() - post_start) * 1000.0)
            self._profile_record("thermal_render_ms", (time.perf_counter() - render_start) * 1000.0)

            if self.thermal_publisher is not None:
                publish_start = time.perf_counter()
                self.thermal_publisher.publish(thermal_rgb, stamp_sec=sim_timestamp_sec)
                self._profile_record("thermal_publish_ms", (time.perf_counter() - publish_start) * 1000.0)
            else:
                self._profile_record("thermal_publish_ms", 0.0)

            frame_id += 1
            self.robot._camera_render_count += 1
            self.robot._last_camera_render_time = time.time()
            elapsed_s = time.perf_counter() - frame_start
            self._profile_record("frame_total_ms", elapsed_s * 1000.0)
            self._profile_flush_if_needed(frame_id)

            next_publish_time += period_s
            now = time.perf_counter()
            if next_publish_time < now - period_s:
                next_publish_time = now

        try:
            if self.thermal_sensor is not None and self.thermal_sensor.renderer is not None:
                self.thermal_sensor.renderer.close()
        except Exception:
            pass

    def start(self) -> bool:
        if self.is_running:
            return True
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, name="thermal_render_loop", daemon=True)
        self.thread.start()
        self.init_event.wait(timeout=5.0)
        if self.init_error is not None:
            self.is_running = False
            print(f"❌ 热成像线程初始化失败: {self.init_error}")
            return False
        if not self.init_event.is_set():
            self.is_running = False
            print("❌ 热成像线程初始化超时")
            return False
        print(f"✓ Thermal render thread started ({self.thermal_publish_hz:.1f}Hz)")
        return True

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            self.thread = None


# ROS2 Joint State Subscriber (optional)
try:
    from ros2_joint_state_subscriber import JointStateSubscriberROS2, ros2_available
except ImportError:
    JointStateSubscriberROS2 = None
    def ros2_available():
        return False


class EETargetSubscriberROS2:
    """
    订阅 /ee_target 话题，并提取 base_link 坐标系下的目标末端姿态。
    """

    def __init__(
        self,
        topic: str = "/ee_target",
        node_name: str = "ee_target_subscriber",
        callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
    ) -> None:
        if not ros2_available():
            raise ImportError("ROS2 not available")

        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped

        self.callback = callback
        self.latest_position = None
        self.latest_quaternion = None
        self.received_count = 0
        self._reported_first_message = False
        self.lock = threading.Lock()

        self.context = rclpy.Context()
        rclpy.init(context=self.context)
        self.node = Node(node_name, context=self.context)
        self.pose_subscription = self.node.create_subscription(
            PoseStamped,
            topic,
            self._pose_stamped_callback,
            50,
        )
        self.executor = rclpy.executors.SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.running = False
        self.spin_thread = None

        print("✓ EE target subscriber initialized")
        print(f"  - Topic: {topic}")
        print("  - Type: geometry_msgs/PoseStamped")
        print(f"  - Node: {node_name}")

    def _store_pose(self, position: np.ndarray, quaternion: np.ndarray, source_type: str) -> None:
        with self.lock:
            self.latest_position = position
            self.latest_quaternion = quaternion
            self.received_count += 1
        if not self._reported_first_message:
            self._reported_first_message = True
            print(
                f"✓ /ee_target first message received ({source_type}, base_link frame): "
                f"pos={position.tolist()}, quat_wxyz={quaternion.tolist()}"
            )
        if self.callback is not None:
            self.callback(position.copy(), quaternion.copy())

    def _pose_stamped_callback(self, msg):
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            dtype=np.float64,
        )
        quat_norm = float(np.linalg.norm(quaternion))
        if quat_norm < 1e-8:
            quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            quaternion = quaternion / quat_norm
        self._store_pose(position, quaternion, "PoseStamped")

    def get_latest_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self.lock:
            if self.latest_position is None:
                return None
            position = self.latest_position.copy()
            if self.latest_quaternion is None:
                quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                quaternion = self.latest_quaternion.copy()
            return position, quaternion

    def start_spinning(self):
        if self.running:
            return
        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

    def stop_spinning(self):
        if not self.running:
            return
        self.running = False
        if self.spin_thread is not None:
            self.spin_thread.join(timeout=1.0)
            self.spin_thread = None

    def _spin_loop(self):
        while self.running:
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                if self.running:
                    print(f"❌ EE target spin error: {e}")
                break

    def shutdown(self):
        self.stop_spinning()
        try:
            self.executor.remove_node(self.node)
            self.executor.shutdown()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            import rclpy
            rclpy.shutdown(context=self.context)
        except Exception:
            pass


def _find_robot_base_body_id(model) -> int:
    for candidate_name in ['base_link', 'base']:
        candidate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate_name)
        if candidate_id >= 0:
            return int(candidate_id)
    return -1


def _pose_from_base_link_to_world(
    robot,
    base_body_id: int,
    position_base: np.ndarray,
    quaternion_base: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 base_link 下的目标姿态转换到 MuJoCo world。

    位置变换公式：
      p_world = R_world_base * p_base + t_world_base
    姿态变换公式：
      R_world_target = R_world_base * R_base_target
    其中 R_world_base/t_world_base 来自 MuJoCo body 的 xmat/xpos。
    """
    if base_body_id < 0:
        return position_base.copy(), quaternion_base.copy()

    base_rot_world = robot.data.xmat[base_body_id].reshape(3, 3).copy()
    base_pos_world = robot.data.xpos[base_body_id].copy()
    position_world = base_pos_world + base_rot_world @ position_base

    from scipy.spatial.transform import Rotation as R

    base_rot_obj = R.from_matrix(base_rot_world)
    target_rot_base = R.from_quat([
        quaternion_base[1],
        quaternion_base[2],
        quaternion_base[3],
        quaternion_base[0],
    ])
    target_rot_world = base_rot_obj * target_rot_base
    quat_xyzw = target_rot_world.as_quat()
    quaternion_world = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    return position_world.astype(np.float64), quaternion_world

# Teaching Status Subscriber
try:
    from teaching_status_subscriber import TeachingStatusSubscriber
    teaching_recording_available = True
except ImportError:
    TeachingStatusSubscriber = None
    teaching_recording_available = False

# Cartesian Target / RRT / IK 链路已移除

# ROS2 Rosbag Recorder
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../tools/ros2_recorder'))
    from rosbag_recorder import RosbagRecorder
    rosbag_recorder_available = True
except ImportError:
    RosbagRecorder = None
    rosbag_recorder_available = False
    print("⚠ RosbagRecorder not available")

class ContinuousArmController:
    """
    连续机械臂控制器。
    
    功能:
    1. 全程保持机械臂控制
    2. 接收 /joint_target 关节命令
    3. 在没有新命令时维持当前状态
    4. 处理示教记录、状态发布与安全 reset
    """
    
    def __init__(
        self,
        env: ArmEnv,
        enable_ros_control: bool = False,
        enable_teaching_recording: bool = True,
        enable_rosbag: bool = False,
        enable_topic_rename: bool = True,
        record_depth_topics: bool = False,
        record_thermal_topic: bool = False,
        record_tactile_topics: bool = False,
        release_target_xyz: Optional[np.ndarray] = None,
        release_target_radius: float = 0.08,
        grasp_plane_z: float = 0.0,
        lift_clearance: float = 0.02,
        placed_stable_min_frames: int = 5,
        placed_stable_max_lin_vel: float = 0.03,
        placed_stable_max_tilt_deg: float = 25.0,
        placed_plane_tolerance: float = 0.015,
        task_snapshot_provider: Optional[Callable[[int], dict]] = None,
    ):
        self.env = env
        self._base_body_id = _find_robot_base_body_id(self.env.robot.model)
        self._base_body_name = None
        if self._base_body_id >= 0:
            self._base_body_name = mujoco.mj_id2name(
                self.env.robot.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self._base_body_id,
            )
        self.target_queue = Queue()
        self.is_running = False
        self.control_thread = None
        self._reset_pause_event = threading.Event()

        # 控制参数
        self.control_frequency = 500  # Hz (高频率控制+Joint State发布)
        self.dt = 1.0 / self.control_frequency

        # ROS2 spin 优化（用于高频发布）
        self.ros2_spin_counter = 0  # spin计数器

        # ROS控制模式
        self.enable_ros_control = enable_ros_control and ros2_available()
        self.ros_subscriber = None
        self.ros_target_joints = None
        self.control_mode = "ros_direct"
        self.ros_target_lock = threading.Lock()  # 默认ROS2直接控制模式

        # 夹爪抓取状态发布
        # gripper_command > closing_threshold 时认为夹爪在关闭
        # 且 EE 到最近可抓取物体 < grasp_distance_threshold 时认为已抓住
        self.grasp_distance_threshold = 0.05   # m
        self.gripper_closing_threshold = 128.0
        self.grasp_contact_distance_threshold = 0.003  # m，接触距离阈值（<=该值视为有效接触）
        self.grasp_normal_force_threshold = 0.5  # N，接触法向力阈值
        # 固定放置目标点（XYZ）与阈值
        self.release_target_xyz = np.array(
            release_target_xyz if release_target_xyz is not None else [0.50, 0.00, 0.15],
            dtype=np.float64
        )
        self.release_target_radius = float(release_target_radius)  # m

        # 抬离抓取平面判定参数
        self.grasp_plane_z = float(grasp_plane_z)   # m
        self.lift_clearance = float(lift_clearance) # m

        # 放置稳定判定参数（用于“放置后无滚动翻倒”）
        self.placed_stable_min_frames = max(1, int(placed_stable_min_frames))
        self.placed_stable_max_lin_vel = float(placed_stable_max_lin_vel)
        self.placed_stable_max_tilt_deg = float(placed_stable_max_tilt_deg)
        self.placed_plane_tolerance = float(placed_plane_tolerance)
        self.task_snapshot_provider = task_snapshot_provider
        self._placed_stable_counter = 0
        self._tracked_object_prev_pos = None
        self._tracked_object_prev_t = None

        self._gripper_status_pub_node = None
        self._gripper_status_pub = None
        self._grasp_dist_pub = None
        self._task_snapshot_pub_info = None
        self._graspable_body_ids = []   # 在 start() 中扫描
        self._tracked_object_id = None
        self._body_half_height_cache = {}
        self._gripper_body_ids = []     # 在 start() 中扫描
        self._left_finger_body_ids = []
        self._right_finger_body_ids = []
        # 状态编码器：统一管理 bitmask 的触发时机（含 drop_event 边沿锁存）
        self._gripper_status_encoder = GripperStatusEncoder(drop_latch_frames=3)
        self._ee_pose_pub_info = None

        # Teaching 和记录功能
        self.enable_teaching_recording = enable_teaching_recording and teaching_recording_available
        self.teaching_subscriber = None
        self.current_teaching_status = None
        self.pending_reset_after_teaching_end = False

        # Rosbag 录制功能
        self.enable_rosbag = enable_rosbag and rosbag_recorder_available
        self.enable_topic_rename = bool(enable_topic_rename)
        self.record_depth_topics = bool(record_depth_topics)
        self.record_thermal_topic = bool(record_thermal_topic)
        self.record_tactile_topics = bool(record_tactile_topics)
        self.rosbag_recorder = None
        self.rosbag_executor = None  # 保存 executor 引用以便清理
        self.rosbag_publisher = None  # 用于发布 /teaching_status 话题给 rosbag recorder
        # 按键检测示教
        self.keyboard_teaching_enabled = True
        self.last_key_time = 0

        # 相机图像缓存（避免OpenGL冲突）
        self.latest_camera_image = None
        self.camera_image_lock = threading.Lock()
        self.camera_thread = None
        self.camera_thread_running = False

        # Cartesian Target / RRT / IK 链路已移除

        if self.enable_ros_control:
            try:
                import time
                # 生成唯一节点名避免冲突
                node_name = f"arm_controller_subscriber_{int(time.time() * 1000) % 10000}"
                self.ros_subscriber = JointStateSubscriberROS2(
                    topic="/joint_target",
                    node_name=node_name,
                    callback=self._ros_joint_callback
                )
                print(f"✓ ROS2 Joint State Subscriber initialized (node: {node_name})")
            except Exception as e:
                print(f"⚠ Failed to initialize ROS2 subscriber: {e}")
                self.enable_ros_control = False

        # 初始化夹爪状态发布器（/gripper_status, /grasp_distance）
        if ros2_available():
            try:
                import rclpy
                from rclpy.node import Node
                from std_msgs.msg import Float64 as Float64Msg, UInt32 as UInt32Msg

                _pub_ctx = rclpy.Context()
                rclpy.init(context=_pub_ctx)
                _pub_node_name = f"gripper_status_publisher_{int(time.time() * 1000) % 10000}"
                _pub_node = Node(_pub_node_name, context=_pub_ctx)
                self._gripper_status_pub = _pub_node.create_publisher(UInt32Msg, '/gripper_status', 10)
                self._grasp_dist_pub = _pub_node.create_publisher(Float64Msg, '/grasp_distance', 10)
                self._gripper_status_pub_node = _pub_node
                print(f"✓ 夹爪状态发布器已启动 (节点: {_pub_node_name})")
                print(f"  - /gripper_status  (std_msgs/UInt32): bitmask状态")
                print(f"    * bit0(IDLE): 当前状态为 idle 时置位")
                print(f"    * bit1(GRASPING): 当前状态为 grasping 时置位")
                print(f"    * bit2(GRASPED): 当前状态为 grasped 时置位")
                print(f"    * bit3(DROP_EVENT): grasped->非grasped 边沿触发并短时锁存")
                print(f"    * bit4(COLLISION): 当前检测到碰撞时置位")
                print(f"    * bit5(JOINT_LIMIT): 当前检测到关节越限时置位")
                print(f"    * bit6(PREMATURE_DROP_EVENT): 未到目标区提前掉落时置位")
                print(f"    * bit7(AT_TARGET_ZONE): 物体进入固定目标区时置位")
                print(f"    * bit8(LIFTED_FROM_PLANE): 物体完全抬离抓取平面时置位")
                print(f"    * bit9(PLACED_STABLE): 物体放置后稳定且无滚动翻倒时置位")
                print(f"  - /grasp_distance  (std_msgs/Float64): EE到最近物体距离 (m)")
            except Exception as e:
                print(f"⚠ 夹爪状态发布器初始化失败: {e}")

        # 初始化任务场景快照发布器（裁判离线评估使用）
        if ros2_available():
            try:
                import rclpy
                from rclpy.node import Node
                from std_msgs.msg import String as StringMsg

                _snapshot_pub_ctx = rclpy.Context()
                rclpy.init(context=_snapshot_pub_ctx)
                _snapshot_pub_node_name = f"task_snapshot_publisher_{int(time.time() * 1000) % 10000}"
                _snapshot_pub_node = Node(_snapshot_pub_node_name, context=_snapshot_pub_ctx)
                _snapshot_pub = _snapshot_pub_node.create_publisher(StringMsg, '/task_scene_snapshot', 10)
                self._task_snapshot_pub_info = {
                    'context': _snapshot_pub_ctx,
                    'node': _snapshot_pub_node,
                    'publisher': _snapshot_pub,
                }
                print(f"✓ 任务场景快照发布器已启动 (节点: {_snapshot_pub_node_name})")
                print("  - /task_scene_snapshot (std_msgs/String): JSON 场景快照，用于离线评分")
            except Exception as e:
                print(f"⚠ 任务场景快照发布器初始化失败: {e}")

        # 初始化末端位姿发布器（/ee_pose）
        if ros2_available():
            try:
                import rclpy
                from rclpy.node import Node
                from geometry_msgs.msg import PoseStamped
                from std_msgs.msg import Float64MultiArray

                _ee_pub_ctx = rclpy.Context()
                rclpy.init(context=_ee_pub_ctx)
                _ee_pub_node_name = f"ee_pose_publisher_{int(time.time() * 1000) % 10000}"
                _ee_pub_node = Node(_ee_pub_node_name, context=_ee_pub_ctx)
                _ee_pose_pub = _ee_pub_node.create_publisher(PoseStamped, '/ee_pose', 10)
                _ee_pose_gripper_pub = _ee_pub_node.create_publisher(Float64MultiArray, '/ee_pose_gripper', 10)
                self._ee_pose_pub_info = {
                    'context': _ee_pub_ctx,
                    'node': _ee_pub_node,
                    'publisher': _ee_pose_pub,
                    'pose_gripper_publisher': _ee_pose_gripper_pub,
                }
                print(f"✓ EE pose publisher initialized (node: {_ee_pub_node_name})")
                print(f"  - /ee_pose (geometry_msgs/PoseStamped, base_link frame)")
                print(f"  - /ee_pose_gripper (std_msgs/Float64MultiArray: [x, y, z, rot6d(6), gripper])")
            except Exception as e:
                print(f"⚠ EE pose publisher init failed: {e}")

        # 初始化 Teaching 功能
        if self.enable_teaching_recording:
            try:
                # 仅保留 /teaching_status 订阅，用于触发 MCAP 录制。
                if ros2_available():
                    teaching_node_name = f"teaching_status_subscriber_{int(time.time() * 1000) % 10000}"
                    self.teaching_subscriber = TeachingStatusSubscriber(
                        topic="/teaching_status",
                        node_name=teaching_node_name,
                        callback=self._teaching_status_callback
                    )
                    print(f"✓ ROS2 Teaching Status Subscriber: {teaching_node_name}")

                print(f"✓ Teaching Status System initialized")
                print(f"  - 控制话题: /teaching_status")
                print(f"  - 录制方式: 仅触发 MCAP 录制")
                print(f"  - 本地 data/ 视频、JSON、NPZ 录制: 已禁用")
            except Exception as e:
                print(f"⚠ Failed to initialize Teaching Recording: {e}")
                self.enable_teaching_recording = False

        # 初始化 Rosbag 录制器
        if self.enable_rosbag:
            try:
                # rosbag 输出目录默认使用“项目根目录/rosbag_data”，避免写死绝对路径
                default_rosbag_dir = str(PROJECT_ROOT / "rosbag_data")
                default_archive_dir = str(PROJECT_ROOT / "rosbag_archive")
                # 可通过环境变量覆盖输出目录
                rosbag_output_dir = os.environ.get('ARM_PROJECT_ROSBAG_DIR', default_rosbag_dir)
                archive_output_dir = os.environ.get('ARM_PROJECT_ARCHIVE_DIR', default_archive_dir)
                topic_rename_map = {
                    '/joint_states_sim': '/joint_states',
                    '/joint_target': '/joint_cmd'
                } if self.enable_topic_rename else {}
                rosbag_topics = [
                    '/ee_camera/camera_info',
                    '/ee_camera/rgb/image_raw',
                    '/ee_pose',
                    '/ee_pose_gripper',
                    '/ee_target',
                    '/external_camera/camera_info',
                    '/external_camera/rgb/image_raw',
                    '/grasp_distance',
                    '/joint_states_R',
                    '/joint_states_sim',
                    '/joint_target',
                    '/parameter_events',
                    '/rosout',
                    '/gripper_status',
                    '/task_scene_snapshot',
                    '/teaching_status',
                ]
                if self.record_depth_topics:
                    rosbag_topics.extend([
                        '/ee_camera/depth/image_raw',
                        '/external_camera/depth/image_raw',
                    ])
                if self.record_thermal_topic:
                    rosbag_topics.append('/thermal_camera/image')
                if self.record_tactile_topics:
                    rosbag_topics.extend([
                        '/gripper_tactile/left/vector',
                        '/gripper_tactile/right/vector',
                        '/gripper_tactile/left/tangential',
                        '/gripper_tactile/right/tangential',
                        '/gripper_tactile/left/pad',
                        '/gripper_tactile/right/pad',
                    ])

                # 创建 RosbagRecorder 实例（在独立线程中运行）
                self.rosbag_recorder = RosbagRecorder(
                    output_dir=rosbag_output_dir,
                    topics=rosbag_topics,
                    storage_format="mcap",
                    compression_mode="message",  # 启用压缩
                    compression_format="zstd",
                    topic_rename_map=topic_rename_map,
                    auto_export_archive=True,
                    archive_output_dir=archive_output_dir,
                    archive_video_format="mp4_x264_lossless",
                    record_depth_topics=self.record_depth_topics,
                    record_thermal_topic=self.record_thermal_topic,
                    record_tactile_topics=self.record_tactile_topics,
                )

                # 创建 ROS2 publisher 用于向 rosbag recorder 发送控制命令
                if ros2_available():
                    try:
                        import rclpy
                        from std_msgs.msg import String

                        # 创建一个临时节点用于发布 teaching_status
                        teaching_pub_node_name = f"teaching_status_publisher_{int(time.time() * 1000) % 10000}"

                        # 注意: 这里需要一个简单的publisher，不需要完整的节点
                        # 我们将在 start() 中初始化它
                        self.rosbag_publisher_node_name = teaching_pub_node_name

                    except Exception as e:
                        print(f"⚠ 无法创建 teaching_status publisher: {e}")

                print(f"✓ Rosbag Recording System initialized")
                print(f"  - 输出目录: {rosbag_output_dir}")
                print(f"  - 自动归档目录: {archive_output_dir}")
                print(f"  - 压缩模式: message (zstd)")
                print(f"  - 图像录制: 原生 /rgb/image_raw topic")
                print(f"  - 深度录制: {'启用' if self.record_depth_topics else '禁用'}")
                print(f"  - 红外录制: {'启用' if self.record_thermal_topic else '禁用'}")
                print(f"  - 触觉录制: {'启用' if self.record_tactile_topics else '禁用'}")
                print(f"  - 停止录制后自动导出: mp4_x264_lossless")
                print(f"  - 话题重命名: {'启用' if self.enable_topic_rename else '禁用'}")
                print(f"  - 按键控制: 'b' 开始/停止 rosbag 录制")

            except Exception as e:
                print(f"⚠ Failed to initialize Rosbag Recorder: {e}")
                import traceback
                traceback.print_exc()
                self.enable_rosbag = False

        print(f"✓ 控制器初始化 - 频率: {self.control_frequency}Hz, 模式: 直接关节控制")
        if self.enable_ros_control:
            print(f"✓ ROS2控制模式可用 - 订阅话题: /joint_target")
        if self.enable_teaching_recording:
            print(f"✓ Teaching记录模式可用 - 订阅话题: /teaching_status")
        
    def add_target(self, target_type: str = "random", target_pos: Optional[np.ndarray] = None):
        """轨迹目标队列已停用，保留接口仅用于兼容。"""
        print("⚠ 当前版本已不支持轨迹目标队列，请直接通过 /joint_target 控制。")
        
    def _ros_joint_callback(self, joint_positions: np.ndarray):
        """ROS joint state callback function."""
        try:
            if hasattr(self, '_callback_count'):
                self._callback_count += 1
            else:
                self._callback_count = 1

            # /joint_target 方向映射：将 J3、J4 取反（1-based: J3/J4 -> index 2/3）
            remapped_joints = joint_positions.copy()
            if len(remapped_joints) >= 5:
                remapped_joints[2] = -remapped_joints[2]
                remapped_joints[3] = -remapped_joints[3]
                remapped_joints[5] = -remapped_joints[5]

            
            with self.ros_target_lock:
                if hasattr(self, '_last_ros_joints') and self._last_ros_joints is not None:
                    if np.allclose(remapped_joints, self._last_ros_joints, atol=1e-6):
                        return
                
                self.ros_target_joints = remapped_joints
                self._last_ros_joints = remapped_joints.copy()

                self.control_mode = "ros_direct"

            pass
                
        except Exception as e:
            print(f"❌ ROS2回调处理错误: {e}")

    def _teaching_status_callback(self, status: str):
        """Teaching status callback function."""
        try:
            self.current_teaching_status = status

            if status == "start_teaching":
                print("📚 开始Teaching - 等待 MCAP 录制")
                if self.rosbag_recorder is not None:
                    self._publish_teaching_status("start_teaching")
                    self._write_initial_scene_when_recording_ready()
                else:
                    print("❌ rosbag recorder 未初始化，无法开始录制")

            elif status == "end_teaching":
                print("📚 结束Teaching - 停止 MCAP 录制")
                if self.rosbag_recorder is not None:
                    self._publish_teaching_status("end_teaching")
                else:
                    print("❌ rosbag recorder 未初始化，无法停止录制")
                self.pending_reset_after_teaching_end = True
                print("↻ 已收到 end_teaching，主循环将执行一次安全 reset")

        except Exception as e:
            print(f"Teaching status callback error: {e}")

    def _publish_teaching_status(self, status: str):
        """发布 teaching_status 消息到 RosbagRecorder"""
        try:
            if not ros2_available():
                print("⚠ ROS2 不可用，无法发布 teaching_status")
                return

            # 直接调用 rosbag_recorder 的回调
            if self.rosbag_recorder:
                from std_msgs.msg import String
                msg = String()
                msg.data = status
                self.rosbag_recorder.teaching_status_callback(msg)
                print(f"📤 发送 teaching_status: {status}")

        except Exception as e:
            print(f"❌ 发布 teaching_status 失败: {e}")
            import traceback
            traceback.print_exc()


    def set_control_mode(self, mode: str):
        """
        设置控制模式
        
        Args:
            mode: 仅支持 "ros_direct"
        """
        if mode != "ros_direct":
            print("⚠ 当前版本已移除轨迹 / 插值控制，仅保留 ros_direct 模式")
            return False

        if not self.enable_ros_control:
            print("⚠ ROS控制模式不可用，请检查ROS2环境")
            return False

        self.control_mode = "ros_direct"
        print("✓ 切换到ROS直接控制模式 - 等待 /joint_target 消息（支持7个关节：6个机械臂关节 + 1个gripper关节）")
        return True

    def get_control_mode(self) -> str:
        """获取当前控制模式"""
        return self.control_mode

    def is_ros_control_available(self) -> bool:
        """检查ROS控制是否可用"""
        return self.enable_ros_control
        
    def set_trajectory_duration(self, duration: float):
        """轨迹执行时间设置已停用。"""
        print("⚠ 当前版本已移除插值轨迹执行，set_trajectory_duration 不再生效。")
        
    def generate_trajectory_to_target(self, target_pos: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """轨迹生成接口已停用。"""
        print("⚠ 当前版本已移除轨迹生成与插值功能，请直接发布 /joint_target。")
        return None, None
        
    def control_loop(self):
        """主控制循环 - 仅保留 ROS 直接关节控制。"""
        loop_count = 0
        while self.is_running:
            loop_start_time = time.time()
            loop_count += 1

            try:
                if self._reset_pause_event.is_set():
                    time.sleep(self.dt)
                    continue

                if self.control_mode == "ros_direct":
                    with self.ros_target_lock:
                        target_joints_copy = self.ros_target_joints.copy() if self.ros_target_joints is not None else None

                    if target_joints_copy is not None:
                        self.env.robot.update_joint_state()

                        if len(target_joints_copy) >= 7:
                            self.env.robot.apply_joint_control(target_joints_copy[:6])
                            gripper_scaled = target_joints_copy[6] * 255.0
                            self.env.robot.set_gripper_command(gripper_scaled)
                        else:
                            self.env.robot.apply_joint_control(target_joints_copy)

                    else:
                        self.env.robot.update_joint_state()
                        current_joints = self.env.robot.get_joint_positions()
                        self.env.robot.apply_joint_control(current_joints)
                else:
                    self.env.robot.update_joint_state()
                    current_joints = self.env.robot.get_joint_positions()
                    self.env.robot.apply_joint_control(current_joints)
                
                # 执行一步仿真（这里会应用PID力矩控制）
                self.env.robot.update_obstacles()
                self.env.robot.step()  # 应用PID计算的力矩

                # 发布夹爪状态（每 50 步 ≈ 10Hz）
                if loop_count % 50 == 0 and self._gripper_status_pub is not None:
                    try:
                        from std_msgs.msg import Float64 as Float64Msg, UInt32 as UInt32Msg

                        status_str, min_dist, grasped_obj_id = self._estimate_grasp_state()
                        self._update_tracked_object(grasped_obj_id)
                        collision_flag = self._estimate_collision_state()
                        joint_limit_flag = self._estimate_joint_limit_state()
                        at_release_target_flag = self._is_release_target_reached()
                        lifted_from_plane_flag = self._is_object_lifted_from_plane()
                        placed_stable_flag = self._is_object_placed_stable(
                            status_str=status_str,
                            at_release_target_flag=at_release_target_flag,
                            lifted_from_plane_flag=lifted_from_plane_flag,
                        )
                        status_bits = self._build_gripper_status_bits(
                            status_str=status_str,
                            collision_flag=collision_flag,
                            joint_limit_flag=joint_limit_flag,
                            at_release_target_flag=at_release_target_flag,
                            lifted_from_plane_flag=lifted_from_plane_flag,
                            placed_stable_flag=placed_stable_flag,
                        )

                        bm = UInt32Msg()
                        bm.data = int(status_bits)
                        self._gripper_status_pub.publish(bm)

                        dm = Float64Msg()
                        dm.data = min_dist if min_dist != float('inf') else -1.0
                        self._grasp_dist_pub.publish(dm)
                    except Exception:
                        pass

                # 发布任务场景快照（每 50 步 ≈ 10Hz），供离线评估目标盘和温度配对。
                if loop_count % 50 == 0 and self._task_snapshot_pub_info is not None:
                    try:
                        from std_msgs.msg import String as StringMsg

                        snapshot = self._build_task_scene_snapshot(loop_count)
                        msg = StringMsg()
                        msg.data = json.dumps(snapshot, ensure_ascii=False)
                        self._task_snapshot_pub_info['publisher'].publish(msg)
                    except Exception:
                        pass

                # 发布末端位姿（每 10 步 ≈ 50Hz）
                if loop_count % 10 == 0 and self._ee_pose_pub_info is not None:
                    try:
                        from geometry_msgs.msg import PoseStamped
                        from std_msgs.msg import Float64MultiArray

                        ee_pos_world = self.env.robot.get_ee_position()
                        ee_quat_world = self.env.robot.get_ee_orientation()

                        if self._base_body_id >= 0:
                            base_rot_world = self.env.robot.data.xmat[self._base_body_id].reshape(3, 3).copy()
                            base_pos_world = self.env.robot.data.xpos[self._base_body_id].copy()
                            ee_pos = base_rot_world.T @ (ee_pos_world - base_pos_world)

                            from scipy.spatial.transform import Rotation as R
                            base_rot_obj = R.from_matrix(base_rot_world)
                            world_rot_obj = R.from_quat([
                                ee_quat_world[1], ee_quat_world[2], ee_quat_world[3], ee_quat_world[0]
                            ])
                            ee_rot_base = base_rot_obj.inv() * world_rot_obj
                            ee_quat_base_xyzw = ee_rot_base.as_quat()
                            ee_quat = np.array([
                                ee_quat_base_xyzw[3],
                                ee_quat_base_xyzw[0],
                                ee_quat_base_xyzw[1],
                                ee_quat_base_xyzw[2],
                            ])
                        else:
                            ee_pos = ee_pos_world
                            ee_quat = ee_quat_world

                        ee_msg = PoseStamped()
                        ee_msg.header.stamp = self._ee_pose_pub_info['node'].get_clock().now().to_msg()
                        ee_msg.header.frame_id = 'base_link'
                        ee_msg.pose.position.x = float(ee_pos[0])
                        ee_msg.pose.position.y = float(ee_pos[1])
                        ee_msg.pose.position.z = float(ee_pos[2])
                        ee_msg.pose.orientation.w = float(ee_quat[0])
                        ee_msg.pose.orientation.x = float(ee_quat[1])
                        ee_msg.pose.orientation.y = float(ee_quat[2])
                        ee_msg.pose.orientation.z = float(ee_quat[3])
                        self._ee_pose_pub_info['publisher'].publish(ee_msg)

                        # 发布数据集所需的 10 维末端状态：
                        # [x, y, z, rot6d(6), gripper]
                        #
                        # rot6d 使用旋转矩阵前两列展开：
                        #   rot6d = [R[:,0], R[:,1]]
                        # 这是常见的连续旋转表示，可避免四元数双覆盖问题。
                        from scipy.spatial.transform import Rotation as R
                        ee_rotmat = R.from_quat([
                            ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]
                        ]).as_matrix()
                        rot6d = np.concatenate([ee_rotmat[:, 0], ee_rotmat[:, 1]], axis=0)

                        pose_gripper_msg = Float64MultiArray()
                        pose_gripper_msg.data = [
                            float(ee_pos[0]),
                            float(ee_pos[1]),
                            float(ee_pos[2]),
                            float(rot6d[0]),
                            float(rot6d[1]),
                            float(rot6d[2]),
                            float(rot6d[3]),
                            float(rot6d[4]),
                            float(rot6d[5]),
                            float(self.env.robot.gripper_command / 255.0),
                        ]
                        self._ee_pose_pub_info['pose_gripper_publisher'].publish(pose_gripper_msg)
                    except Exception:
                        pass

                # V4: 相机渲染移到显示循环中，避免重复渲染导致 OpenGL 冲突
                # 控制循环只负责机械臂控制，不渲染相机
                
                # 发布ROS2 Joint State (如果启用) - 500Hz高频率发布
                if self.env.robot.enable_joint_state_ros2 and self.env.robot.joint_state_pub is not None:
                    try:
                        # 每10次循环才调用spin_once以提高性能 (500Hz/10 = 50Hz spin频率)
                        skip_spin = (self.ros2_spin_counter % 10) != 0
                        self.env.robot.publish_joint_state_ros2(
                            stamp_sec=time.time(), 
                            skip_spin=skip_spin
                        )
                        self.ros2_spin_counter += 1
                    except Exception as e:
                        print(f"⚠ Joint State发布失败: {e}")
                # 同步可视化
                if self.env.viewer is not None:
                    self.env.viewer.sync()
                
                # 控制循环频率
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, self.dt - loop_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Control loop error: {e}")
                break

        print("Control loop ended")

    def _estimate_grasp_state(self):
        """估计当前抓取状态。

        规则：
            1) 夹爪未闭合 -> idle
            2) 左右手指均与同一可抓取物体接触，且接触法向力超过阈值 -> grasped
            3) 夹爪闭合但不满足2 -> grasping
        """
        gripper_cmd = self.env.robot.gripper_command
        is_grasping = gripper_cmd > self.gripper_closing_threshold

        # 计算 EE 到最近可抓取物体距离
        min_dist = float('inf')
        if self._graspable_body_ids:
            ee_pos = self.env.robot.get_ee_position()
            for bid in self._graspable_body_ids:
                obj_pos = self.env.robot.data.xpos[bid]
                d = float(np.linalg.norm(ee_pos - obj_pos))
                if d < min_dist:
                    min_dist = d

        # 接触检测：左右手指分别接触到“同一个”可抓取物体，且法向接触力达阈值
        has_two_side_contact = False
        grasped_obj_id = None
        if self._left_finger_body_ids and self._right_finger_body_ids and self._graspable_body_ids:
            graspable_set = set(self._graspable_body_ids)
            left_set = set(self._left_finger_body_ids)
            right_set = set(self._right_finger_body_ids)

            left_contact_objs = set()
            right_contact_objs = set()

            for i in range(self.env.robot.data.ncon):
                c = self.env.robot.data.contact[i]
                if c.dist > self.grasp_contact_distance_threshold:
                    continue

                b1 = int(self.env.robot.model.geom_bodyid[c.geom1])
                b2 = int(self.env.robot.model.geom_bodyid[c.geom2])

                # 读取接触力（contact frame）: force_torque[0] 是法向分量
                force_torque = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.env.robot.model, self.env.robot.data, i, force_torque)
                normal_force = abs(float(force_torque[0]))
                if normal_force < self.grasp_normal_force_threshold:
                    continue

                # 左指-物体
                if b1 in left_set and b2 in graspable_set:
                    left_contact_objs.add(b2)
                elif b2 in left_set and b1 in graspable_set:
                    left_contact_objs.add(b1)

                # 右指-物体
                if b1 in right_set and b2 in graspable_set:
                    right_contact_objs.add(b2)
                elif b2 in right_set and b1 in graspable_set:
                    right_contact_objs.add(b1)

            common_objs = list(left_contact_objs & right_contact_objs)
            has_two_side_contact = len(common_objs) > 0
            if has_two_side_contact:
                if len(common_objs) == 1:
                    grasped_obj_id = int(common_objs[0])
                else:
                    ee_pos = self.env.robot.get_ee_position()
                    grasped_obj_id = int(min(
                        common_objs,
                        key=lambda bid: float(np.linalg.norm(ee_pos - self.env.robot.data.xpos[bid]))
                    ))

        has_grasped = is_grasping and has_two_side_contact

        if has_grasped:
            return "grasped", min_dist, grasped_obj_id
        if is_grasping:
            return "grasping", min_dist, None
        return "idle", min_dist, None

    def _estimate_collision_state(self):
        """估计碰撞状态（与障碍物/环境接触时置位）。"""
        try:
            if hasattr(self.env.robot, 'check_obstacle_collision'):
                info = self.env.robot.check_obstacle_collision()
                return bool(info.get('collision', False))
        except Exception:
            pass
        return False

    def _estimate_joint_limit_state(self):
        """估计关节越限状态。"""
        try:
            q = np.asarray(self.env.robot.get_joint_positions(), dtype=np.float64)
            if hasattr(self.env, 'joint_limits'):
                limits = np.asarray(self.env.joint_limits, dtype=np.float64)
                low = limits[:, 0] - 1e-3
                high = limits[:, 1] + 1e-3
                return bool(np.any((q < low) | (q > high)))
        except Exception:
            pass
        return False

    def _update_tracked_object(self, grasped_obj_id: Optional[int]):
        """更新当前被操作物体 ID。"""
        if grasped_obj_id is not None:
            new_id = int(grasped_obj_id)
            if self._tracked_object_id != new_id:
                self._placed_stable_counter = 0
                self._tracked_object_prev_pos = None
                self._tracked_object_prev_t = None
            self._tracked_object_id = new_id

    def _estimate_body_half_height(self, body_id: int) -> float:
        """估计物体半高（用于底部高度判定）。"""
        if body_id in self._body_half_height_cache:
            return self._body_half_height_cache[body_id]
        geom_ids = np.where(self.env.robot.model.geom_bodyid == int(body_id))[0]
        half_h = 0.0
        for gid in geom_ids:
            size = self.env.robot.model.geom_size[gid]
            half_h = max(half_h, float(np.max(size)))
        self._body_half_height_cache[body_id] = half_h
        return half_h

    def _is_release_target_reached(self):
        """判定物体是否进入固定目标区。"""
        if self._tracked_object_id is None:
            return False
        try:
            obj_pos = self.env.robot.data.xpos[int(self._tracked_object_id)]
            d = float(np.linalg.norm(obj_pos - self.release_target_xyz))
            return d <= self.release_target_radius
        except Exception:
            return False

    def _is_object_lifted_from_plane(self):
        """判定物体是否完全抬离抓取平面。"""
        if self._tracked_object_id is None:
            return False
        try:
            body_id = int(self._tracked_object_id)
            obj_pos = self.env.robot.data.xpos[body_id]
            half_h = self._estimate_body_half_height(body_id)
            # 底部高度公式：z_bottom = z_obj - h_half
            # 完全抬离判定：z_bottom >= z_plane + lift_clearance
            z_bottom = float(obj_pos[2] - half_h)
            return z_bottom >= float(self.grasp_plane_z + self.lift_clearance)
        except Exception:
            return False

    def _estimate_object_tilt_deg(self, body_id: int) -> float:
        """估计物体局部 z 轴相对世界竖直方向的倾角（度）。"""
        xmat = self.env.robot.data.xmat[int(body_id)].reshape(3, 3)
        local_z_in_world = xmat[:, 2]
        cos_theta = float(np.clip(np.dot(local_z_in_world, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        theta_rad = np.arccos(cos_theta)
        # 倾角公式：theta = arccos( z_local · z_world )
        return float(np.degrees(theta_rad))

    def _is_object_placed_stable(self, status_str: str, at_release_target_flag: bool, lifted_from_plane_flag: bool):
        """判定物体是否“放置后稳定且无滚动翻倒”。"""
        if self._tracked_object_id is None:
            self._placed_stable_counter = 0
            return False

        try:
            body_id = int(self._tracked_object_id)
            obj_pos = np.asarray(self.env.robot.data.xpos[body_id], dtype=np.float64)
            now_t = float(time.time())

            if (self._tracked_object_prev_pos is None) or (self._tracked_object_prev_t is None):
                lin_vel = float('inf')
            else:
                dt = max(1e-6, now_t - float(self._tracked_object_prev_t))
                lin_vel = float(np.linalg.norm(obj_pos - self._tracked_object_prev_pos) / dt)

            self._tracked_object_prev_pos = obj_pos.copy()
            self._tracked_object_prev_t = now_t

            half_h = self._estimate_body_half_height(body_id)
            z_bottom = float(obj_pos[2] - half_h)
            tilt_deg = self._estimate_object_tilt_deg(body_id)

            # 放置稳定条件：
            # 1) 已到目标区；2) 已释放（不在 grasped）；3) 物体回到桌面附近；
            # 4) 线速度低于阈值；5) 倾角低于阈值（无明显翻倒）。
            on_plane = abs(z_bottom - float(self.grasp_plane_z)) <= self.placed_plane_tolerance
            released = (status_str != 'grasped')
            low_speed = lin_vel <= self.placed_stable_max_lin_vel
            low_tilt = tilt_deg <= self.placed_stable_max_tilt_deg
            not_lifted = (not lifted_from_plane_flag)

            is_stable_now = bool(
                at_release_target_flag and released and on_plane and not_lifted and low_speed and low_tilt
            )
            if is_stable_now:
                self._placed_stable_counter += 1
            else:
                self._placed_stable_counter = 0

            return self._placed_stable_counter >= self.placed_stable_min_frames
        except Exception:
            self._placed_stable_counter = 0
            return False

    def _build_gripper_status_bits(
        self,
        status_str: str,
        collision_flag: bool,
        joint_limit_flag: bool,
        at_release_target_flag: bool,
        lifted_from_plane_flag: bool,
        placed_stable_flag: bool,
    ) -> int:
        """构建 /gripper_status 的 UInt32 bitmask。

        触发时机：
            1) idle/grasping/grasped：每次发布都更新（互斥）
            2) drop_event：grasped -> 非grasped 边沿触发，并锁存若干发布周期
            3) premature_drop_event：drop_event 且未到目标区时置位
            4) at_target_zone/lifted_from_plane/placed_stable：当前成立时置位（电平信号）
            5) collision/joint_limit：当前成立时置位（电平信号）
        """
        return self._gripper_status_encoder.encode(
            status_str=status_str,
            collision_flag=collision_flag,
            joint_limit_flag=joint_limit_flag,
            at_release_target_flag=at_release_target_flag,
            lifted_from_plane_flag=lifted_from_plane_flag,
            placed_stable_flag=placed_stable_flag,
        )

    def _build_task_scene_snapshot(self, frame_idx: int = 0) -> dict:
        """构建任务场景快照，优先使用外部 provider 以保留温度信息。"""
        if self.task_snapshot_provider is not None:
            return self.task_snapshot_provider(int(frame_idx))

        def collect_body(body_name: str) -> dict:
            body_id = mujoco.mj_name2id(self.env.robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                return {
                    "body_name": body_name,
                    "exists": False,
                    "temperature_c": None,
                    "position_xyz": None,
                }
            pos = self.env.robot.data.xpos[body_id]
            return {
                "body_name": body_name,
                "exists": True,
                "temperature_c": None,
                "position_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
            }

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame": int(frame_idx),
            "movable_objects": [
                collect_body("beaker1"),
                collect_body("graduated_cylinder"),
                collect_body("erlenmeyer_flask"),
            ],
            "target_plates": [
                collect_body("target_place_table_1"),
                collect_body("target_place_table_2"),
                collect_body("target_place_table_3"),
            ],
        }
        
    def start(self):
        """启动连续控制"""
        if self.is_running:
            print("Controller already running")
            return

        self.is_running = True
        self._gripper_status_encoder.reset()

        # 扫描场景中的可抓取物体（排除机器人、地板、桌面、夹爪自身等）
        gripper_kw = ['gripper', 'finger', 'pad', 'coupler', 'follower', 'driver', 'robotiq']
        exclude_kw = ['world', 'base', 'link', 'floor', 'table', 'bench',
                      'wall', 'obstacle', 'target', 'gripper', 'camera',
                      'light', 'site']
        self._graspable_body_ids = []
        self._gripper_body_ids = []
        self._left_finger_body_ids = []
        self._right_finger_body_ids = []
        try:
            for i in range(self.env.robot.model.nbody):
                name = self.env.robot.model.body(i).name.lower()
                if not name:
                    continue
                if any(kw in name for kw in exclude_kw):
                    continue
                if any(kw in name for kw in gripper_kw):
                    continue

                # 仅把 freejoint 的可动物体作为可抓取候选
                body_jntnum = int(self.env.robot.model.body_jntnum[i])
                if body_jntnum <= 0:
                    continue
                jnt_adr = int(self.env.robot.model.body_jntadr[i])
                if int(self.env.robot.model.jnt_type[jnt_adr]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    continue

                if self.env.robot.model.body_mass[i] > 0.01:
                    self._graspable_body_ids.append(i)

            # 扫描夹爪/手指相关 body
            for i in range(self.env.robot.model.nbody):
                name = self.env.robot.model.body(i).name.lower()
                if not name:
                    continue
                if any(kw in name for kw in gripper_kw):
                    self._gripper_body_ids.append(i)
                    if ('left' in name) and any(k in name for k in ['finger', 'pad', 'follower', 'coupler', 'driver']):
                        self._left_finger_body_ids.append(i)
                    if ('right' in name) and any(k in name for k in ['finger', 'pad', 'follower', 'coupler', 'driver']):
                        self._right_finger_body_ids.append(i)
        except Exception as e:
            print(f"⚠ 可抓取物体扫描失败: {e}")
        print(f"✓ 固定目标点: {self.release_target_xyz.tolist()}, 半径={self.release_target_radius:.3f}m")
        print(f"✓ 抬离平面参数: z_plane={self.grasp_plane_z:.3f}m, clearance={self.lift_clearance:.3f}m")
        print(
            f"✓ 放置稳定参数: min_frames={self.placed_stable_min_frames}, "
            f"max_lin_vel={self.placed_stable_max_lin_vel:.3f}m/s, "
            f"max_tilt={self.placed_stable_max_tilt_deg:.1f}deg, "
            f"plane_tol={self.placed_plane_tolerance:.3f}m"
        )

        # 启动时主动发布一次 gripper 状态，避免话题初始为空
        if self._gripper_status_pub is not None:
            try:
                from std_msgs.msg import Float64 as Float64Msg, UInt32 as UInt32Msg
                init_bits = set_status_bit(0, GRIPPER_STATUS_IDLE_BIT)
                bm = UInt32Msg()
                bm.data = int(init_bits)
                self._gripper_status_pub.publish(bm)

                if self._grasp_dist_pub is not None:
                    dm = Float64Msg()
                    dm.data = -1.0
                    self._grasp_dist_pub.publish(dm)
            except Exception:
                pass


        if self.ros_subscriber is not None:
            self.ros_subscriber.start_spinning()

        # 启动Teaching订阅器
        if self.teaching_subscriber is not None:
            self.teaching_subscriber.start_spinning()

        # 启动 Rosbag 录制器（使用 MultiThreadedExecutor 在独立线程中运行）
        if self.rosbag_recorder is not None:
            try:
                import rclpy
                from rclpy.executors import MultiThreadedExecutor

                def run_rosbag_recorder():
                    """在独立线程中运行 rosbag recorder"""
                    try:
                        # 创建独立的 MultiThreadedExecutor
                        self.rosbag_executor = MultiThreadedExecutor(num_threads=2)
                        self.rosbag_executor.add_node(self.rosbag_recorder)

                        # 使用 executor.spin() 而不是 rclpy.spin()
                        self.rosbag_executor.spin()
                    except Exception as e:
                        print(f"Rosbag recorder thread error: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # 清理 executor
                        if self.rosbag_executor is not None:
                            self.rosbag_executor.shutdown()
                            self.rosbag_executor = None

                self.rosbag_thread = threading.Thread(target=run_rosbag_recorder, daemon=True)
                self.rosbag_thread.start()

            except Exception as e:
                print(f"Rosbag recorder startup failed: {e}")
                import traceback
                traceback.print_exc()

        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        
    def stop(self):
        """停止连续控制"""
        print("Stopping controller...")
        self.is_running = False

        # 停止ROS订阅器
        if self.ros_subscriber is not None:
            try:
                self.ros_subscriber.stop_spinning()
                time.sleep(0.1)
            except Exception as e:
                print(f"ROS2 subscriber stop error: {e}")

        # 清理夹爪状态发布器
        if self._gripper_status_pub_node is not None:
            try:
                self._gripper_status_pub_node.destroy_node()
            except Exception:
                pass

        # 清理任务场景快照发布器
        if self._task_snapshot_pub_info is not None:
            try:
                self._task_snapshot_pub_info['node'].destroy_node()
                self._task_snapshot_pub_info['context'].shutdown()
            except Exception:
                pass

        # 清理末端位姿发布器
        if self._ee_pose_pub_info is not None:
            try:
                self._ee_pose_pub_info['node'].destroy_node()
                self._ee_pose_pub_info['context'].shutdown()
            except Exception:
                pass

        # 停止Teaching订阅器
        if self.teaching_subscriber is not None:
            try:
                self.teaching_subscriber.stop_spinning()
                time.sleep(0.1)
            except Exception as e:
                print(f"Teaching subscriber stop error: {e}")

        # 停止 Rosbag 录制器
        if self.rosbag_recorder is not None:
            try:
                if self.rosbag_recorder.is_recording:
                    self._publish_teaching_status("end_teaching")
                    time.sleep(0.5)

                if self.rosbag_executor is not None:
                    self.rosbag_executor.shutdown()

                if hasattr(self, 'rosbag_thread') and self.rosbag_thread is not None:
                    self.rosbag_thread.join(timeout=5.0)
                    if self.rosbag_thread.is_alive():
                        print("Rosbag thread did not exit within timeout")

                self.rosbag_recorder.shutdown()

            except Exception as e:
                print(f"Rosbag recorder stop error: {e}")
                import traceback
                traceback.print_exc()

        # 停止控制线程
        if self.control_thread is not None:
            self.control_thread.join(timeout=2.0)
            if self.control_thread.is_alive():
                print("Control thread did not exit within timeout")
            self.control_thread = None

        print("Controller stopped")

    def prepare_for_env_reset(self):
        """在环境 reset 前清空控制状态。"""
        with self.ros_target_lock:
            self.ros_target_joints = None

        try:
            while True:
                self.target_queue.get_nowait()
        except Exception:
            pass

    def reset_environment_safely(self):
        """暂停控制循环后执行环境 reset，避免自动 reset 与控制线程并发。"""
        self._reset_pause_event.set()
        self.env.robot.pause_rendering()
        self.prepare_for_env_reset()
        time.sleep(self.dt * 2.0)
        try:
            _, info = self.env.reset()
            return info
        finally:
            self.env.robot.resume_rendering()
            self._reset_pause_event.clear()

    def _capture_initial_scene_state_safely(self) -> Optional[dict]:
        """短暂停止控制/渲染后读取一次录制初始场景状态。"""
        if not hasattr(self.env, "capture_scene_state"):
            return None

        self._reset_pause_event.set()
        self.env.robot.pause_rendering()
        time.sleep(self.dt * 2.0)
        acquired = self.env.robot._step_lock.acquire(timeout=1.0)
        if not acquired:
            print("⚠ 初始场景状态采集超时，跳过本次 scene_state")
            self.env.robot.resume_rendering()
            self._reset_pause_event.clear()
            return None

        try:
            return self.env.capture_scene_state()
        except Exception as e:
            print(f"⚠ 初始场景状态采集失败: {e}")
            return None
        finally:
            self.env.robot._step_lock.release()
            self.env.robot.resume_rendering()
            self._reset_pause_event.clear()

    def _write_initial_scene_when_recording_ready(self):
        """在 rosbag 目录创建后写入 initial_scene.json。"""
        if self.rosbag_recorder is None:
            return

        def _worker():
            bag_dir = None
            for _ in range(50):
                candidate = getattr(self.rosbag_recorder, "current_bag_dir", None)
                if candidate is not None and Path(candidate).exists():
                    bag_dir = Path(candidate)
                    break
                time.sleep(0.02)

            if bag_dir is None:
                print("⚠ 未等到 rosbag 目录，跳过 initial_scene.json")
                return

            scene_state = self._capture_initial_scene_state_safely()
            if scene_state is None:
                print("⚠ initial_scene.json 未写入：没有可用场景状态")
                return

            payload = {
                "format_version": 1,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "scene_state": scene_state,
            }
            try:
                path = bag_dir / "initial_scene.json"
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(payload, fp, ensure_ascii=False, indent=2)
                print(f"✓ 已保存 initial_scene: {path}")
            except Exception as e:
                print(f"⚠ 写入 initial_scene.json 失败: {e}")

        threading.Thread(target=_worker, name="initial_scene_writer", daemon=True).start()


def test_integrated_trajectory_control(
    model_path: Optional[str] = None,
    enable_visualization: bool = True,
    enable_ros_control: bool = True,
    enable_rosbag: bool = True,
    enable_topic_rename: bool = True,
    enable_image_publish: bool = True,
    enable_depth_render: bool = True,
    enable_thermal: bool = True,
    enable_teaching_recording: bool = True,
    enable_tactile_ui: bool = False,
    enable_task_snapshot_info: bool = False,
    tactile_ui_update_every: int = 5,
    control_loop_hz: float = 2000.0,
    image_publish_hz: float = 30.0,
    thermal_render_hz: float = 30.0,
    enable_profile_logs: bool = False,
    record_depth_topics: bool = False,
    record_thermal_topic: bool = False,
    record_tactile_topics: bool = False,
    release_target_xyz: Optional[np.ndarray] = None,
    release_target_radius: float = 0.08,
    grasp_plane_z: float = 0.0,
    lift_clearance: float = 0.02,
    placed_stable_min_frames: int = 5,
    placed_stable_max_lin_vel: float = 0.03,
    placed_stable_max_tilt_deg: float = 25.0,
    placed_plane_tolerance: float = 0.015,
):
    """Integrated direct control and visualization test"""

    def _collect_body_temp_and_pose(env_obj, thermal_sensor_obj, body_name: str) -> dict:
        """读取指定 body 的温度与世界坐标。"""
        body_id = mujoco.mj_name2id(env_obj.robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {
                "body_name": body_name,
                "exists": False,
                "temperature_c": None,
                "position_xyz": None,
                "mass_kg": None,
                "gravity_n": None,
                "contact_geom": None,
                "material_name": None,
                "friction": None,
                "solref": None,
                "solimp": None,
            }

        temp = None
        if thermal_sensor_obj is not None:
            temp = float(thermal_sensor_obj.body_temperatures.get(body_id, np.nan))
            if np.isnan(temp):
                temp = None

        pos = env_obj.robot.data.xpos[body_id]
        mass_kg = float(env_obj.robot.model.body_mass[body_id])
        gravity_n = float(mass_kg * np.linalg.norm(env_obj.robot.model.opt.gravity))

        # 读取该 body 的一个代表性接触 geom 参数（优先选择绑定了 material 的 geom）
        geom_adr = int(env_obj.robot.model.body_geomadr[body_id])
        geom_num = int(env_obj.robot.model.body_geomnum[body_id])
        rep_geom_id = None
        for gi in range(geom_adr, geom_adr + geom_num):
            if int(env_obj.robot.model.geom_matid[gi]) >= 0:
                rep_geom_id = gi
                break
        if rep_geom_id is None and geom_num > 0:
            rep_geom_id = geom_adr

        contact_geom = None
        material_name = None
        friction = None
        solref = None
        solimp = None
        if rep_geom_id is not None:
            contact_geom = mujoco.mj_id2name(env_obj.robot.model, mujoco.mjtObj.mjOBJ_GEOM, rep_geom_id)
            mat_id = int(env_obj.robot.model.geom_matid[rep_geom_id])
            if mat_id >= 0:
                material_name = mujoco.mj_id2name(env_obj.robot.model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)
            friction = [float(v) for v in env_obj.robot.model.geom_friction[rep_geom_id]]
            solref = [float(v) for v in env_obj.robot.model.geom_solref[rep_geom_id]]
            solimp = [float(v) for v in env_obj.robot.model.geom_solimp[rep_geom_id]]

        return {
            "body_name": body_name,
            "exists": True,
            "temperature_c": temp,
            "position_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
            "mass_kg": mass_kg,
            "gravity_n": gravity_n,
            "contact_geom": contact_geom,
            "material_name": material_name,
            "friction": friction,
            "solref": solref,
            "solimp": solimp,
        }

    def _print_live_task_scene_info(
        env_obj,
        thermal_sensor_obj,
        frame_idx: int,
        print_to_console: bool = True,
    ) -> dict:
        """采集关键物体与目标盘的温度/位置快照，按需打印。"""
        movable_object_names = [
            "beaker1",
            "graduated_cylinder",
            "erlenmeyer_flask",
        ]
        target_plate_names = [
            "target_place_table_1",
            "target_place_table_2",
            "target_place_table_3",
        ]

        snapshot = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame": int(frame_idx),
            "movable_objects": [
                _collect_body_temp_and_pose(env_obj, thermal_sensor_obj, n)
                for n in movable_object_names
            ],
            "target_plates": [
                _collect_body_temp_and_pose(env_obj, thermal_sensor_obj, n)
                for n in target_plate_names
            ],
        }

        if print_to_console:
            print(f"\n=== Live Task Snapshot | frame={frame_idx} | t={snapshot['timestamp']} ===")
            print("Movable objects:")
            for item in snapshot["movable_objects"]:
                print(
                    f"  - {item['body_name']}: temp={item['temperature_c']}°C, "
                    f"pos={item['position_xyz']}, "
                    f"mass={item['mass_kg']:.4f}kg, gravity={item['gravity_n']:.4f}N"
                )
                if item["friction"] is not None:
                    print(
                        f"    material={item['material_name']}, geom={item['contact_geom']}, "
                        f"friction={item['friction']}, solref={item['solref']}, solimp={item['solimp']}"
                    )
            print("Target plates:")
            for item in snapshot["target_plates"]:
                print(
                    f"  - {item['body_name']}: temp={item['temperature_c']}°C, "
                    f"pos={item['position_xyz']}, "
                    f"mass={item['mass_kg']:.4f}kg, gravity={item['gravity_n']:.4f}N"
                )

        return snapshot

    def _get_body_world_position(env_obj, body_name: str) -> Optional[np.ndarray]:
        """读取 body 世界坐标，不存在时返回 None。"""
        body_id = mujoco.mj_name2id(env_obj.robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return None
        return np.array(env_obj.robot.data.xpos[body_id], dtype=np.float64)

    def _evaluate_auto_reset_condition(
        env_obj,
        movable_object_names,
        target_body_names,
        xy_distance_threshold: float,
        z_distance_threshold: float,
    ) -> Tuple[bool, dict]:
        """
        判断三个物体是否都靠近目标区域。

        这里使用平面距离:
            d_xy = ||p_obj_xy - p_target_xy||_2
        仅当 d_xy 小于阈值、且高度差 |z_obj - z_target| 小于阈值时，
        才认为该物体已经到达某个 target 附近。
        """
        target_positions = {}
        for target_name in target_body_names:
            target_pos = _get_body_world_position(env_obj, target_name)
            if target_pos is not None:
                target_positions[target_name] = target_pos

        if len(target_positions) != len(target_body_names):
            return False, {}

        distance_report = {}
        all_objects_near_target = True

        for object_name in movable_object_names:
            object_pos = _get_body_world_position(env_obj, object_name)
            if object_pos is None:
                all_objects_near_target = False
                continue

            nearest_target_name = None
            nearest_xy_distance = None
            nearest_z_distance = None

            for target_name, target_pos in target_positions.items():
                xy_distance = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
                z_distance = float(abs(object_pos[2] - target_pos[2]))
                if nearest_xy_distance is None or xy_distance < nearest_xy_distance:
                    nearest_target_name = target_name
                    nearest_xy_distance = xy_distance
                    nearest_z_distance = z_distance

            is_near_target = (
                nearest_xy_distance is not None
                and nearest_xy_distance <= xy_distance_threshold
                and nearest_z_distance is not None
                and nearest_z_distance <= z_distance_threshold
            )
            all_objects_near_target = all_objects_near_target and is_near_target
            distance_report[object_name] = {
                "nearest_target": nearest_target_name,
                "xy_distance_m": nearest_xy_distance,
                "z_distance_m": nearest_z_distance,
                "is_near_target": is_near_target,
            }

        return all_objects_near_target, distance_report

    print("=" * 70)
    print("Arm Direct Control Test")
    print("=" * 70)

    try:
        # 1. 创建环境
        print("\nInitializing environment...")
        env_xml_path = model_path or os.path.join(os.path.dirname(__file__), '../robot_model/exp/env_robot_torque_tactile.xml')
        env = ArmEnv(
            model_path=env_xml_path,
            enable_visualization=enable_visualization,
            enable_depth_render=enable_depth_render,
            enable_robot_cameras=False,
            maintain_orientation=True,
            sample_mode='full',
            uniform_sampling=True         # 均匀采样
        )

        # 2. 重置环境
        _, info = env.reset()

        # 可选：打开文档同款触觉可视化窗口（热力图+切向箭头）
        tactile_visualizer = None
        if enable_tactile_ui:
            if tactile_visualizer_available and GripperTactileVisualizer is not None:
                try:
                    tactile_visualizer = GripperTactileVisualizer(
                        env.robot.model,
                        env.robot.data,
                        prefixes=("left", "right"),
                        rows=10,
                        cols=5,
                        update_every=max(1, int(tactile_ui_update_every)),
                    )
                    print("✓ Tactile UI enabled (direct MuJoCo sensors)")
                except Exception as e:
                    tactile_visualizer = None
                    print(f"⚠ Failed to initialize tactile UI: {e}")
            else:
                print("⚠ tactile UI requested but tactile_visualizer sensor is unavailable")

        current_pos = info['ee_position']
        print(f"Environment reset, EE position: {current_pos}")

        # 检查ROS2 Joint State发布状态
        if env.robot.enable_joint_state_ros2 and env.robot.joint_state_pub is not None:
            print(f"ROS2 Joint State: Enabled ({env.robot.joint_state_topic}, 500Hz)")
        else:
            print("ROS2 Joint State: Disabled")

        # 3. 禁用障碍物运动（默认移除障碍物）
        env.robot.disable_obstacles()
        print("Obstacles: Disabled")

        # 设置为力矩控制模式以启用PID控制
        env.robot.set_control_mode("torque")
        print("Control mode: Torque (PID enabled)")

        ee_target_subscriber = None
        ee_target_mocap_id = -1
        ee_target_base_body_id = _find_robot_base_body_id(env.robot.model)
        if ros2_available():
            try:
                marker_body_id = int(mujoco.mj_name2id(env.robot.model, mujoco.mjtObj.mjOBJ_BODY, "ee_pose_target_marker"))
                if marker_body_id >= 0:
                    ee_target_mocap_id = int(env.robot.model.body_mocapid[marker_body_id])
                if ee_target_mocap_id >= 0:
                    ee_target_subscriber = EETargetSubscriberROS2(
                        topic="/ee_target",
                        node_name=f"ee_target_visualizer_{int(time.time() * 1000) % 10000}",
                    )
                    ee_target_subscriber.start_spinning()
                    print("✓ /ee_target target marker enabled")
                    print("  - Visual body: ee_pose_target_marker")
                    print("  - Topic: /ee_target (base_link frame) -> mocap body pose")
            except Exception as e:
                ee_target_subscriber = None
                ee_target_mocap_id = -1
                print(f"⚠ EE pose target marker init failed: {e}")

        def _configure_default_thermal_scene(thermal_sensor_obj):
            thermal_sensor_obj.set_body_temperature("bench", 35.0)
            base_temp = float(np.random.uniform(20.0, 40.0))
            target_temps = np.array([base_temp, base_temp + 20.0, base_temp + 40.0], dtype=np.float64)
            np.random.shuffle(target_temps)
            thermal_sensor_obj.set_body_temperature("target_place_table_1", float(target_temps[0]))
            thermal_sensor_obj.set_body_temperature("target_place_table_2", float(target_temps[1]))
            thermal_sensor_obj.set_body_temperature("target_place_table_3", float(target_temps[2]))
            print(
                "  target_place_table temperatures: "
                f"{target_temps[0]:.1f}°C, {target_temps[1]:.1f}°C, {target_temps[2]:.1f}°C"
            )
            obj_base_temp = float(np.random.uniform(0.0, 60.0))
            object_temps = np.array(
                [obj_base_temp, obj_base_temp + 20.0, obj_base_temp + 40.0],
                dtype=np.float64,
            )
            np.random.shuffle(object_temps)
            thermal_sensor_obj.set_liquid_temperature("beaker1", float(object_temps[0]), glass_conductivity=0.5)
            thermal_sensor_obj.set_liquid_temperature("graduated_cylinder", float(object_temps[1]), glass_conductivity=0.5)
            thermal_sensor_obj.set_liquid_temperature("erlenmeyer_flask", float(object_temps[2]), glass_conductivity=0.5)
            print(
                "  movable object temperatures: "
                f"beaker1={object_temps[0]:.1f}°C, "
                f"graduated_cylinder={object_temps[1]:.1f}°C, "
                f"erlenmeyer_flask={object_temps[2]:.1f}°C"
            )

        render_thread = ReplicaRenderThread(
            robot=env.robot,
            enable_depth_render=enable_depth_render,
            enable_thermal=False,
            thermal_sensor_initializer=None,
            rgbd_publishers={},
            enable_profile_logs=enable_profile_logs,
        )

        # 4. 创建独立的 ROS2 触觉发布线程
        ros2_publisher_thread = None

        if ros2_available():
            try:
                camera_configs = []
                node_suffix = int(time.time() * 1000) % 100000
                if enable_image_publish:
                    camera_configs = [
                        {'camera_name': 'ee_camera', 'node_name': f'ee_camera_publish_thread_{node_suffix}'},
                        {'camera_name': 'external_camera', 'node_name': f'external_camera_publish_thread_{node_suffix}'},
                    ]
                ros2_publisher_thread = ROS2PublisherThread(
                    camera_configs,
                )
                if ros2_publisher_thread.start():
                    if enable_image_publish:
                        print(f"ROS2 RGBD/tactile publisher thread: Started ({image_publish_hz:.1f}Hz RGBD)")
                    else:
                        print("ROS2 tactile publisher thread: Started")
                else:
                    ros2_publisher_thread = None

            except Exception as e:
                print(f"ROS2 publisher initialization failed: {e}")
                import traceback
                traceback.print_exc()
                ros2_publisher_thread = None
        else:
            print("ROS2 not available, publisher thread disabled")

        # 5. 创建连续控制器并启用ROS2控制和Teaching记录
        print("\nCreating controller...")
        controller = ContinuousArmController(
            env,
            enable_ros_control=enable_ros_control,
            enable_teaching_recording=enable_teaching_recording,
            enable_rosbag=enable_rosbag,
            enable_topic_rename=enable_topic_rename,
            record_depth_topics=record_depth_topics,
            record_thermal_topic=record_thermal_topic,
            record_tactile_topics=record_tactile_topics,
            release_target_xyz=release_target_xyz,
            release_target_radius=release_target_radius,
            grasp_plane_z=grasp_plane_z,
            lift_clearance=lift_clearance,
            placed_stable_min_frames=placed_stable_min_frames,
            placed_stable_max_lin_vel=placed_stable_max_lin_vel,
            placed_stable_max_tilt_deg=placed_stable_max_tilt_deg,
            placed_plane_tolerance=placed_plane_tolerance,
        )

        # 启动连续控制
        print("\n🚀 启动连续机械臂控制...")
        controller.start()

        # 初始化热成像 ROS2 发布器
        thermal_publisher = None
        if enable_thermal:
            try:
                from thermal_sensor import ThermalImagePublisher
                thermal_publisher = ThermalImagePublisher(
                    topic="/thermal_camera/image",
                    node_name="thermal_image_publisher"
                )
                print("✓ Thermal ROS2 publisher: /thermal_camera/image")
            except Exception as e:
                print(f"⚠ Thermal ROS2 publisher init failed: {e}")
                thermal_publisher = None

        thermal_thread = None
        if enable_thermal:
            thermal_thread = ThermalRenderThread(
                robot=env.robot,
                thermal_publish_hz=thermal_render_hz,
                thermal_sensor_initializer=_configure_default_thermal_scene,
                thermal_publisher=thermal_publisher,
                enable_profile_logs=enable_profile_logs,
            )

        def _build_task_snapshot_for_controller(frame_idx: int) -> dict:
            return _print_live_task_scene_info(
                env,
                thermal_thread if thermal_thread is not None else render_thread,
                frame_idx,
                print_to_console=False,
            )
        controller.task_snapshot_provider = _build_task_snapshot_for_controller

        # 图像直发分发器：渲染完成后立即发布，减少 copy 和中间缓存
        render_dispatcher = DirectFrameDispatcher(
            ros2_publisher_thread=ros2_publisher_thread,
            thermal_publisher=None,
            controller=controller,
            thermal_publish_hz=thermal_render_hz,
            image_publish_hz=image_publish_hz,
            enable_profile_logs=enable_profile_logs,
        )
        render_thread.frame_callback = render_dispatcher.handle_frame

        render_dispatcher.start()
        if not render_thread.start():
            raise RuntimeError("渲染副本线程启动失败，请检查前面的初始化失败/超时日志")
        if thermal_thread is not None and not thermal_thread.start():
            raise RuntimeError("热成像线程启动失败，请检查前面的初始化失败/超时日志")

        # 6. 主监控循环
        try:
            print(f"\nStarting monitor loop ({control_loop_hz:.1f}Hz)")
            print("Press Ctrl+C to exit")

            frame_count = 0
            last_scene_report_time = 0.0
            scene_report_interval_s = 1.0
            




            while True:
                loop_start = time.time()
                # print(loop_start)
               
                try:
                    frame_count = render_dispatcher.bridge_frame_count

                    # 运行中仅按需打印轻量调试快照，不录入 bag。
                    if (
                        enable_task_snapshot_info
                        and loop_start - last_scene_report_time >= scene_report_interval_s
                    ):
                        _print_live_task_scene_info(
                            env,
                            thermal_thread if thermal_thread is not None else render_thread,
                            frame_count,
                            print_to_console=True,
                        )
                        last_scene_report_time = loop_start

                    if controller.pending_reset_after_teaching_end:
                        print("↻ 检测到 end_teaching 触发的 reset 请求，执行安全 reset")
                        info = controller.reset_environment_safely()
                        controller.pending_reset_after_teaching_end = False
                        current_pos = info['ee_position']
                        print(f"  reset 完成，EE position: {current_pos}")
                        continue

                    # 更新独立触觉可视化窗口（不依赖 ROS2）
                    if tactile_visualizer is not None:
                        try:
                            tactile_visualizer.update()
                        except Exception as e:
                            if frame_count % 100 == 1:
                                print(f"⚠ tactile UI update failed: {e}")

                    if ee_target_subscriber is not None and ee_target_mocap_id >= 0:
                        try:
                            target_pose = ee_target_subscriber.get_latest_pose()
                            if target_pose is not None:
                                target_position_base, target_quaternion_base = target_pose
                                with env.robot._step_lock:
                                    target_position_world, target_quaternion_world = _pose_from_base_link_to_world(
                                        env.robot,
                                        ee_target_base_body_id,
                                        target_position_base,
                                        target_quaternion_base,
                                    )
                                    env.robot.data.mocap_pos[ee_target_mocap_id] = target_position_world
                                    env.robot.data.mocap_quat[ee_target_mocap_id] = target_quaternion_world
                        except Exception as e:
                            if frame_count % 200 == 1:
                                print(f"⚠ EE pose target marker update failed: {e}")

                    # 触觉数据仍在主循环中更新，避免改动现有传感器读取路径
                    if ros2_publisher_thread is not None:
                        try:
                            if hasattr(env.robot, 'has_tactile_force_sensors') and env.robot.has_tactile_force_sensors():
                                tactile_components = env.robot.get_tactile_force_components(clip_min=0.0)
                                if tactile_components is not None:
                                    ros2_publisher_thread.update_tactile_data(
                                        tactile_components=tactile_components,
                                        timestamp_sec=env.robot.sim_time,
                                    )

                        except Exception as e:
                            if frame_count % 100 == 1:
                                print(f"⚠ 触觉数据更新失败: {e}")

                except Exception as e:
                    print(f"⚠ 主监控循环错误: {e}")

 
                # 控制循环频率到60Hz
                loop_duration = time.time() - loop_start
                target_duration = 1.0 / control_loop_hz
                sleep_time = target_duration - loop_duration

                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nUser interrupted")
        except EOFError:
            print("\nNon-interactive environment detected, keeping robot in direct-control idle mode")
            time.sleep(10)
            print("\nDemo completed")

        # 7. 停止控制器
        print("\nStopping controller...")
        controller.stop()

        # 关闭触觉可视化窗口
        if tactile_visualizer is not None:
            try:
                tactile_visualizer.close()
            except Exception as e:
                print(f"Tactile visualizer close error: {e}")

        if 'ee_target_subscriber' in locals() and ee_target_subscriber is not None:
            try:
                ee_target_subscriber.shutdown()
            except Exception as e:
                print(f"EE pose target subscriber stop error: {e}")

        # 8. 停止 ROS2 发布线程
        if ros2_publisher_thread is not None:
            print("Stopping ROS2 publisher...")
            ros2_publisher_thread.stop()

        if 'render_dispatcher' in locals() and render_dispatcher is not None:
            try:
                print("Stopping RGBD timed publish thread...")
                render_dispatcher.stop()
            except Exception as e:
                print(f"RGBD timed publish thread stop error: {e}")

        if render_thread is not None:
            try:
                print("Stopping replica render thread...")
                render_thread.stop()
            except Exception as e:
                print(f"Replica render thread stop error: {e}")

        if 'thermal_thread' in locals() and thermal_thread is not None:
            try:
                print("Stopping thermal render thread...")
                thermal_thread.stop()
            except Exception as e:
                print(f"Thermal render thread stop error: {e}")

        # 9. 关闭环境
        print("Closing environment...")
        env.close()

        print("\nTest completed")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="MuJoCo 机械臂仿真接口")
        parser.add_argument("--model", type=str, default=None, help="MuJoCo 场景 XML 路径")
        parser.add_argument("--headless", action="store_true", help="无可视化模式（默认开启GUI）")
        parser.set_defaults(headless=False)
        parser.add_argument("--no-ros-control", action="store_true", help="禁用 ROS 控制输入")
        parser.add_argument("--no-rosbag", action="store_true", help="禁用 rosbag 录制")
        parser.add_argument("--no-topic-rename", action="store_true", help="禁用 rosbag 录制完成后的话题重命名")
        parser.add_argument("--no-image-publish", action="store_true", help="禁用 RGBD ROS2 发布")
        parser.add_argument("--no-depth-render", action="store_true", help="禁用 RGBD 相机深度渲染")
        parser.add_argument("--no-thermal", action="store_true", help="禁用热成像发布")
        parser.add_argument("--tactile-ui", action="store_true", help="启用独立触觉可视化窗口（热力图+切向箭头）")
        parser.add_argument("--print-task-snapshot-info", action="store_true", help="打印实时任务快照信息（Live Task Snapshot）")
        parser.add_argument("--control-loop-hz", type=float, default=15.0, help="主循环频率 (Hz)")
        parser.add_argument("--image-publish-hz", type=float, default=30.0, help="RGBD 图像定时发布频率 (Hz)")
        parser.add_argument("--thermal-render-hz", type=float, default=30.0, help="热成像渲染频率 (Hz)")
        parser.add_argument("--profile-logs", action="store_true", help="输出 Render/Dispatch/Thermal profile 的 info 级日志")
        parser.add_argument("--record-depth-topics", action="store_true", help="录制深度图 topic，并在归档时导出深度可视化 MP4")
        parser.add_argument("--record-thermal-topic", action="store_true", help="录制红外 /thermal_camera/image topic，并在归档时导出 MP4")
        parser.add_argument("--record-tactile-topics", action="store_true", help="录制触觉 topic")
        args = parser.parse_args()

        logging.basicConfig(
            level=logging.INFO if args.profile_logs else logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        test_integrated_trajectory_control(
            model_path=args.model,
            enable_visualization=not args.headless,
            enable_ros_control=not args.no_ros_control,
            enable_rosbag=not args.no_rosbag,
            enable_topic_rename=not args.no_topic_rename,
            enable_image_publish=not args.no_image_publish,
            enable_depth_render=not args.no_depth_render,
            enable_thermal=not args.no_thermal,
            enable_tactile_ui=args.tactile_ui,
            enable_task_snapshot_info=args.print_task_snapshot_info,
            enable_teaching_recording=True,
            control_loop_hz=args.control_loop_hz,
            image_publish_hz=args.image_publish_hz,
            thermal_render_hz=args.thermal_render_hz,
            enable_profile_logs=args.profile_logs,
            record_depth_topics=args.record_depth_topics,
            record_thermal_topic=args.record_thermal_topic,
            record_tactile_topics=args.record_tactile_topics,
        )
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"\nProgram error: {e}")
        import traceback
        traceback.print_exc()
