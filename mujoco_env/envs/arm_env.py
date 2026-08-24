"""
机械臂环境底层实现。

说明:
1. 当前主线已不再面向 RL 使用
2. 当前主线已不再依赖 IK 规划器
3. 该文件暂作为底层兼容实现保留
"""

import numpy as np
import mujoco
from mujoco import viewer as mujoco_viewer
from typing import Tuple, Dict, Any, Optional, List
import sys
import os
import time

# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../robot'))

# 导入Robot类
from robot import Robot


class ArmEnv:
    """
    机械臂环境底层实现。

    控制流程:
    1. 输入笛卡尔增量: action = [Δx, Δy, Δz]
    2. 计算目标末端位置: target_pos = current_pos + delta
    3. 使用局部雅可比伪逆更新目标关节角度
    4. 控制机械臂移动到目标关节角度
    """

    def __init__(
        self,
        model_path: str = None,
        max_episode_steps: int = 500,
        delta_scale: float = 0.01,
        target_reach_threshold: float = 0.05,
        render_mode: Optional[str] = None,
        enable_visualization: bool = False,
        enable_depth_render: bool = True,
        enable_robot_cameras: bool = True,
        maintain_orientation: bool = True,  # 是否保持末端姿态不变
        sample_mode: str = 'full',  # 采样模式: 'single' (只采样joint1) 或 'full' (全关节采样)
        uniform_sampling: bool = True,  # 是否使用均匀采样（否则使用随机采样）
        initial_joint_positions: Optional[np.ndarray] = None  # 自定义初始关节位置，形状 (6,)
    ):
        """
        初始化环境

        Args:
            model_path: MuJoCo XML模型路径
            max_episode_steps: 每个episode的最大步数
            delta_scale: 动作缩放因子
            target_reach_threshold: 目标到达阈值
            render_mode: 渲染模式
            enable_visualization: 是否启用MuJoCo viewer
            enable_depth_render: 是否启用 RGBD 相机深度渲染
            enable_robot_cameras: 是否在 Robot 主线程初始化相机/渲染器
            maintain_orientation: 是否保持末端姿态不变
            sample_mode: 采样模式 ('single': 只采样joint1, 'full': 全关节采样)
            uniform_sampling: 是否使用均匀采样 (True: 均匀网格采样, False: 随机采样)
            initial_joint_positions: 自定义初始关节位置 (6,) 数组，单位：弧度
        """
        # 设置模型路径
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "../robot_model/exp/env_robot_torque.xml")

        # 初始化Robot
        self.robot = Robot(
            model_path,
            enable_ros2=True,              # 启用经典ROS2发布功能（摄像头）
            enable_cameras=enable_robot_cameras,
            enable_joint_state_ros2=True,  # 启用ROS2关节状态发布
            joint_state_topic="/joint_states_sim",
            enable_depth_render=enable_depth_render
        )
        self.robot.set_control_mode("zero_gravity")

        # 环境参数
        self.max_episode_steps = max_episode_steps
        self.delta_scale = delta_scale
        self.target_reach_threshold = target_reach_threshold
        self.render_mode = render_mode
        self.enable_visualization = enable_visualization
        self.maintain_orientation = maintain_orientation
        self.sample_mode = sample_mode  # 采样模式
        self.uniform_sampling = uniform_sampling  # 是否均匀采样

        # 关节限位
        self.joint_limits = self.robot.model.jnt_range[:6].copy()  # shape: (6, 2)
        print(f"✓ 关节限位已加载")
        print(f"  - 使用MuJoCo关节限位: {self.joint_limits.flatten()}")

        # Episode状态
        self.current_step = 0

        # 目标状态
        self.target_pos = np.array([0.4, 0.3, 0.5])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 工作空间边界（已禁用）
        # 仍保留字段以兼容旧代码路径，但不再用于裁剪/终止/采样过滤
        self.workspace_center = np.array([0.3, 0.0, 0.5])
        self.workspace_range = np.array([np.inf, np.inf, np.inf])
        self.workspace_min = np.array([-np.inf, -np.inf, -np.inf])
        self.workspace_max = np.array([np.inf, np.inf, np.inf])

        # 观测和动作空间维度
        self.observation_dim = 47
        self.action_dim = 3

        # 初始关节位姿设置
        if initial_joint_positions is not None:
            if isinstance(initial_joint_positions, np.ndarray) and initial_joint_positions.shape == (6,):
                self.initial_joint_positions = initial_joint_positions.copy()
                print(f"  使用自定义初始关节位置(度): {np.rad2deg(self.initial_joint_positions)}")
            else:
                print(f"⚠ 初始关节位置格式错误，使用默认值")
                self.initial_joint_positions = np.deg2rad([0, 0, -90, 0, -90, 0])
        else:
            self.initial_joint_positions = np.deg2rad([0, 0, -90, 0, -90, 0])
            print(f"  使用默认初始关节位置(度): {np.rad2deg(self.initial_joint_positions)}")

        # 机械臂关节名称与关节限位（按关节名读取，避免 freejoint 打乱 qpos 顺序）
        self._arm_joint_names = [f"joint{i+1}" for i in range(6)]
        self.joint_limits = np.array([
            self.robot.model.jnt_range[
                mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self._arm_joint_names
        ], dtype=np.float64)

        # 物体位置泛化配置（仅在 reset 时随机化）
        # 以给定中心点为基准做局部域随机化：
        #   x = x0 + U(-dx, dx)
        #   y = y0 + U(-dy, dy)
        #   z = z0 + U(dz_min, dz_max)
        # 这里使用 z 方向仅向上抬起，避免采样到桌面以下。
        self._generalized_object_center_xyz = np.array([1.1, 0.3, 0.98], dtype=np.float64)
        self._generalized_object_xy_half_range = 0.15
        self._generalized_object_z_offset_range = (0.0, 0.05)
        self._generalized_object_names = [
            'beaker1',
            'graduated_cylinder',
            'erlenmeyer_flask',
        ]
        self._generalized_object_min_separation = 0.08
        self._robot_base_body_name = 'base'
        self._target_place_geom_name = 'target_place_table_base'
        self._target_place_clearance = 0.08

        # 物体材质配置（reset 时随机化）
        self._material_candidates = ['metal_mat', 'wood_mat', 'glass_mat']
        self._object_material_geom_groups = {
            'beaker1': ['beaker1_glass'],
            'graduated_cylinder': ['grad_cyl_body', 'grad_cyl_base'],
            'erlenmeyer_flask': ['flask_base', 'flask_mid', 'flask_neck'],
        }
        self._object_current_materials: Dict[str, str] = {}
        # 材质 -> 接触参数映射
        # friction = [slide, torsional, rolling]
        # solref/solimp 控制接触软硬与恢复特性，可近似理解为“弹性/阻尼”调节
        self._material_contact_profiles = {
            'metal_mat': {
                'friction': np.array([0.65, 0.018, 0.0018], dtype=np.float64),
                'solref': np.array([0.055, 2.5], dtype=np.float64),
                'solimp': np.array([0.93, 0.985, 0.0008, 0.5, 2.0], dtype=np.float64),
            },
            'wood_mat': {
                'friction': np.array([0.85, 0.012, 0.0012], dtype=np.float64),
                'solref': np.array([0.030, 1.0], dtype=np.float64),
                'solimp': np.array([0.90, 0.95, 0.001, 0.5, 2.0], dtype=np.float64),
            },
            'glass_mat': {
                'friction': np.array([0.45, 0.014, 0.0012], dtype=np.float64),
                'solref': np.array([0.045, 2.2], dtype=np.float64),
                'solimp': np.array([0.94, 0.99, 0.0008, 0.5, 2.0], dtype=np.float64),
            },
        }
        # 材质 -> 姿态阻尼系数（单位: 1/s）
        # 角速度阻尼公式：
        #   omega_{t+dt} = omega_t * exp(-lambda * dt)
        # 这是连续时间一阶阻尼方程 d(omega)/dt = -lambda * omega 的解析解。
        # lambda 越大，物体绕姿态的摆动/旋转收敛越快。
        self._material_angular_damping = {
            'metal_mat': 16.0,
            'wood_mat': 6.0,
            'glass_mat': 14.0,
        }
        # 材质 -> 自阻尼系数（单位: 1/s）
        # 线速度/角速度统一采用一阶指数衰减：
        #   v_{t+dt} = v_t * exp(-lambda_v * dt)
        #   omega_{t+dt} = omega_t * exp(-lambda_w * dt)
        # 其中 lambda_v / lambda_w 越大，物体自身收敛越快。
        self._material_self_damping = {
            'metal_mat': {
                'linear': 20.0,
                'angular': 80.0,
            },
            'wood_mat': {
                'linear': 20.0,
                'angular': 80.0,
            },
            'glass_mat': {
                'linear': 20.0,
                'angular': 80.0,
            },
        }
        self._material_name_to_id = {
            mat_name: int(mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name))
            for mat_name in self._material_candidates
        }
        missing_materials = [name for name, mid in self._material_name_to_id.items() if mid < 0]
        if missing_materials:
            raise ValueError(f"场景缺少材质定义: {missing_materials}")

        self._geom_name_to_id: Dict[str, int] = {}
        for geom_names in self._object_material_geom_groups.values():
            for geom_name in geom_names:
                geom_id = int(mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
                if geom_id < 0:
                    raise ValueError(f"找不到 geom: {geom_name}")
                self._geom_name_to_id[geom_name] = geom_id
        self.robot.on_reset_callback = self._reset_scene_objects

        # Viewer
        self.viewer = None
        if self.enable_visualization:
            self.viewer = mujoco_viewer.launch_passive(self.robot.model, self.robot.data)
            # 关闭坐标轴显示（包括末端工具坐标轴）
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            
        print("=" * 70)
        print("✓ ArmEnv backend initialized")
        print("=" * 70)
        print(f"  - 保持姿态: {maintain_orientation}")
        print(f"  - 采样模式: {sample_mode}")
        print(f"  - 均匀采样: {uniform_sampling}")
        print(f"  - Observation dim: {self.observation_dim}")
        print(f"  - Action dim: {self.action_dim}")
        print("=" * 70)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        if seed is None:
            # 默认将随机种子与当前时间关联，避免每次 reset 都走完全相同的随机序列。
            # 这里使用纳秒时间戳:
            #   seed = t_ns mod 2^32
            # 这样可以兼容 numpy 的整数种子范围，同时保证相邻 reset 也有足够区分度。
            seed = int(time.time_ns() % (2 ** 32))
        self._last_reset_seed = int(seed)
        np.random.seed(seed)
        print(f"  reset seed: {seed}")

        self.robot.reset_simulation()
        self._set_arm_joint_positions(self.initial_joint_positions)
        mujoco.mj_forward(self.robot.model, self.robot.data)

        randomized_positions = getattr(self, '_last_randomized_positions', {})
        randomized_materials = getattr(self, '_last_randomized_materials', {})

        for body_name, position in randomized_positions.items():
            print(f"  随机化 {body_name}: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        for body_name, material_name in randomized_materials.items():
            profile = self._material_contact_profiles[material_name]
            friction = np.array2string(profile['friction'], precision=4, separator=', ')
            solref = np.array2string(profile['solref'], precision=4, separator=', ')
            solimp = np.array2string(profile['solimp'], precision=4, separator=', ')
            print(f"  材质随机化 {body_name}: {material_name}")
            print(f"    friction={friction}  solref={solref}  solimp={solimp}")

        # 采样新目标
        self.target_pos, self.target_quat = self._sample_target()
        self._update_target_visualization()

        self.current_step = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _reset_scene_objects(self):
        """在仿真 reset 后自动重新摆放物体并应用材质配置。"""
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self._last_randomized_positions = self._randomize_object_positions()
        self._last_randomized_materials = self._randomize_object_materials()
        mujoco.mj_forward(self.robot.model, self.robot.data)

    def _get_arm_joint_positions(self) -> np.ndarray:
        """按关节名读取 6 轴机械臂关节角。"""
        return np.array(
            [self.robot.data.joint(name).qpos[0] for name in self._arm_joint_names],
            dtype=np.float64,
        )

    def _set_arm_joint_positions(self, joints: np.ndarray):
        """按关节名写入 6 轴机械臂关节角。"""
        values = np.asarray(joints, dtype=np.float64)
        if values.shape != (6,):
            raise ValueError(f"arm joints shape must be (6,), got {values.shape}")
        for name, value in zip(self._arm_joint_names, values):
            self.robot.data.joint(name).qpos[0] = float(value)

    def _get_freejoint_qpos_addr(self, body_name: str) -> int:
        """获取 freejoint body 对应的 qpos 起始地址。"""
        body_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"找不到 body: {body_name}")

        joint_count = int(self.robot.model.body_jntnum[body_id])
        if joint_count <= 0:
            raise ValueError(f"body {body_name} 没有关节，无法随机化位置")

        joint_adr = int(self.robot.model.body_jntadr[body_id])
        if int(self.robot.model.jnt_type[joint_adr]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"body {body_name} 不是 freejoint body")

        return int(self.robot.model.jnt_qposadr[joint_adr])

    def _set_freejoint_body_pose(self, body_name: str, position: np.ndarray, quaternion: Optional[np.ndarray] = None):
        """直接修改 freejoint body 的位姿。"""
        qpos_adr = self._get_freejoint_qpos_addr(body_name)
        self.robot.data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(position, dtype=np.float64)

        if quaternion is None:
            quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.robot.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(quaternion, dtype=np.float64)

    def _get_freejoint_body_qpos(self, body_name: str) -> np.ndarray:
        """读取 freejoint body 的完整 qpos: xyz + quaternion(wxyz)。"""
        qpos_adr = self._get_freejoint_qpos_addr(body_name)
        return np.asarray(self.robot.data.qpos[qpos_adr:qpos_adr + 7], dtype=np.float64).copy()

    def _set_freejoint_body_qpos(self, body_name: str, qpos: np.ndarray):
        """写入 freejoint body 的完整 qpos: xyz + quaternion(wxyz)。"""
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError(f"{body_name} freejoint qpos shape must be (7,), got {values.shape}")
        qpos_adr = self._get_freejoint_qpos_addr(body_name)
        self.robot.data.qpos[qpos_adr:qpos_adr + 7] = values

    def _clear_freejoint_body_velocity(self, body_name: str):
        """清零 freejoint 刚体的线速度与角速度，避免 reset 后带着残余速度抖动。"""
        body_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"找不到 body: {body_name}")

        joint_count = int(self.robot.model.body_jntnum[body_id])
        if joint_count <= 0:
            return

        joint_adr = int(self.robot.model.body_jntadr[body_id])
        if int(self.robot.model.jnt_type[joint_adr]) != int(mujoco.mjtJoint.mjJNT_FREE):
            return

        qvel_adr = int(self.robot.model.jnt_dofadr[joint_adr])
        self.robot.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    def _get_freejoint_qvel_addr(self, body_name: str) -> int:
        """获取 freejoint body 对应的 qvel 起始地址。"""
        body_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"找不到 body: {body_name}")

        joint_count = int(self.robot.model.body_jntnum[body_id])
        if joint_count <= 0:
            raise ValueError(f"body {body_name} 没有关节，无法读取速度")

        joint_adr = int(self.robot.model.body_jntadr[body_id])
        if int(self.robot.model.jnt_type[joint_adr]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"body {body_name} 不是 freejoint body")

        return int(self.robot.model.jnt_dofadr[joint_adr])

    def _apply_object_material(self, body_name: str, material_name: str):
        """将一个可抓取物体恢复到指定材质对应的接触参数。"""
        if body_name not in self._object_material_geom_groups:
            raise ValueError(f"未知可抓取物体: {body_name}")
        if material_name not in self._material_name_to_id:
            raise ValueError(f"未知材质: {material_name}")

        contact_profile = self._material_contact_profiles[material_name]
        for geom_name in self._object_material_geom_groups[body_name]:
            geom_id = self._geom_name_to_id[geom_name]
            self.robot.model.geom_friction[geom_id, :] = contact_profile['friction']
            self.robot.model.geom_solref[geom_id, :] = contact_profile['solref']
            self.robot.model.geom_solimp[geom_id, :] = contact_profile['solimp']
        self._object_current_materials[body_name] = material_name

    def capture_scene_state(self) -> Dict[str, Any]:
        """导出当前可复现的任务场景状态，供录制归档和 replay 恢复。"""
        objects: Dict[str, Any] = {}
        for body_name in self._generalized_object_names:
            body_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                objects[body_name] = {"exists": False}
                continue
            qpos = self._get_freejoint_body_qpos(body_name)
            objects[body_name] = {
                "exists": True,
                "qpos": [float(v) for v in qpos],
                "position_xyz": [float(v) for v in qpos[:3]],
                "quaternion_wxyz": [float(v) for v in qpos[3:7]],
                "material_name": self._object_current_materials.get(body_name),
            }

        return {
            "format_version": 1,
            "reset_seed": int(getattr(self, "_last_reset_seed", -1)),
            "movable_objects": objects,
        }

    def apply_scene_state(self, scene_state: Dict[str, Any]) -> None:
        """恢复 capture_scene_state 导出的任务场景状态。"""
        if not scene_state:
            return

        objects = scene_state.get("movable_objects", {})
        for body_name, state in objects.items():
            if not isinstance(state, dict) or not state.get("exists", True):
                continue
            if body_name not in self._generalized_object_names:
                continue

            qpos = state.get("qpos")
            if qpos is not None:
                self._set_freejoint_body_qpos(body_name, np.asarray(qpos, dtype=np.float64))
            elif state.get("position_xyz") is not None:
                quat = state.get("quaternion_wxyz")
                self._set_freejoint_body_pose(
                    body_name,
                    np.asarray(state["position_xyz"], dtype=np.float64),
                    np.asarray(quat, dtype=np.float64) if quat is not None else None,
                )

            self._clear_freejoint_body_velocity(body_name)

            material_name = state.get("material_name")
            if material_name:
                self._apply_object_material(body_name, str(material_name))

        mujoco.mj_forward(self.robot.model, self.robot.data)

    def _apply_object_angular_damping(self):
        """对三个可抓取物体施加姿态阻尼，压制落桌后的持续转动抖动。"""
        dt = float(self.robot.model.opt.timestep)
        for body_name in self._generalized_object_names:
            material_name = self._object_current_materials.get(body_name)
            if material_name is None:
                continue

            damping_lambda = float(self._material_angular_damping.get(material_name, 0.0))
            if damping_lambda <= 0.0:
                continue

            qvel_adr = self._get_freejoint_qvel_addr(body_name)
            angular_vel = self.robot.data.qvel[qvel_adr + 3:qvel_adr + 6]

            # 一阶指数阻尼：
            #   omega_new = omega_old * exp(-lambda * dt)
            # 其中 lambda 是阻尼系数，dt 是仿真步长。
            decay = float(np.exp(-damping_lambda * dt))
            angular_vel *= decay

            # 对极小角速度做截断，避免数值误差导致“看起来一直在抖”。
            if float(np.linalg.norm(angular_vel)) < 1e-3:
                angular_vel[:] = 0.0

    def _apply_object_self_damping(self):
        """对自由刚体施加材质相关的自阻尼，提升平动与转动收敛速度。"""
        dt = float(self.robot.model.opt.timestep)
        for body_name in self._generalized_object_names:
            material_name = self._object_current_materials.get(body_name)
            if material_name is None:
                continue

            damping_cfg = self._material_self_damping.get(material_name)
            if damping_cfg is None:
                continue

            qvel_adr = self._get_freejoint_qvel_addr(body_name)
            linear_vel = self.robot.data.qvel[qvel_adr:qvel_adr + 3]
            angular_vel = self.robot.data.qvel[qvel_adr + 3:qvel_adr + 6]

            linear_decay = float(np.exp(-float(damping_cfg['linear']) * dt))
            angular_decay = float(np.exp(-float(damping_cfg['angular']) * dt))

            linear_vel *= linear_decay
            angular_vel *= angular_decay

            if float(np.linalg.norm(linear_vel)) < 1e-3:
                linear_vel[:] = 0.0
            if float(np.linalg.norm(angular_vel)) < 1e-3:
                angular_vel[:] = 0.0

    def _get_robot_base_xy(self) -> np.ndarray:
        """获取机械臂底座在世界坐标中的 XY 位置。"""
        for candidate_name in [self._robot_base_body_name, 'base_link']:
            body_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, candidate_name)
            if body_id >= 0:
                return np.asarray(self.robot.data.xpos[body_id][:2], dtype=np.float64)
        raise ValueError("找不到机械臂底座 body（base/base_link）")

    def _get_bench_top_bounds(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """获取主工作台台面的中心、半尺寸和台面高度。"""
        geom_name = 'bench_top'
        geom_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"找不到桌面 geom: {geom_name}")

        center_xy = np.asarray(self.robot.data.geom_xpos[geom_id][:2], dtype=np.float64)
        half_size_xy = np.asarray(self.robot.model.geom_size[geom_id][:2], dtype=np.float64)
        top_z = float(self.robot.data.geom_xpos[geom_id][2] + self.robot.model.geom_size[geom_id][2])
        return center_xy, half_size_xy, top_z

    def _is_position_on_bench_top(self, position: np.ndarray, margin: float = 0.05) -> bool:
        """判断一个世界坐标位置是否落在主工作台台面上。"""
        center_xy, half_size_xy, top_z = self._get_bench_top_bounds()
        xy = np.asarray(position[:2], dtype=np.float64)

        if np.any(np.abs(xy - center_xy) > (half_size_xy - margin)):
            return False

        return float(position[2]) >= top_z - 0.01

    def _is_position_away_from_target_place(self, position: np.ndarray) -> bool:
        """判断位置是否避开目标放置台区域（含安全边距）。"""
        geom_id = mujoco.mj_name2id(
            self.robot.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            self._target_place_geom_name,
        )
        if geom_id < 0:
            # 场景中不存在目标放置台时，不阻塞采样。
            return True

        center_xy = np.asarray(self.robot.data.geom_xpos[geom_id][:2], dtype=np.float64)
        half_size_xy = np.asarray(self.robot.model.geom_size[geom_id][:2], dtype=np.float64)
        half_size_xy = half_size_xy + self._target_place_clearance

        rot = np.asarray(self.robot.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        rot_xy = rot[:2, :2]
        local_xy = rot_xy.T @ (np.asarray(position[:2], dtype=np.float64) - center_xy)

        return bool(np.any(np.abs(local_xy) > half_size_xy))

    def _randomize_object_positions(self) -> Dict[str, np.ndarray]:
        """围绕给定中心点随机化三个物体的位置。"""
        object_names = list(self._generalized_object_names)
        best_positions: Dict[str, np.ndarray] = {}
        _, _, bench_top_z = self._get_bench_top_bounds()
        center_x, center_y, center_z = self._generalized_object_center_xyz
        xy_half_range = float(self._generalized_object_xy_half_range)
        z_offset_min, z_offset_max = self._generalized_object_z_offset_range
        min_spawn_z = float(bench_top_z + 1e-3)

        for _ in range(100):
            candidate_positions: Dict[str, np.ndarray] = {}
            valid = True
            for body_name in object_names:
                x = float(np.random.uniform(center_x - xy_half_range, center_x + xy_half_range))
                y = float(np.random.uniform(center_y - xy_half_range, center_y + xy_half_range))
                z = float(max(center_z + np.random.uniform(z_offset_min, z_offset_max), min_spawn_z))
                candidate_positions[body_name] = np.array([x, y, z], dtype=np.float64)

                if not self._is_position_on_bench_top(candidate_positions[body_name]):
                    valid = False
                    break
                if not self._is_position_away_from_target_place(candidate_positions[body_name]):
                    valid = False
                    break

            if not valid:
                continue
            for i in range(len(object_names)):
                for j in range(i + 1, len(object_names)):
                    p1 = candidate_positions[object_names[i]][:2]
                    p2 = candidate_positions[object_names[j]][:2]
                    if float(np.linalg.norm(p1 - p2)) < self._generalized_object_min_separation:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                best_positions = candidate_positions
                break

        if not best_positions:
            fallback_xy_offsets = [
                np.array([-0.12, -0.12], dtype=np.float64),
                np.array([0.12, -0.12], dtype=np.float64),
                np.array([0.00, 0.12], dtype=np.float64),
            ]
            best_positions = {
                body_name: np.array([
                    center_x + float(fallback_xy_offsets[idx][0]),
                    center_y + float(fallback_xy_offsets[idx][1]),
                    max(center_z, min_spawn_z),
                ], dtype=np.float64)
                for idx, body_name in enumerate(object_names)
            }

        for body_name, position in best_positions.items():
            self._set_freejoint_body_pose(body_name, position)
            self._clear_freejoint_body_velocity(body_name)

        return best_positions

    def _randomize_object_materials(self) -> Dict[str, str]:
        """为三个目标物体随机分配材质，并同步更新对应接触参数。"""
        selected_materials: Dict[str, str] = {}
        body_names = list(self._object_material_geom_groups.keys())

        # 若材质数与物体数一致，则采用随机排列，保证每次三个物体材质互不重复。
        # 这相当于对材质集合做一个随机双射：
        #   body_i -> perm(material_set)_i
        # 既保留域随机化，又避免三个物体恰好抽到同一种材质。
        if len(self._material_candidates) >= len(body_names):
            shuffled_materials = list(np.random.permutation(self._material_candidates[:len(body_names)]))
        else:
            shuffled_materials = [
                str(np.random.choice(self._material_candidates))
                for _ in body_names
            ]

        for body_name, material_name in zip(body_names, shuffled_materials):
            # 方块颜色直接由 XML rgba 控制；reset 时只更新接触参数，避免材质颜色覆盖红/绿/蓝。
            self._apply_object_material(body_name, material_name)
            selected_materials[body_name] = material_name
        self._object_current_materials = selected_materials.copy()

        return selected_materials

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作

        完整流程:
        1. 获取当前末端位姿
        2. 计算目标末端位置 = 当前位置 + 增量
        3. 使用局部雅可比伪逆计算目标关节角度
        4. 控制机械臂移动
        """
        # 限制动作范围
        action = np.clip(action, -1.0, 1.0)

        # 1. 获取当前末端位姿
        current_ee_pos = self.robot.get_ee_position()
        current_ee_quat = self.robot.get_ee_orientation()

        # 2. 计算目标末端位置
        delta = action * self.delta_scale
        target_ee_pos = current_ee_pos + delta

        # 3. 确定目标姿态
        if self.maintain_orientation:
            # 保持当前姿态不变
            target_ee_quat = current_ee_quat
        else:
            # 使用固定姿态
            target_ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 4. 使用IK求解目标关节角度
        success = self._move_to_position_with_ik(target_ee_pos, target_ee_quat)

        # 更新障碍物
        self.robot.update_obstacles()

        # 对自由刚体施加自阻尼，提升整体平动/转动收敛速度。
        self._apply_object_self_damping()

        # 对自由刚体施加姿态阻尼，优先压制玻璃/金属材质的落桌后转动振荡。
        self._apply_object_angular_damping()

        # 执行仿真步
        self.robot.step()

        # 可视化
        if self.enable_visualization and self.viewer is not None:
            self.viewer.sync()

        # 更新状态
        self.current_step += 1
        obs = self._get_observation()
        reward = 0.0
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps

        info = self._get_info()
        info['delta'] = delta
        info['target_ee_pos'] = target_ee_pos
        info['move_success'] = success

        return obs, reward, terminated, truncated, info

    def _move_to_position_with_ik(self, target_pos: np.ndarray,
                                   target_quat: np.ndarray) -> bool:
        """
        使用局部雅可比伪逆移动到目标位置

        Args:
            target_pos: 目标位置 [x, y, z]
            target_quat: 目标姿态 [w, x, y, z]

        Returns:
            success: 目标更新是否成功
        """
        try:
            # 无IK模式：使用局部雅可比伪逆进行关节更新（位置优先）
            current_joints = self.robot.get_joint_positions()
            current_ee_pos = self.robot.get_ee_position()
            pos_error = target_pos - current_ee_pos

            joint_delta = self._compute_jacobian_pseudoinverse(pos_error)
            target_joints = current_joints + joint_delta
            target_joints = np.clip(target_joints, self.joint_limits[:, 0], self.joint_limits[:, 1])

            self.robot.apply_joint_control(target_joints)

            # 位置误差足够小时认为成功
            return float(np.linalg.norm(pos_error)) < 0.01

        except Exception as e:
            print(f"⚠ 关节目标计算失败: {e}")
            return False

    def _compute_jacobian_pseudoinverse(self, pos_error: np.ndarray) -> np.ndarray:
        """数值计算雅可比伪逆"""
        epsilon = 1e-6
        current_joints = self.robot.get_joint_positions()
        current_ee_pos = self.robot.get_ee_position()

        n_joints = len(current_joints)
        jacobian = np.zeros((3, n_joints))

        for i in range(n_joints):
            perturbed_joints = current_joints.copy()
            perturbed_joints[i] += epsilon

            self.robot.data.qpos[:n_joints] = perturbed_joints
            mujoco.mj_forward(self.robot.model, self.robot.data)
            perturbed_ee_pos = self.robot.get_ee_position()

            jacobian[:, i] = (perturbed_ee_pos - current_ee_pos) / epsilon

        # 恢复原始状态
        self.robot.data.qpos[:n_joints] = current_joints
        mujoco.mj_forward(self.robot.model, self.robot.data)

        # 计算伪逆和关节增量
        jacobian_pinv = np.linalg.pinv(jacobian)
        joint_delta = jacobian_pinv @ pos_error
        joint_delta = np.clip(joint_delta, -0.1, 0.1)

        return joint_delta

    def _pose_to_matrix(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """将位置和四元数转换为4x4齐次变换矩阵"""
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def _get_observation(self) -> np.ndarray:
        """获取观测（与原环境相同）"""
        self.robot.update_joint_state()

        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        ee_pos = self.robot.get_ee_position()
        ee_vel = self.robot.get_ee_velocity()
        ee_quat = self.robot.get_ee_orientation()
        ee_angular_vel = self.robot.get_ee_angular_velocity()

        target_pos = self.robot.get_target_position()
        if target_pos is None:
            target_pos = self.target_pos

        target_quat = self.robot.get_target_orientation()
        if target_quat is None:
            target_quat = self.target_quat

        target_vel = self.robot.get_target_velocity()
        if target_vel is None:
            target_vel = np.zeros(3)

        obs1_pos = self.robot.get_obstacle_position('obstacle1')
        obs1_vel = self.robot.get_obstacle_velocity('obstacle1')
        obs2_pos = self.robot.get_obstacle_position('obstacle2')
        obs2_vel = self.robot.get_obstacle_velocity('obstacle2')

        if obs1_pos is None:
            obs1_pos = np.zeros(3)
        if obs1_vel is None:
            obs1_vel = np.zeros(3)
        if obs2_pos is None:
            obs2_pos = np.zeros(3)
        if obs2_vel is None:
            obs2_vel = np.zeros(3)

        observation = np.concatenate([
            joint_pos, joint_vel, ee_pos, ee_vel, ee_quat, ee_angular_vel,
            target_pos, target_quat, target_vel, obs1_pos, obs1_vel,
            obs2_pos, obs2_vel
        ])

        return observation

    def _compute_reward(self) -> float:
        """当前版本不再提供 RL reward，固定返回 0。"""
        return 0.0

    def _check_terminated(self) -> bool:
        """检查终止条件"""
        ee_pos = self.robot.get_ee_position()
        distance = np.linalg.norm(self.target_pos - ee_pos)

        if distance < self.target_reach_threshold:
            return True

        return False

    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """工作空间检查（已禁用，恒为True）"""
        return True

    def _generate_uniform_samples(self, n_samples: int) -> np.ndarray:
        """
        生成均匀分布的关节角度样本

        Args:
            n_samples: 需要生成的样本数量

        Returns:
            samples: 关节角度样本 (n_samples, 6)
        """
        if not hasattr(self, 'joint_limits'):
            self.joint_limits = self.robot.model.jnt_range[:6].copy()

        samples = []

        if self.sample_mode == 'single':
            # 单关节模式：只在joint1上均匀采样
            joint1_values = np.linspace(
                self.joint_limits[0, 0],
                self.joint_limits[0, 1],
                n_samples
            )
            for value in joint1_values:
                sample = np.zeros(6)
                sample[0] = value
                samples.append(sample)

        elif self.sample_mode == 'full':
            # 全关节模式：使用拉丁超立方采样（Latin Hypercube Sampling）
            # 这比纯网格采样更高效，能更好地覆盖高维空间

            # 为每个关节生成均匀分层的样本点
            for i in range(n_samples):
                sample = np.zeros(6)
                for j in range(6):
                    # 将[0,1]区间分成n_samples个子区间，在第i个子区间内随机采样
                    lower = i / n_samples
                    upper = (i + 1) / n_samples
                    random_in_interval = np.random.uniform(lower, upper)
                    # 映射到关节范围
                    sample[j] = self.joint_limits[j, 0] + random_in_interval * (
                        self.joint_limits[j, 1] - self.joint_limits[j, 0]
                    )
                samples.append(sample)

            # 随机打乱每个维度的顺序（拉丁超立方采样的关键步骤）
            samples = np.array(samples)
            for j in range(6):
                np.random.shuffle(samples[:, j])

        return np.array(samples) if isinstance(samples, list) else samples

    def _sample_target(self, max_attempts: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样目标位置

        根据 sample_mode 参数：
        - 'single': 只随机joint1，其他关节固定为0度（快速，适合平面运动）
        - 'full': 随机所有6个关节（完整3D空间采样）

        直接使用FK计算目标位置，跳过IK验证

        Args:
            max_attempts: 最大采样尝试次数

        Returns:
            target_pos: 目标位置 [x, y, z]
            target_quat: 目标姿态四元数 [w, x, y, z]
        """
        for _ in range(max_attempts):
            # 根据采样模式生成关节角度
            if self.sample_mode == 'single':
                # 只随机joint1，其他关节固定为0
                sampled_joints = np.zeros(6)
                sampled_joints[0] = np.random.uniform(
                    self.joint_limits[0, 0],  # joint1下限
                    self.joint_limits[0, 1]   # joint1上限
                )
            elif self.sample_mode == 'full':
                # 随机所有6个关节
                sampled_joints = np.zeros(6)
                for i in range(6):
                    sampled_joints[i] = np.random.uniform(
                        self.joint_limits[i, 0],  # 关节i下限
                        self.joint_limits[i, 1]   # 关节i上限
                    )
            else:
                raise ValueError(f"未知的采样模式: {self.sample_mode}, 必须是 'single' 或 'full'")

            # 保存原始关节状态
            original_joints = self._get_arm_joint_positions()

            # 设置采样的关节角度并执行正向运动学
            self._set_arm_joint_positions(sampled_joints)
            mujoco.mj_forward(self.robot.model, self.robot.data)

            # 获取FK计算的末端位置和姿态
            target_pos = self.robot.get_ee_position()
            target_quat = self.robot.get_ee_orientation()

            # 恢复原始关节状态
            self._set_arm_joint_positions(original_joints)
            mujoco.mj_forward(self.robot.model, self.robot.data)

            # 工作空间过滤已禁用，直接返回采样结果
            return target_pos, target_quat

        # 理论上不会到达这里，保留兜底
        target_pos = self.robot.get_ee_position()
        target_quat = self.robot.get_ee_orientation()
        return target_pos, target_quat

    def _update_target_visualization(self):
        """更新目标可视化"""
        target_body_id = mujoco.mj_name2id(
            self.robot.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_body"
        )

        if target_body_id != -1:
            mocap_id = self.robot.model.body_mocapid[target_body_id]
            if mocap_id != -1:
                self.robot.data.mocap_pos[mocap_id] = self.target_pos
                self.robot.data.mocap_quat[mocap_id] = self.target_quat
                mujoco.mj_forward(self.robot.model, self.robot.data)

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        ee_pos = self.robot.get_ee_position()
        distance_to_target = np.linalg.norm(self.target_pos - ee_pos)
        distance_info = self.robot.get_minimum_distance_to_obstacles()

        info = {
            'step': self.current_step,
            'distance_to_target': distance_to_target,
            'ee_position': ee_pos,
            'target_position': self.target_pos,
            'in_workspace': self._is_in_workspace(ee_pos),
            'min_obstacle_distance': distance_info['min_distance'],
            'nearest_obstacle': distance_info['obstacle_name'],
            'nearest_robot_geom': distance_info['robot_geom'],
        }

        return info

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco_viewer.launch_passive(self.robot.model, self.robot.data)
            self.viewer.sync()

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        print("✓ ArmEnv backend closed")

    # ========== 障碍物控制方法 ==========
    
    def enable_obstacles(self):
        """启用所有障碍物运动"""
        self.robot.enable_obstacles()

    def disable_obstacles(self):
        """禁用所有障碍物运动"""
        self.robot.disable_obstacles()

    def toggle_obstacles(self):
        """切换障碍物运动状态"""
        return self.robot.toggle_obstacles()

    def is_obstacles_enabled(self):
        """返回当前障碍物启用状态"""
        return self.robot.is_obstacles_enabled()

    def set_obstacle_frequency(self, obstacle_idx, frequency):
        """设置障碍物运动频率"""
        self.robot.set_obstacle_frequency(obstacle_idx, frequency)

    def set_obstacle_amplitude(self, obstacle_idx, amplitude):
        """设置障碍物运动幅度"""
        self.robot.set_obstacle_amplitude(obstacle_idx, amplitude)
