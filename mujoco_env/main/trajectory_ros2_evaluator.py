#!/usr/bin/python3
"""
机械臂任务轨迹评估器
====================
当前主入口为离线评估模式：读取 rosbag2 / mcap，输出评分结果。

离线读取的话题:
    /joint_states 或 /joint_states_sim (sensor_msgs/JointState) - 关节位置/速度/力矩
    /gripper_status (std_msgs/UInt32)                             - 夹爪状态bitmask，用于阶段划分与惩罚
                                                                   其中 bit6=提前掉落, bit7=到达目标区, bit8=抬离平面, bit9=放置稳定

任务阶段划分:
    Phase 1 - approach:   自动记录接近阶段
    Phase 2 - grasping:   gripper_status 变为 "grasping"              → 夹爪开始闭合
    Phase 3 - transport:  gripper_status 变为 "grasped"               → 搬运物体到目标点
    Phase 4 - release:    gripper_status 从 grasped/抓取中 变回 idle   → 放下物体并结束评估

评分公式:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  [每阶段内] 阶段分 = 0.40×平滑度分 + 0.35×稳定性分 + 0.25×力矩效率分    │
    │  [跨阶段]  总体分 = Σ( 阶段权重 × 阶段分 )  （各权重之和 = 1）           │
    │                                                                          │
    │  阶段权重（抓住物体后权重显著提高）:                                       │
    │      approach  (移动到目标): 0.10                                         │
    │      grasping  (抓取中):     0.15                                         │
    │      transport (搬运物体):   0.50   ← 已抓住，最关键                      │
    │      release   (放下物体):   0.25   ← 已抓住，次关键                      │
    └──────────────────────────────────────────────────────────────────────────┘

    子指标计算 (均为 0~100 分):
        平滑度分   = max(0, 100 × (1 - overall_jerk  / 150.0))
        稳定性分   = max(0, 100 × (1 - end_vel_norm  /   0.3))
        力矩效率分 = max(0, 100 × (1 - rms_utilization / 80%))

    参考量含义:
        overall_jerk    : 阶段内所有关节 jerk (Δvel/Δt) 的 RMS 均值
        end_vel_norm    : 阶段结束时刻的关节速度向量范数
        rms_utilization : 阶段内 RMS 力矩 / 力矩限制 的平均利用率

    注意: 若某阶段无数据，其权重按比例重新分配给有数据的阶段。
    惩罚项总公式:
        total_penalty =
            jitter_count * 10
          + drop_count * 30
          + collision_count * 1
          + joint_limit_count * 1

        overall_score = max(0, overall_raw - total_penalty)
        若 drop_count > 0，则 overall_score = min(overall_score, 40)

运行方式:
    # 读取 bag 目录或 mcap 所在目录
    /usr/bin/python3 main/trajectory_ros2_evaluator.py \
      --bag /path/to/rosbag_or_mcap_dir

    # 也可指定输出目录
    /usr/bin/python3 main/trajectory_ros2_evaluator.py \
      --bag /path/to/rosbag_or_mcap_dir \
      --output_dir /tmp/eval_results
"""

import sys
import os
import time
import json
import argparse
import threading
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum

from gripper_status_bits import (
    GRIPPER_STATUS_IDLE_BIT,
    GRIPPER_STATUS_GRASPING_BIT,
    GRIPPER_STATUS_GRASPED_BIT,
    GRIPPER_STATUS_DROP_EVENT_BIT,
    GRIPPER_STATUS_COLLISION_BIT,
    GRIPPER_STATUS_JOINT_LIMIT_BIT,
    GRIPPER_STATUS_PREMATURE_DROP_EVENT_BIT,
    GRIPPER_STATUS_AT_TARGET_ZONE_BIT,
    GRIPPER_STATUS_LIFTED_FROM_PLANE_BIT,
    GRIPPER_STATUS_PLACED_STABLE_BIT,
    extract_status_bit,
    status_from_bits,
)


# ==============================================================================
# 常量 & 配置
# ==============================================================================

JOINT_NAMES   = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
TORQUE_LIMITS = np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0])  # N·m
# 关节角位置限位（rad），用于“关节越限”惩罚统计
# 来源：robot_arm620_assets.xml 的 ctrlrange
JOINT_POSITION_LIMITS = np.array([
    [-2.9670,  2.9670],   # J1
    [-2.9670,  2.9670],   # J2
    [-1.5708,  1.5708],   # J3
    [-2.9670,  2.9670],   # J4
    [-1.5708,  1.5708],   # J5
    [-2.9670,  2.9670],   # J6
], dtype=np.float64)

# 评分参考值（根据实际机械臂调整）
REF_JERK_MAX       = 150.0   # rad/s³  超过此值 Jerk 评分归 0
REF_END_VEL_MAX    = 0.3     # rad/s   阶段末尾速度超过此值稳定性归 0
REF_EFFORT_MAX_PCT = 80.0    # %       力矩利用率超过此值效率归 0

# 抖动相关阈值（建议由“无抖动理想抓取轨迹”的 3σ 标定得到）
# 当前为默认值，可按机械臂型号与采样频率调整
REF_RMSE_MAX       = 0.12    # rad/s   速度 RMSE 参考上限（默认）
REF_STD_MAX        = 0.08    # rad/s   速度 STD 参考上限（默认）

# 赛题惩罚机制缩放：每次“剧烈抖动”扣分
SEVERE_JITTER_SCORE_THRESHOLD = 50.0   # 任一关节 Jerk 得分 < 50 触发
SEVERE_JITTER_PENALTY_POINTS  = 5.0   # 总分扣 5 分 / 次

# 掉落惩罚（中途由 grasped 失持）
DROP_PENALTY_POINTS = 10.0             # 总分扣 30 分 / 次
DROP_FAIL_SCORE_CAP = 80.0             # 发生掉落时总分上限

# 稳定性与安全性惩罚（文档要求）
COLLISION_PENALTY_POINTS = 1.0          # 与非目标物体或环境碰撞：-1 / 次
JOINT_LIMIT_PENALTY_POINTS = 1.0        # 关节越限/非安全姿态：-1 / 次
JOINT_LIMIT_EPS = 1e-3                  # 关节越限判定容差（rad）

# 基本技能成功率（40分）判定阈值
# /gripper_status 发布频率约 10Hz，默认连续 3 帧约等于 0.3s 的稳定接触窗口
MIN_GRASPED_CONSEC_FRAMES = 3

# target place 下圆形 pad 的半径（单位 m）
# env_worldbody.xml 中圆柱 pad 半径约 0.09
TARGET_PAD_RADIUS = 0.09

# 放置配对得分在总分中的占比（0~1）
PLACEMENT_SCORE_WEIGHT = 0.20

# 任务效率（20分）默认目标完成时间（秒）
DEFAULT_EFFICIENCY_TARGET_TIME_S = 30.0

# 阶段权重（平滑度 / 稳定性 / 力矩效率）
PHASE_WEIGHTS = {
    'smoothness': 0.40,
    'stability':  0.35,
    'effort':     0.25,
}

# 各阶段在总体评分中的权重（未抓物体权重低，抓住后权重高）
OVERALL_PHASE_WEIGHTS = {
    'approach':  0.10,   # 移动到目标，尚未抓取
    'grasping':  0.15,   # 夹爪闭合中，尚未抓住
    'transport': 0.50,   # 已抓住，搬运最关键
    'release':   0.25,   # 已抓住，放置次关键
}

PHASE_LABELS = {
    'approach':  'Phase 1 · 移动到目标',
    'grasping':  'Phase 2 · 抓取',
    'transport': 'Phase 3 · 搬运',
    'release':   'Phase 4 · 放下',
}

PHASE_ORDER = ['approach', 'grasping', 'transport', 'release']


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class Frame:
    """单步关节状态快照"""
    timestamp:  float
    positions:  np.ndarray   # (6,) rad
    velocities: np.ndarray   # (6,) rad/s
    efforts:    np.ndarray   # (6,) N·m
    phase:      str
    gripper_status: str      # idle | grasping | grasped
    status_bits: int         # /gripper_status 位掩码（0~2^32-1）
    is_grasped_flag: int     # 0/1，便于后处理筛选


@dataclass
class PhaseMetrics:
    """单阶段评价指标"""
    name:            str
    frame_count:     int   = 0
    duration:        float = 0.0   # s

    jerk_per_joint:  np.ndarray = field(default_factory=lambda: np.zeros(6))
    jerk_score_per_joint: np.ndarray = field(default_factory=lambda: np.zeros(6))
    overall_jerk:    float      = 0.0   # rad/s³
    severe_jitter:   bool       = False
    severe_jitter_count: int    = 0

    end_vel_norm:    float      = 0.0   # rad/s  阶段末尾 30 帧速度均值
    effort_rms:      np.ndarray = field(default_factory=lambda: np.zeros(6))
    effort_rms_mean: float      = 0.0   # 各关节 RMS 均值

    score_smoothness: float = 0.0  # 0~100
    score_stability:  float = 0.0
    score_effort:     float = 0.0
    score_total:      float = 0.0


# ==============================================================================
# 评估器核心逻辑
# ==============================================================================

class TrajectoryEvaluator:
    """
    纯计算类，不依赖 ROS2。
    接收帧列表，按阶段切片后计算指标并打印报告。
    """

    def __init__(
        self,
        efficiency_target_time_s: float = DEFAULT_EFFICIENCY_TARGET_TIME_S,
        placement_score_weight: float = PLACEMENT_SCORE_WEIGHT,
    ):
        self.efficiency_target_time_s = max(1e-6, float(efficiency_target_time_s))
        self.placement_score_weight = float(np.clip(placement_score_weight, 0.0, 1.0))

    def evaluate(self,
                 frames: List[Frame],
                 collision_count: int = 0,
                 target_place_eval: Optional[dict] = None) -> dict:
        """
        计算所有阶段的指标。

        Args:
            frames: 所有记录帧（按时间顺序）
            collision_count: 碰撞次数（来自碰撞话题离线统计）

        Returns:
            dict: phase_name -> PhaseMetrics，以及 'overall_score'
        """
        # 按阶段分组
        phase_frames: dict[str, List[Frame]] = {p: [] for p in PHASE_ORDER}
        for f in frames:
            if f.phase in phase_frames:
                phase_frames[f.phase].append(f)

        results = {}
        for phase_name in PHASE_ORDER:
            pf = phase_frames[phase_name]
            if len(pf) < 5:
                m = PhaseMetrics(name=phase_name, frame_count=len(pf))
                results[phase_name] = m
                continue
            results[phase_name] = self._compute_phase_metrics(phase_name, pf)

        # 综合评分：各阶段按 OVERALL_PHASE_WEIGHTS 加权求和
        # 若某阶段无数据，其权重按比例重新分配给有数据的阶段
        valid_weight_sum = sum(
            OVERALL_PHASE_WEIGHTS[p] for p in PHASE_ORDER
            if results[p].frame_count >= 5
        )
        if valid_weight_sum > 0:
            overall_raw = sum(
                OVERALL_PHASE_WEIGHTS[p] / valid_weight_sum * results[p].score_total
                for p in PHASE_ORDER
                if results[p].frame_count >= 5
            )
        else:
            overall_raw = 0.0

        # 赛题惩罚：任一关节 Jerk 得分 < 50 视为一次“剧烈抖动”（按阶段计次）
        severe_count = sum(
            1 for p in PHASE_ORDER
            if results[p].frame_count >= 5 and results[p].severe_jitter
        )
        jitter_penalty = severe_count * SEVERE_JITTER_PENALTY_POINTS

        # 掉落检测：
        # 1) gripper_status 从 grasped -> 非grasped 记为一次“失持”
        # 2) 最后一次失持且后续不再 grasped，视作正常 release
        # 3) 其余失持视为掉落（drop）
        drop_count, normal_release_detected = self._count_drop_events(frames)
        premature_drop_count = self._count_bit_events(frames, GRIPPER_STATUS_PREMATURE_DROP_EVENT_BIT)
        at_target_event_count = self._count_bit_events(frames, GRIPPER_STATUS_AT_TARGET_ZONE_BIT)
        lifted_event_count = self._count_bit_events(frames, GRIPPER_STATUS_LIFTED_FROM_PLANE_BIT)
        placed_stable_event_count = self._count_bit_events(frames, GRIPPER_STATUS_PLACED_STABLE_BIT)
        at_target_frame_ratio = self._bit_frame_ratio(frames, GRIPPER_STATUS_AT_TARGET_ZONE_BIT)
        lifted_frame_ratio = self._bit_frame_ratio(frames, GRIPPER_STATUS_LIFTED_FROM_PLANE_BIT)
        placed_stable_frame_ratio = self._bit_frame_ratio(frames, GRIPPER_STATUS_PLACED_STABLE_BIT)
        drop_penalty = drop_count * DROP_PENALTY_POINTS

        # 关节越限统计：优先使用状态位，缺失时从 JointState 位置数据估计
        joint_limit_count = self._count_bit_events(frames, GRIPPER_STATUS_JOINT_LIMIT_BIT)
        if joint_limit_count == 0:
            joint_limit_count = self._count_joint_limit_events(frames)
        joint_limit_penalty = joint_limit_count * JOINT_LIMIT_PENALTY_POINTS

        # 碰撞统计：优先使用状态位，其次使用外部碰撞话题计数
        bit_collision_count = self._count_bit_events(frames, GRIPPER_STATUS_COLLISION_BIT)
        collision_count = max(bit_collision_count, max(0, int(collision_count)))
        collision_penalty = collision_count * COLLISION_PENALTY_POINTS

        total_penalty = jitter_penalty + drop_penalty + joint_limit_penalty + collision_penalty
        overall = max(0.0, overall_raw - total_penalty)
        if drop_count > 0:
            overall = min(overall, DROP_FAIL_SCORE_CAP)

        # 任务效率（20分）：
        # 1) 先计算效率比 ratio = actual_time / target_time（按你的要求）
        # 2) 再换算为 20 分制：score = clamp(20 * target_time / actual_time, 0, 20)
        #    等价于 score = clamp(20 / ratio, 0, 20)
        operation_time_s = self._estimate_operation_time_s(frames)
        efficiency_ratio = float(operation_time_s / self.efficiency_target_time_s)
        efficiency_score_20 = float(np.clip(20.0 / max(efficiency_ratio, 1e-6), 0.0, 20.0))

        has_grasped = any(f.gripper_status == 'grasped' for f in frames)
        has_at_target = at_target_event_count > 0
        has_placed_stable = placed_stable_event_count > 0
        task_success = has_grasped and has_at_target and has_placed_stable and (drop_count == 0) and normal_release_detected

        # 基本技能 6 项判定（对应 40 分维度）
        crit1_stable_contact = self._has_consecutive_grasped(frames, MIN_GRASPED_CONSEC_FRAMES)
        crit2_lifted_from_plane = lifted_event_count > 0
        crit3_grasp_hold_stable = crit1_stable_contact and (drop_count == 0) and normal_release_detected
        crit4_no_drop = (drop_count == 0)
        crit5_placed_in_target_zone = has_at_target
        crit6_placed_stable_no_rollover = has_placed_stable
        basic_skill_hit_count = int(sum([
            crit1_stable_contact,
            crit2_lifted_from_plane,
            crit3_grasp_hold_stable,
            crit4_no_drop,
            crit5_placed_in_target_zone,
            crit6_placed_stable_no_rollover,
        ]))
        basic_skill_score_40 = float(40.0 * basic_skill_hit_count / 6.0)

        results['overall_score_raw'] = float(overall_raw)
        results['jitter_penalty_count'] = int(severe_count)
        results['jitter_penalty_points'] = float(jitter_penalty)
        results['drop_count'] = int(drop_count)
        results['premature_drop_count'] = int(premature_drop_count)
        results['at_target_event_count'] = int(at_target_event_count)
        results['lifted_event_count'] = int(lifted_event_count)
        results['placed_stable_event_count'] = int(placed_stable_event_count)
        results['at_target_frame_ratio'] = float(at_target_frame_ratio)
        results['lifted_frame_ratio'] = float(lifted_frame_ratio)
        results['placed_stable_frame_ratio'] = float(placed_stable_frame_ratio)
        results['drop_penalty_points'] = float(drop_penalty)
        results['normal_release_detected'] = bool(normal_release_detected)
        results['task_success'] = bool(task_success)
        results['collision_count'] = int(collision_count)
        results['collision_penalty_points'] = float(collision_penalty)
        results['joint_limit_count'] = int(joint_limit_count)
        results['joint_limit_penalty_points'] = float(joint_limit_penalty)
        results['total_penalty_points'] = float(total_penalty)
        results['overall_score'] = float(overall)
        results['efficiency_target_time_s'] = float(self.efficiency_target_time_s)
        results['operation_time_s'] = float(operation_time_s)
        results['efficiency_ratio'] = float(efficiency_ratio)
        results['efficiency_score_20'] = float(efficiency_score_20)

        results['basic_skill_criteria'] = {
            'stable_contact': bool(crit1_stable_contact),
            'lifted_from_plane': bool(crit2_lifted_from_plane),
            'grasp_hold_stable': bool(crit3_grasp_hold_stable),
            'no_drop': bool(crit4_no_drop),
            'placed_in_target_zone': bool(crit5_placed_in_target_zone),
            'placed_stable_no_rollover': bool(crit6_placed_stable_no_rollover),
        }
        results['basic_skill_hit_count'] = int(basic_skill_hit_count)
        results['basic_skill_score_40'] = float(basic_skill_score_40)
        results['target_place_eval'] = target_place_eval or {}

        # 放置配对规则得分：同时计算“同温配对”和“反序配对”
        placement_eval = target_place_eval or {}
        same_rule = placement_eval.get('same_rank_rule', {})
        reverse_rule = placement_eval.get('reverse_rank_rule', {})
        same_score = float(same_rule.get('score_100', 0.0))
        reverse_score = float(reverse_rule.get('score_100', 0.0))
        placement_score_100 = float(np.clip(0.5 * (same_score + reverse_score), 0.0, 100.0))
        overall_with_placement = float(
            np.clip(
                (1.0 - self.placement_score_weight) * overall + self.placement_score_weight * placement_score_100,
                0.0,
                100.0,
            )
        )
        results['placement_rule_score_same_100'] = same_score
        results['placement_rule_score_reverse_100'] = reverse_score
        results['placement_score_100'] = placement_score_100
        results['placement_score_weight'] = self.placement_score_weight
        results['overall_score_with_placement'] = overall_with_placement

        return results

    def _count_bit_events(self, frames: List[Frame], bit_index: int) -> int:
        """按上升沿统计某个状态位的触发次数。"""
        if not frames:
            return 0
        prev_on = False
        count = 0
        for f in frames:
            on = extract_status_bit(f.status_bits, bit_index)
            if on and (not prev_on):
                count += 1
            prev_on = on
        return int(count)

    def _bit_frame_ratio(self, frames: List[Frame], bit_index: int) -> float:
        """统计某状态位在全部帧中的占比。"""
        if not frames:
            return 0.0
        on_count = sum(1 for f in frames if extract_status_bit(f.status_bits, bit_index))
        return float(on_count / len(frames))

    def _has_consecutive_grasped(self, frames: List[Frame], min_frames: int) -> bool:
        """是否出现过连续 min_frames 帧的 grasped。"""
        if not frames:
            return False
        need = max(1, int(min_frames))
        run = 0
        for f in frames:
            if f.gripper_status == 'grasped':
                run += 1
                if run >= need:
                    return True
            else:
                run = 0
        return False

    def _estimate_operation_time_s(self, frames: List[Frame]) -> float:
        """估计完成操作耗时（秒）。

        规则：
            - 起点：首帧时间；
            - 终点：首个 bit9(PLACED_STABLE) 置位帧；
            - 若无 bit9，则退化为末帧时间。
        """
        if not frames:
            return 0.0
        t0 = float(frames[0].timestamp)
        t_end = float(frames[-1].timestamp)
        for f in frames:
            if extract_status_bit(f.status_bits, GRIPPER_STATUS_PLACED_STABLE_BIT):
                t_end = float(f.timestamp)
                break
        return float(max(0.0, t_end - t0))

    def _count_drop_events(self, frames: List[Frame]) -> tuple[int, bool]:
        """统计掉落次数并识别是否存在最终正常 release。"""
        if len(frames) < 2:
            return 0, False

        # 优先使用 premature_drop_event 位（未到目标区提前掉落）
        bit_premature_drop_count = self._count_bit_events(frames, GRIPPER_STATUS_PREMATURE_DROP_EVENT_BIT)
        bit_drop_count = self._count_bit_events(frames, GRIPPER_STATUS_DROP_EVENT_BIT)

        statuses = [f.gripper_status for f in frames]
        loss_indices = []
        for i in range(1, len(statuses)):
            if statuses[i - 1] == 'grasped' and statuses[i] != 'grasped':
                loss_indices.append(i)

        if not loss_indices:
            return int(max(bit_drop_count, bit_premature_drop_count)), False

        last_loss_idx = loss_indices[-1]
        # 若最后一次失持之后没有重新 grasped，则视为正常 release
        normal_release_detected = not any(s == 'grasped' for s in statuses[last_loss_idx + 1:])
        drop_count = len(loss_indices) - (1 if normal_release_detected else 0)
        drop_count = max(0, drop_count)
        # 优先采用“未到目标区提前掉落”状态位；若无则回退到一般掉落位/启发式
        if bit_premature_drop_count > 0:
            drop_count = bit_premature_drop_count
        else:
            drop_count = max(drop_count, bit_drop_count)
        return int(drop_count), normal_release_detected

    def _count_joint_limit_events(self, frames: List[Frame]) -> int:
        """统计关节越限事件次数（按进入越限状态的上升沿计次）。"""
        if not frames:
            return 0

        low = JOINT_POSITION_LIMITS[:, 0] - JOINT_LIMIT_EPS
        high = JOINT_POSITION_LIMITS[:, 1] + JOINT_LIMIT_EPS

        prev_violate = False
        count = 0
        for f in frames:
            q = np.asarray(f.positions, dtype=np.float64)
            violate = bool(np.any((q < low) | (q > high)))
            if violate and not prev_violate:
                count += 1
            prev_violate = violate
        return int(count)

    def _compute_phase_metrics(self, name: str, frames: List[Frame]) -> PhaseMetrics:
        N = len(frames)
        m = PhaseMetrics(name=name, frame_count=N)

        timestamps  = np.array([f.timestamp  for f in frames])
        velocities  = np.array([f.velocities for f in frames])   # (N,6)
        efforts     = np.array([f.efforts    for f in frames])   # (N,6)

        m.duration = float(timestamps[-1] - timestamps[0])

        # ── 平均控制周期
        dts = np.diff(timestamps)
        dt  = float(np.mean(dts)) if len(dts) > 0 else 0.002

        # ── Jerk（速度二阶差分 / dt²）
        if N > 2:
            jerk = np.diff(velocities, n=2, axis=0) / (dt ** 2)   # (N-2, 6)
            m.jerk_per_joint = np.mean(np.abs(jerk), axis=0)
            m.overall_jerk   = float(np.mean(np.abs(jerk)))
            m.jerk_score_per_joint = np.maximum(
                0.0,
                100.0 * (1.0 - m.jerk_per_joint / REF_JERK_MAX)
            )
            severe_mask = m.jerk_score_per_joint < SEVERE_JITTER_SCORE_THRESHOLD
            m.severe_jitter_count = int(np.sum(severe_mask))
            m.severe_jitter = m.severe_jitter_count > 0

        # ── 末尾稳定性（最后 min(30, N//5) 帧的速度范数均值）
        tail = max(5, min(30, N // 5))
        end_vels = velocities[-tail:]
        m.end_vel_norm = float(np.mean(np.linalg.norm(end_vels, axis=1)))

        # ── 力矩 RMS
        m.effort_rms      = np.sqrt(np.mean(efforts ** 2, axis=0))
        utilization       = m.effort_rms / TORQUE_LIMITS * 100.0  # %
        m.effort_rms_mean = float(np.mean(utilization))

        # ── 评分
        m.score_smoothness = max(0.0, 100.0 * (1.0 - m.overall_jerk / REF_JERK_MAX))
        m.score_stability  = max(0.0, 100.0 * (1.0 - m.end_vel_norm / REF_END_VEL_MAX))
        m.score_effort     = max(0.0, 100.0 * (1.0 - m.effort_rms_mean / REF_EFFORT_MAX_PCT))

        m.score_total = (
            PHASE_WEIGHTS['smoothness'] * m.score_smoothness
            + PHASE_WEIGHTS['stability']  * m.score_stability
            + PHASE_WEIGHTS['effort']     * m.score_effort
        )

        return m

    def report(self, results: dict, session_name: str = ""):
        """打印格式化评估报告"""
        sep = '=' * 68
        print(f'\n{sep}')
        print(f'  机械臂任务轨迹评估报告  {session_name}')
        print(sep)

        for phase_name in PHASE_ORDER:
            m: PhaseMetrics = results[phase_name]
            label = PHASE_LABELS[phase_name]

            print(f'\n  [{label}]')
            if m.frame_count < 5:
                print(f'    无数据（帧数: {m.frame_count}）')
                continue

            print(f'    时长: {m.duration:.2f} s    帧数: {m.frame_count}')

            # Jerk
            jerk_str = '  '.join(f'{v:.1f}' for v in m.jerk_per_joint)
            print(f'    Jerk /关节 (rad/s³): {jerk_str}')
            print(f'    综合 Jerk: {m.overall_jerk:.2f} rad/s³'
                  f'   →  平滑度评分: {m.score_smoothness:.1f}/100')
            jerk_score_str = '  '.join(f'{v:.1f}' for v in m.jerk_score_per_joint)
            print(f'    Jerk得分/关节: {jerk_score_str}')
            if m.severe_jitter:
                print(f'    ⚠ 剧烈抖动: 是（低于{SEVERE_JITTER_SCORE_THRESHOLD:.0f}分的关节数: {m.severe_jitter_count}）')
            else:
                print('    剧烈抖动: 否')

            # 稳定性
            print(f'    末尾速度范数: {m.end_vel_norm:.4f} rad/s'
                  f'   →  稳定性评分: {m.score_stability:.1f}/100')

            # 力矩
            rms_str = '  '.join(f'{v:.1f}' for v in m.effort_rms)
            print(f'    力矩 RMS (N·m): {rms_str}')
            print(f'    平均利用率: {m.effort_rms_mean:.1f}%'
                  f'   →  效率评分: {m.score_effort:.1f}/100')

            # 阶段分数
            w = OVERALL_PHASE_WEIGHTS[phase_name]
            bar_len = int(m.score_total / 100 * 30)
            bar = '[' + '#' * bar_len + '-' * (30 - bar_len) + ']'
            print(f'    阶段评分: {m.score_total:.1f}/100  {bar}  (总体权重 {w:.0%})')

        # 总分
        overall_raw = results.get('overall_score_raw', results.get('overall_score', 0.0))
        penalty_count = results.get('jitter_penalty_count', 0)
        penalty_points = results.get('jitter_penalty_points', 0.0)
        drop_count = results.get('drop_count', 0)
        premature_drop_count = results.get('premature_drop_count', 0)
        drop_penalty_points = results.get('drop_penalty_points', 0.0)
        at_target_event_count = results.get('at_target_event_count', 0)
        lifted_event_count = results.get('lifted_event_count', 0)
        placed_stable_event_count = results.get('placed_stable_event_count', 0)
        at_target_frame_ratio = results.get('at_target_frame_ratio', 0.0)
        lifted_frame_ratio = results.get('lifted_frame_ratio', 0.0)
        placed_stable_frame_ratio = results.get('placed_stable_frame_ratio', 0.0)
        collision_count = results.get('collision_count', 0)
        collision_penalty_points = results.get('collision_penalty_points', 0.0)
        joint_limit_count = results.get('joint_limit_count', 0)
        joint_limit_penalty_points = results.get('joint_limit_penalty_points', 0.0)
        total_penalty_points = results.get(
            'total_penalty_points',
            penalty_points + drop_penalty_points + collision_penalty_points + joint_limit_penalty_points
        )
        task_success = results.get('task_success', False)
        overall = results.get('overall_score', 0.0)
        print(f'\n  {"-" * 50}')
        print(f'  抖动惩罚: {penalty_count} 次  × {SEVERE_JITTER_PENALTY_POINTS:.0f} 分 = -{penalty_points:.1f} 分')
        print(f'  提前掉落事件: {premature_drop_count} 次（bit6）')
        print(f'  到达目标区事件: {at_target_event_count} 次（bit7）, 占比 {at_target_frame_ratio:.1%}')
        print(f'  抬离平面事件: {lifted_event_count} 次（bit8）, 占比 {lifted_frame_ratio:.1%}')
        print(f'  放置稳定事件: {placed_stable_event_count} 次（bit9）, 占比 {placed_stable_frame_ratio:.1%}')
        efficiency_target_time_s = results.get('efficiency_target_time_s', DEFAULT_EFFICIENCY_TARGET_TIME_S)
        operation_time_s = results.get('operation_time_s', 0.0)
        efficiency_ratio = results.get('efficiency_ratio', 0.0)
        efficiency_score_20 = results.get('efficiency_score_20', 0.0)
        print(f'  任务效率: 实际{operation_time_s:.2f}s / 目标{efficiency_target_time_s:.2f}s = {efficiency_ratio:.3f}')
        print(f'  任务效率(20分): {efficiency_score_20:.2f}/20')
        print(f'  掉落惩罚: {drop_count} 次  × {DROP_PENALTY_POINTS:.0f} 分 = -{drop_penalty_points:.1f} 分')
        print(f'  碰撞惩罚: {collision_count} 次  × {COLLISION_PENALTY_POINTS:.0f} 分 = -{collision_penalty_points:.1f} 分')
        print(f'  越限惩罚: {joint_limit_count} 次  × {JOINT_LIMIT_PENALTY_POINTS:.0f} 分 = -{joint_limit_penalty_points:.1f} 分')
        print(f'  总惩罚: -{total_penalty_points:.1f} 分')
        print(f'  惩罚前总分: {overall_raw:.1f}/100')
        bar_len = int(overall / 100 * 30)
        bar = '[' + '#' * bar_len + '-' * (30 - bar_len) + ']'
        if drop_count > 0:
            grade = '发生掉落，任务失败'
        elif overall >= 90:
            grade = '无抖动，稳定性优秀'
        elif overall >= 70:
            grade = '轻微抖动，稳定性良好'
        elif overall >= 50:
            grade = '中度抖动，稳定性一般'
        else:
            grade = '严重抖动，触发惩罚'
        print(f'  总体评分: {overall:.1f}/100  {bar}  {grade}')
        basic_skill_hit_count = results.get('basic_skill_hit_count', 0)
        basic_skill_score_40 = results.get('basic_skill_score_40', 0.0)
        basic_skill_criteria = results.get('basic_skill_criteria', {})
        target_place_eval = results.get('target_place_eval', {})
        print('  基本技能(40分):')
        print(f"    1) 稳定接触: {basic_skill_criteria.get('stable_contact', False)}")
        print(f"    2) 完全抬离: {basic_skill_criteria.get('lifted_from_plane', False)}")
        print(f"    3) 抓取保持稳定: {basic_skill_criteria.get('grasp_hold_stable', False)}")
        print(f"    4) 全程无掉落: {basic_skill_criteria.get('no_drop', False)}")
        print(f"    5) 放置到目标区: {basic_skill_criteria.get('placed_in_target_zone', False)}")
        print(f"    6) 放置后稳定: {basic_skill_criteria.get('placed_stable_no_rollover', False)}")
        print(f'    命中: {basic_skill_hit_count}/6  ->  {basic_skill_score_40:.1f}/40')
        if target_place_eval:
            print('  目标放置台中心判定（来自 /task_scene_snapshot）:')
            print(f"    快照帧数: {target_place_eval.get('snapshot_count', 0)}")
            print(f"    pad 半径: {target_place_eval.get('pad_radius')}")
            print(f"    pad 中心: {target_place_eval.get('pad_centers_xy')}")
            for name, status in target_place_eval.get('object_in_target_place', {}).items():
                print(f"    {name}: {status}")
            same_rule = target_place_eval.get('same_rank_rule', {})
            reverse_rule = target_place_eval.get('reverse_rank_rule', {})
            print('  温度配对规则得分:')
            print(f"    同温配对规则: {same_rule.get('matched_count', 0)}/{same_rule.get('total_count', 0)}"
                f" -> {same_rule.get('score_100', 0.0):.1f}/100")
            print(f"    反序配对规则: {reverse_rule.get('matched_count', 0)}/{reverse_rule.get('total_count', 0)}"
                f" -> {reverse_rule.get('score_100', 0.0):.1f}/100")
            print(f"    综合放置得分: {results.get('placement_score_100', 0.0):.1f}/100"
                f" (权重 {results.get('placement_score_weight', 0.0):.2f})")
            print(f"    融合后总分: {results.get('overall_score_with_placement', overall):.1f}/100")
        print(f'  任务判定: {"SUCCESS" if task_success else "FAIL"}')
        print(f'{sep}\n')

    def save_metrics(self, results: dict, path: str):
        """保存评估结果到 JSON"""
        out = {}
        for phase_name in PHASE_ORDER:
            m = results[phase_name]
            out[phase_name] = {
                'frame_count':      m.frame_count,
                'duration':         m.duration,
                'overall_jerk':     m.overall_jerk,
                'jerk_per_joint':   m.jerk_per_joint.tolist(),
                'jerk_score_per_joint': m.jerk_score_per_joint.tolist(),
                'severe_jitter':    m.severe_jitter,
                'severe_jitter_count': m.severe_jitter_count,
                'end_vel_norm':     m.end_vel_norm,
                'effort_rms':       m.effort_rms.tolist(),
                'effort_rms_mean':  m.effort_rms_mean,
                'score_smoothness': m.score_smoothness,
                'score_stability':  m.score_stability,
                'score_effort':     m.score_effort,
                'score_total':      m.score_total,
            }
        out['config'] = {
            'REF_JERK_MAX': REF_JERK_MAX,
            'REF_END_VEL_MAX': REF_END_VEL_MAX,
            'REF_EFFORT_MAX_PCT': REF_EFFORT_MAX_PCT,
            'REF_RMSE_MAX': REF_RMSE_MAX,
            'REF_STD_MAX': REF_STD_MAX,
            'SEVERE_JITTER_SCORE_THRESHOLD': SEVERE_JITTER_SCORE_THRESHOLD,
            'SEVERE_JITTER_PENALTY_POINTS': SEVERE_JITTER_PENALTY_POINTS,
            'DROP_PENALTY_POINTS': DROP_PENALTY_POINTS,
            'DROP_FAIL_SCORE_CAP': DROP_FAIL_SCORE_CAP,
            'COLLISION_PENALTY_POINTS': COLLISION_PENALTY_POINTS,
            'JOINT_LIMIT_PENALTY_POINTS': JOINT_LIMIT_PENALTY_POINTS,
            'JOINT_LIMIT_EPS': JOINT_LIMIT_EPS,
            'JOINT_POSITION_LIMITS': JOINT_POSITION_LIMITS.tolist(),
            'GRIPPER_STATUS_BITS': {
                'IDLE': GRIPPER_STATUS_IDLE_BIT,
                'GRASPING': GRIPPER_STATUS_GRASPING_BIT,
                'GRASPED': GRIPPER_STATUS_GRASPED_BIT,
                'DROP_EVENT': GRIPPER_STATUS_DROP_EVENT_BIT,
                'COLLISION': GRIPPER_STATUS_COLLISION_BIT,
                'JOINT_LIMIT': GRIPPER_STATUS_JOINT_LIMIT_BIT,
                'PREMATURE_DROP_EVENT': GRIPPER_STATUS_PREMATURE_DROP_EVENT_BIT,
                'AT_TARGET_ZONE': GRIPPER_STATUS_AT_TARGET_ZONE_BIT,
                'LIFTED_FROM_PLANE': GRIPPER_STATUS_LIFTED_FROM_PLANE_BIT,
                'PLACED_STABLE': GRIPPER_STATUS_PLACED_STABLE_BIT,
            },
        }
        out['overall_score_raw'] = results.get('overall_score_raw', results.get('overall_score', 0.0))
        out['jitter_penalty_count'] = results.get('jitter_penalty_count', 0)
        out['jitter_penalty_points'] = results.get('jitter_penalty_points', 0.0)
        out['drop_count'] = results.get('drop_count', 0)
        out['premature_drop_count'] = results.get('premature_drop_count', 0)
        out['at_target_event_count'] = results.get('at_target_event_count', 0)
        out['lifted_event_count'] = results.get('lifted_event_count', 0)
        out['placed_stable_event_count'] = results.get('placed_stable_event_count', 0)
        out['at_target_frame_ratio'] = results.get('at_target_frame_ratio', 0.0)
        out['lifted_frame_ratio'] = results.get('lifted_frame_ratio', 0.0)
        out['placed_stable_frame_ratio'] = results.get('placed_stable_frame_ratio', 0.0)
        out['drop_penalty_points'] = results.get('drop_penalty_points', 0.0)
        out['collision_count'] = results.get('collision_count', 0)
        out['collision_penalty_points'] = results.get('collision_penalty_points', 0.0)
        out['joint_limit_count'] = results.get('joint_limit_count', 0)
        out['joint_limit_penalty_points'] = results.get('joint_limit_penalty_points', 0.0)
        out['normal_release_detected'] = results.get('normal_release_detected', False)
        out['task_success'] = results.get('task_success', False)
        out['total_penalty_points'] = results.get('total_penalty_points', 0.0)
        out['overall_score'] = results.get('overall_score', 0.0)
        out['efficiency_target_time_s'] = results.get('efficiency_target_time_s', DEFAULT_EFFICIENCY_TARGET_TIME_S)
        out['operation_time_s'] = results.get('operation_time_s', 0.0)
        out['efficiency_ratio'] = results.get('efficiency_ratio', 0.0)
        out['efficiency_score_20'] = results.get('efficiency_score_20', 0.0)
        out['basic_skill_criteria'] = results.get('basic_skill_criteria', {})
        out['basic_skill_hit_count'] = results.get('basic_skill_hit_count', 0)
        out['basic_skill_score_40'] = results.get('basic_skill_score_40', 0.0)
        out['target_place_eval'] = results.get('target_place_eval', {})
        out['placement_rule_score_same_100'] = results.get('placement_rule_score_same_100', 0.0)
        out['placement_rule_score_reverse_100'] = results.get('placement_rule_score_reverse_100', 0.0)
        out['placement_score_100'] = results.get('placement_score_100', 0.0)
        out['placement_score_weight'] = results.get('placement_score_weight', self.placement_score_weight)
        out['overall_score_with_placement'] = results.get('overall_score_with_placement', out['overall_score'])

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'[Evaluator] 指标已保存: {path}')

    def save_raw_data(self, frames: List[Frame], path: str):
        """保存原始帧到 .npz"""
        if not frames:
            return
        grasped_flags = np.array([f.is_grasped_flag for f in frames], dtype=np.int32)
        np.savez_compressed(
            path,
            timestamps  = np.array([f.timestamp  for f in frames]),
            positions   = np.array([f.positions   for f in frames]),
            velocities  = np.array([f.velocities  for f in frames]),
            efforts     = np.array([f.efforts     for f in frames]),
            phases      = np.array([f.phase       for f in frames]),
            gripper_status = np.array([f.gripper_status for f in frames]),
            status_bits = np.array([f.status_bits for f in frames], dtype=np.uint32),
            is_grasped_flag = grasped_flags,
        )
        n_grasped = int(np.sum(grasped_flags))
        n_free = int(len(frames) - n_grasped)
        print(f'[Evaluator] 原始数据已保存: {path}  ({len(frames)} 帧)')
        print(f'[Evaluator] 轨迹标记统计: free_motion={n_free}, grasped_motion={n_grasped}')


# ==============================================================================
# ROS2 节点
# ==============================================================================

class PhaseState(str, Enum):
    IDLE      = 'idle'
    APPROACH  = 'approach'
    GRASPING  = 'grasping'
    TRANSPORT = 'transport'
    RELEASE   = 'release'
    DONE      = 'done'


# ==============================================================================
# 主函数
# ==============================================================================

def _resolve_bag_uri(input_path: str) -> str:
    p = os.path.abspath(input_path)
    if os.path.isdir(p):
        if os.path.exists(os.path.join(p, 'metadata.yaml')):
            return p
        raise RuntimeError(f'目录 {p} 中无 metadata.yaml，不是有效 rosbag 目录。')
    if os.path.isfile(p):
        if p.endswith(('.mcap', '.mca')):
            parent = os.path.dirname(p)
            if os.path.exists(os.path.join(parent, 'metadata.yaml')):
                return parent
            raise RuntimeError(f'文件 {p} 同目录缺少 metadata.yaml，无法读取。')
        raise RuntimeError(f'文件 {p} 扩展名非 .mcap/.mca，无法识别。')
    raise RuntimeError(f'无效路径（文件或目录不存在）：{p}')


def _parse_collision_value(msg) -> float:
    """从碰撞消息中提取数值。无法提取时返回 0。"""
    if hasattr(msg, 'data'):
        data = msg.data
        if isinstance(data, bool):
            return 1.0 if data else 0.0
        if isinstance(data, (int, float)):
            return float(data)
        s = str(data).strip().lower()
    else:
        s = str(msg).strip().lower()
        # 若是 Empty 等无字段消息，默认每条消息记 1 次事件
        if s:
            return 1.0
        return 0.0

    if s in ('true', 'collision', 'collide', 'hit'):
        return 1.0
    if s in ('false', 'none', 'no_collision', 'safe'):
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _count_collision_events_from_rows(collision_rows: List[tuple[float, float]]) -> int:
    """从碰撞消息序列估计碰撞次数。"""
    if not collision_rows:
        return 0

    vals = np.array([v for _, v in collision_rows], dtype=np.float64)

    # 二值状态：按上升沿计数（0->1）
    is_binary = np.all(np.logical_or(np.isclose(vals, 0.0), np.isclose(vals, 1.0)))
    if is_binary:
        prev = 0.0
        count = 0
        for v in vals:
            if v > 0.5 and prev <= 0.5:
                count += 1
            prev = v
        return int(count)

    # 非负单调：视为累计计数器
    diffs = np.diff(vals)
    if np.all(vals >= 0.0) and np.all(diffs >= -1e-9):
        return int(max(0.0, vals[-1] - vals[0]))

    # 兜底：每条正值消息记为 1 次
    return int(np.sum(vals > 0.0))


def _parse_task_scene_snapshot_msg(msg: Any) -> Optional[dict]:
    """解析 /task_scene_snapshot 的 JSON payload。"""
    if not hasattr(msg, 'data'):
        return None
    payload = str(msg.data).strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _evaluate_target_place_from_snapshots(snapshot_rows: List[tuple[float, dict]]) -> dict:
    """根据 snapshot 计算每个物体到各个圆形 pad 的距离。"""
    if not snapshot_rows:
        return {}

    latest_snapshot = snapshot_rows[-1][1]
    target_plates = latest_snapshot.get('target_plates', [])
    movable_objects = latest_snapshot.get('movable_objects', [])

    if len(target_plates) < 3:
        return {
            'snapshot_count': len(snapshot_rows),
            'error': 'target_plates 数据不足（需要至少3个点）',
        }

    try:
        pad_centers = {
            str(item.get('body_name', f'pad_{idx}')): np.asarray(item['position_xyz'][:2], dtype=np.float64)
            for idx, item in enumerate(target_plates)
        }
    except Exception:
        return {
            'snapshot_count': len(snapshot_rows),
            'error': 'target_plates 缺少 position_xyz',
        }

    obj_status = {}
    obj_distance_table = {}
    obj_temp_table = {}
    pad_temp_table = {}
    for item in movable_objects:
        name = str(item.get('body_name', 'unknown'))
        pos = item.get('position_xyz', None)
        obj_temp = item.get('temperature_c', None)
        obj_temp_table[name] = (float(obj_temp) if isinstance(obj_temp, (int, float)) else None)
        if not isinstance(pos, list) or len(pos) < 2:
            obj_status[name] = None
            obj_distance_table[name] = {}
            continue
        obj_xy = np.asarray(pos[:2], dtype=np.float64)

        dist_map = {
            pad_name: float(np.linalg.norm(obj_xy - pad_xy))
            for pad_name, pad_xy in pad_centers.items()
        }
        obj_distance_table[name] = dist_map
        if not dist_map:
            obj_status[name] = None
            continue

        nearest_pad_name = min(dist_map, key=dist_map.get)
        nearest_pad_dist = float(dist_map[nearest_pad_name])
        obj_status[name] = {
            'nearest_pad': nearest_pad_name,
            'nearest_pad_distance': nearest_pad_dist,
            'in_nearest_pad': bool(nearest_pad_dist <= TARGET_PAD_RADIUS),
            'in_any_pad': bool(any(d <= TARGET_PAD_RADIUS for d in dist_map.values())),
        }

    for item in target_plates:
        pad_name = str(item.get('body_name', 'unknown_pad'))
        pad_temp = item.get('temperature_c', None)
        pad_temp_table[pad_name] = (float(pad_temp) if isinstance(pad_temp, (int, float)) else None)

    # 规则1: 同温排序配对（高->高, 中->中, 低->低）
    # 规则2: 反序排序配对（高->低, 中->中, 低->高）
    valid_obj = [(k, v) for k, v in obj_temp_table.items() if v is not None and k in obj_status and obj_status[k] is not None]
    valid_pad = [(k, v) for k, v in pad_temp_table.items() if v is not None]

    same_rule = {
        'matched_count': 0,
        'total_count': 0,
        'score_100': 0.0,
        'expected_map': {},
        'actual_map': {},
        'per_object_match': {},
    }
    reverse_rule = {
        'matched_count': 0,
        'total_count': 0,
        'score_100': 0.0,
        'expected_map': {},
        'actual_map': {},
        'per_object_match': {},
    }

    if len(valid_obj) >= 3 and len(valid_pad) >= 3:
        obj_rank = sorted(valid_obj, key=lambda kv: (-kv[1], kv[0]))
        pad_rank = sorted(valid_pad, key=lambda kv: (-kv[1], kv[0]))
        n = min(len(obj_rank), len(pad_rank))

        same_expected = {obj_rank[i][0]: pad_rank[i][0] for i in range(n)}
        reverse_expected = {obj_rank[i][0]: pad_rank[n - 1 - i][0] for i in range(n)}

        same_rule['expected_map'] = same_expected
        reverse_rule['expected_map'] = reverse_expected

        matched_same = 0
        matched_reverse = 0
        for obj_name, _ in obj_rank[:n]:
            nearest = obj_status.get(obj_name, {}).get('nearest_pad')
            same_rule['actual_map'][obj_name] = nearest
            reverse_rule['actual_map'][obj_name] = nearest

            same_ok = bool(nearest == same_expected.get(obj_name))
            reverse_ok = bool(nearest == reverse_expected.get(obj_name))
            same_rule['per_object_match'][obj_name] = same_ok
            reverse_rule['per_object_match'][obj_name] = reverse_ok
            matched_same += int(same_ok)
            matched_reverse += int(reverse_ok)

        same_rule['matched_count'] = matched_same
        same_rule['total_count'] = n
        same_rule['score_100'] = float(100.0 * matched_same / n) if n > 0 else 0.0

        reverse_rule['matched_count'] = matched_reverse
        reverse_rule['total_count'] = n
        reverse_rule['score_100'] = float(100.0 * matched_reverse / n) if n > 0 else 0.0

    return {
        'snapshot_count': len(snapshot_rows),
        'pad_radius': TARGET_PAD_RADIUS,
        'pad_centers_xy': {
            k: [float(v[0]), float(v[1])] for k, v in pad_centers.items()
        },
        'object_to_pad_distance': obj_distance_table,
        'object_in_target_place': obj_status,
        'object_temperatures': obj_temp_table,
        'pad_temperatures': pad_temp_table,
        'same_rank_rule': same_rule,
        'reverse_rank_rule': reverse_rule,
        'snapshot_time_s': float(snapshot_rows[-1][0]),
    }


def _load_frames_from_bag(bag_uri: str,
                          joint_topic_candidates: List[str],
                          gripper_topic: str,
                          collision_topic: Optional[str] = None,
                          snapshot_topic: Optional[str] = None) -> tuple[List[Frame], str, str, int, str, dict, str]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        print(f"[ERROR] rosbag2_py 导入失败: {e}")
        print("离线读取 rosbag 需要 ROS2 环境，请运行:")
        print("  source /opt/ros/humble/setup.bash")
        raise

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    
    print(f"[Bag Info] Metadata 话题: {list(type_map.keys())}", file=sys.stderr)

    # 先扫描实际 bag 中的话题（可能被重命名）
    actual_topics = set()
    temp_reader = rosbag2_py.SequentialReader()
    temp_reader.open(storage_options, converter_options)
    scan_count = 0
    while temp_reader.has_next() and scan_count < 100:
        topic, _, _ = temp_reader.read_next()
        actual_topics.add(topic)
        scan_count += 1
    
    print(f"[Bag Info] 实际话题（扫描前100条）: {sorted(actual_topics)}", file=sys.stderr)

    # 优先使用实际话题匹配
    joint_topic = None
    for t in joint_topic_candidates:
        if t in actual_topics:
            joint_topic = t
            break
    if joint_topic is None:
        raise RuntimeError(f'未找到关节话题，候选: {joint_topic_candidates}，实际: {sorted(actual_topics)}')
    if gripper_topic not in actual_topics:
        print(f"[Warn] Gripper 话题 {gripper_topic} 不在实际话题中，尝试继续...", file=sys.stderr)
    if collision_topic and (collision_topic not in actual_topics):
        print(f"[Warn] Collision 话题 {collision_topic} 不在实际话题中，将按 0 次碰撞处理。", file=sys.stderr)
    if snapshot_topic and (snapshot_topic not in actual_topics):
        print(f"[Warn] Snapshot 话题 {snapshot_topic} 不在实际话题中，将跳过 target_place 判定。", file=sys.stderr)

    # 建立实际话题到 type 的映射（处理重命名）
    # 尝试从 metadata 推断：找同类型话题
    actual_type_map = {}
    for actual in actual_topics:
        # 直接匹配
        if actual in type_map:
            actual_type_map[actual] = type_map[actual]
        else:
            # 尝试匹配 JointState 类型（关节话题）
            for meta_topic, meta_type in type_map.items():
                if 'JointState' in meta_type and 'joint' in actual.lower():
                    actual_type_map[actual] = meta_type
                    break
                elif actual == gripper_topic and ('UInt32' in meta_type):
                    actual_type_map[actual] = meta_type
                    break
                elif collision_topic and actual == collision_topic:
                    actual_type_map[actual] = meta_type
                    break
                elif snapshot_topic and actual == snapshot_topic and ('String' in meta_type):
                    actual_type_map[actual] = meta_type
                    break

    print(f"[Bag Info] 实际话题类型映射: {actual_type_map}", file=sys.stderr)

    msg_cache = {}

    def decode(topic: str, payload: bytes):
        # 懒加载类型：即使扫描前100条未覆盖到某话题，也可在此按 metadata 回退
        if topic not in actual_type_map:
            if topic in type_map:
                actual_type_map[topic] = type_map[topic]
            elif topic == gripper_topic:
                # gripper 回退：优先找 UInt32
                for _mt, _ty in type_map.items():
                    if 'UInt32' in _ty:
                        actual_type_map[topic] = _ty
                        break
            elif topic == joint_topic:
                for _mt, _ty in type_map.items():
                    if 'JointState' in _ty:
                        actual_type_map[topic] = _ty
                        break
            elif collision_topic and topic == collision_topic:
                # collision 类型不固定，兜底取 metadata 中任一匹配话题
                if topic in type_map:
                    actual_type_map[topic] = type_map[topic]
            elif snapshot_topic and topic == snapshot_topic:
                for _mt, _ty in type_map.items():
                    if 'String' in _ty:
                        actual_type_map[topic] = _ty
                        break

        if topic not in actual_type_map:
            raise KeyError(f'话题 {topic} 无类型信息')
        if topic not in msg_cache:
            msg_cache[topic] = get_message(actual_type_map[topic])
        return deserialize_message(payload, msg_cache[topic])

    joint_rows = []       # (t, pos, vel, eff)
    gripper_rows = []     # (t, status, bits)
    collision_rows = []   # (t, value)
    snapshot_rows = []    # (t, snapshot_dict)
    msg_count = {joint_topic: 0, gripper_topic: 0}
    if collision_topic:
        msg_count[collision_topic] = 0
    if snapshot_topic:
        msg_count[snapshot_topic] = 0

    while reader.has_next():
        topic, payload, t_ns = reader.read_next()
        t = float(t_ns) * 1e-9
        
        # 调试：打印前几个话题以检查匹配
        if msg_count[joint_topic] == 0 and msg_count[gripper_topic] < 3:
            print(f"[Debug] 读取话题 '{topic}', joint_topic='{joint_topic}', match={topic==joint_topic}", file=sys.stderr)

        if topic == joint_topic:
            try:
                msg = decode(topic, payload)
                n = min(6, len(msg.position))
                pos = np.array(list(msg.position[:n]) + [0.0] * (6 - n), dtype=np.float64)
                vel = np.array(list(msg.velocity[:n]) + [0.0] * (6 - n), dtype=np.float64) if len(msg.velocity) else np.zeros(6)
                eff = np.array(list(msg.effort[:n]) + [0.0] * (6 - n), dtype=np.float64) if len(msg.effort) else np.zeros(6)
                joint_rows.append((t, pos, vel, eff))
                msg_count[joint_topic] += 1
            except Exception as e:
                print(f"[Warn] 解码 {joint_topic} 失败: {e}", file=sys.stderr)
        elif topic == gripper_topic:
            try:
                msg = decode(topic, payload)
                bits = int(msg.data) & 0xFFFFFFFF
                s = status_from_bits(bits)
                gripper_rows.append((t, s, bits))
                msg_count[gripper_topic] += 1
            except Exception as e:
                print(f"[Warn] 解码 {gripper_topic} 失败: {e}", file=sys.stderr)
        elif collision_topic and topic == collision_topic:
            try:
                msg = decode(topic, payload)
                v = _parse_collision_value(msg)
                collision_rows.append((t, float(v)))
                msg_count[collision_topic] += 1
            except Exception as e:
                print(f"[Warn] 解码 {collision_topic} 失败: {e}", file=sys.stderr)
        elif snapshot_topic and topic == snapshot_topic:
            try:
                msg = decode(topic, payload)
                snapshot_data = _parse_task_scene_snapshot_msg(msg)
                if snapshot_data is not None:
                    snapshot_rows.append((t, snapshot_data))
                    msg_count[snapshot_topic] += 1
            except Exception as e:
                print(f"[Warn] 解码 {snapshot_topic} 失败: {e}", file=sys.stderr)

    collision_msg_num = msg_count.get(collision_topic, 0) if collision_topic else 0
    snapshot_msg_num = msg_count.get(snapshot_topic, 0) if snapshot_topic else 0
    print(
        f"[Bag Info] 读取到 {msg_count[joint_topic]} 条关节消息，{msg_count[gripper_topic]} 条 gripper 消息，"
        f"{collision_msg_num} 条 collision 消息，{snapshot_msg_num} 条 snapshot 消息",
        file=sys.stderr
    )

    if not joint_rows:
        raise RuntimeError(f'bag 中无关节数据，无法评分。(尝试了话题: {joint_topic})')
    if not gripper_rows:
        gripper_rows = [(joint_rows[0][0], 'idle', 1 << GRIPPER_STATUS_IDLE_BIT)]

    joint_rows.sort(key=lambda x: x[0])
    gripper_rows.sort(key=lambda x: x[0])

    frames: List[Frame] = []
    gi = 0
    current_status = gripper_rows[0][1]
    current_bits = int(gripper_rows[0][2])
    seen_grasped = False

    for t, pos, vel, eff in joint_rows:
        while gi + 1 < len(gripper_rows) and gripper_rows[gi + 1][0] <= t:
            gi += 1
            current_status = gripper_rows[gi][1]
            current_bits = int(gripper_rows[gi][2])

        if current_status == 'grasped':
            phase = PhaseState.TRANSPORT.value
            seen_grasped = True
        elif current_status == 'grasping':
            phase = PhaseState.GRASPING.value
        else:
            phase = PhaseState.RELEASE.value if seen_grasped else PhaseState.APPROACH.value

        frames.append(Frame(
            timestamp=t,
            positions=pos,
            velocities=vel,
            efforts=eff,
            phase=phase,
            gripper_status=current_status,
            status_bits=current_bits,
            is_grasped_flag=1 if current_status == 'grasped' else 0,
        ))

    collision_count = _count_collision_events_from_rows(collision_rows) if collision_topic else 0
    collision_topic_used = collision_topic if collision_topic else ''
    target_place_eval = _evaluate_target_place_from_snapshots(snapshot_rows)
    snapshot_topic_used = snapshot_topic if (snapshot_topic and snapshot_msg_num > 0) else ''
    return frames, joint_topic, gripper_topic, collision_count, collision_topic_used, target_place_eval, snapshot_topic_used

def main():
    parser = argparse.ArgumentParser(description='机械臂轨迹离线评估器（读取 rosbag2/mcap）')
    parser.add_argument('--bag', type=str, required=True,
                        help='rosbag 目录，或其中 .mcap 文件路径')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='评估结果输出目录 (默认: ./eval_results)')
    parser.add_argument('--joint_topics', type=str, default='/joint_states,/joint_states_sim',
                        help='关节话题候选，逗号分隔（优先使用排在前面的）')
    parser.add_argument('--gripper_topic', type=str, default='/gripper_status',
                        help='gripper 状态话题')
    parser.add_argument('--collision_topic', type=str, default='',
                        help='可选，碰撞事件话题（支持 Bool/数值/String）')
    parser.add_argument('--snapshot_topic', type=str, default='/task_scene_snapshot',
                        help='可选，任务快照话题（std_msgs/String JSON）')
    parser.add_argument('--efficiency_target_time_s', type=float, default=DEFAULT_EFFICIENCY_TARGET_TIME_S,
                        help='任务效率目标时间(秒)，用于计算 actual/target 与 20分制效率分')
    parser.add_argument('--placement_score_weight', type=float, default=PLACEMENT_SCORE_WEIGHT,
                        help='放置配对得分占总分权重(0~1)，默认0.20')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bag_uri = _resolve_bag_uri(args.bag)
    joint_candidates = [x.strip() for x in args.joint_topics.split(',') if x.strip()]

    frames, joint_topic, gripper_topic, collision_count, collision_topic_used, target_place_eval, snapshot_topic_used = _load_frames_from_bag(
        bag_uri=bag_uri,
        joint_topic_candidates=joint_candidates,
        gripper_topic=args.gripper_topic,
        collision_topic=args.collision_topic.strip() or None,
        snapshot_topic=args.snapshot_topic.strip() or None,
    )

    print('=' * 60)
    print('[Evaluator] 离线评估模式')
    print(f'[Evaluator] Bag目录: {bag_uri}')
    print(f'[Evaluator] Joint话题: {joint_topic}')
    print(f'[Evaluator] Gripper话题: {gripper_topic}')
    print(f'[Evaluator] Collision话题: {collision_topic_used if collision_topic_used else "(未启用)"}')
    print(f'[Evaluator] Snapshot话题: {snapshot_topic_used if snapshot_topic_used else "(未启用/无数据)"}')
    print(f'[Evaluator] 统计碰撞次数: {collision_count}')
    print(f'[Evaluator] 读取帧数: {len(frames)}')
    print('=' * 60)

    evaluator = TrajectoryEvaluator(
        efficiency_target_time_s=args.efficiency_target_time_s,
        placement_score_weight=args.placement_score_weight,
    )
    results = evaluator.evaluate(
        frames,
        collision_count=collision_count,
        target_place_eval=target_place_eval,
    )

    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    evaluator.report(results, session_name=session)

    metrics_path = os.path.join(args.output_dir, f'metrics_{session}.json')
    raw_path = os.path.join(args.output_dir, f'raw_{session}.npz')
    evaluator.save_metrics(results, metrics_path)
    evaluator.save_raw_data(frames, raw_path)


if __name__ == '__main__':
    main()
