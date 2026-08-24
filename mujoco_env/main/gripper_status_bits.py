#!/usr/bin/python3
"""
夹爪状态 bitmask 共享定义（UInt32, 4 bytes）。

位语义：
    bit0: IDLE（夹爪空闲）
    bit1: GRASPING（夹爪闭合中）
    bit2: GRASPED（已稳定抓取）
    bit3: DROP_EVENT（掉落事件脉冲位）
    bit4: COLLISION（碰撞状态位）
    bit5: JOINT_LIMIT（关节越限状态位）
    bit6: PREMATURE_DROP_EVENT（未到目标区提前掉落事件）
    bit7: AT_TARGET_ZONE（物体到达固定目标区）
    bit8: LIFTED_FROM_PLANE（物体完全抬离抓取平面）
    bit9: PLACED_STABLE（物体放置后稳定，无滚动翻倒）
"""

from __future__ import annotations

# 状态位定义
GRIPPER_STATUS_IDLE_BIT = 0
GRIPPER_STATUS_GRASPING_BIT = 1
GRIPPER_STATUS_GRASPED_BIT = 2
GRIPPER_STATUS_DROP_EVENT_BIT = 3
GRIPPER_STATUS_COLLISION_BIT = 4
GRIPPER_STATUS_JOINT_LIMIT_BIT = 5
GRIPPER_STATUS_PREMATURE_DROP_EVENT_BIT = 6
GRIPPER_STATUS_AT_TARGET_ZONE_BIT = 7
GRIPPER_STATUS_LIFTED_FROM_PLANE_BIT = 8
GRIPPER_STATUS_PLACED_STABLE_BIT = 9


def set_status_bit(status_byte: int, bit_index: int) -> int:
    """置位：将 status_byte 的 bit_index 位设置为 1。"""
    return int(status_byte | (1 << bit_index))


def clear_status_bit(status_byte: int, bit_index: int) -> int:
    """清位：将 status_byte 的 bit_index 位清零。"""
    return int(status_byte & (~(1 << bit_index) & 0xFFFFFFFF))


def extract_status_bit(status_byte: int, bit_index: int) -> bool:
    """提取：读取 status_byte 的 bit_index 位。"""
    return bool((int(status_byte) >> bit_index) & 0x01)


def status_from_bits(status_bits: int) -> str:
    """从 bitmask 还原三态抓取状态。"""
    if extract_status_bit(status_bits, GRIPPER_STATUS_GRASPED_BIT):
        return "grasped"
    if extract_status_bit(status_bits, GRIPPER_STATUS_GRASPING_BIT):
        return "grasping"
    return "idle"


class GripperStatusEncoder:
    """
    夹爪状态编码器。

    触发规则：
        1) IDLE/GRASPING/GRASPED 三个状态位互斥，始终只置位一个
        2) DROP_EVENT 在 grasped -> 非grasped 的边沿触发
        3) DROP_EVENT 触发后保持 drop_latch_frames 个发布周期，避免单帧丢失
        4) PREMATURE_DROP_EVENT 在 grasped->非grasped 且未到目标区时触发
        5) COLLISION / JOINT_LIMIT 为电平位，只要当前成立就置位
        6) AT_TARGET_ZONE / LIFTED_FROM_PLANE / PLACED_STABLE 为电平位，只要当前成立就置位
    """

    def __init__(self, drop_latch_frames: int = 3):
        self.drop_latch_frames = max(1, int(drop_latch_frames))
        self._prev_grasped = False
        self._drop_latch_count = 0
        self._premature_drop_latch_count = 0

    def reset(self):
        """重置边沿检测内部状态。"""
        self._prev_grasped = False
        self._drop_latch_count = 0
        self._premature_drop_latch_count = 0

    def encode(
        self,
        status_str: str,
        collision_flag: bool,
        joint_limit_flag: bool,
        at_release_target_flag: bool,
        lifted_from_plane_flag: bool,
        placed_stable_flag: bool,
    ) -> int:
        """
        编码当前状态为 UInt32 bitmask。

        Args:
            status_str: "idle" | "grasping" | "grasped"
            collision_flag: 当前碰撞状态
            joint_limit_flag: 当前越限状态
            at_release_target_flag: 当前是否到达放置目标区
            lifted_from_plane_flag: 当前是否完全抬离抓取平面
            placed_stable_flag: 当前放置后是否稳定（低速且未翻倒）
        """
        status_byte = 0

        # 互斥主状态位
        if status_str == "grasped":
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_GRASPED_BIT)
        elif status_str == "grasping":
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_GRASPING_BIT)
        else:
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_IDLE_BIT)

        # 掉落事件位（边沿触发 + 短时锁存）
        current_grasped = (status_str == "grasped")
        if self._prev_grasped and (not current_grasped):
            self._drop_latch_count = self.drop_latch_frames
            if not at_release_target_flag:
                self._premature_drop_latch_count = self.drop_latch_frames
        if self._drop_latch_count > 0:
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_DROP_EVENT_BIT)
            self._drop_latch_count -= 1
        if self._premature_drop_latch_count > 0:
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_PREMATURE_DROP_EVENT_BIT)
            self._premature_drop_latch_count -= 1
        self._prev_grasped = current_grasped

        # 电平状态位
        if collision_flag:
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_COLLISION_BIT)
        if joint_limit_flag:
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_JOINT_LIMIT_BIT)
        if at_release_target_flag:
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_AT_TARGET_ZONE_BIT)
        if lifted_from_plane_flag:
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_LIFTED_FROM_PLANE_BIT)
        if placed_stable_flag:
            status_byte = set_status_bit(status_byte, GRIPPER_STATUS_PLACED_STABLE_BIT)

        return int(status_byte & 0xFFFFFFFF)
