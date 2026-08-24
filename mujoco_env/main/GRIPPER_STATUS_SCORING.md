# Gripper Status 位标准与评判规则

本文档定义 `/gripper_status`（`std_msgs/UInt32`）的统一口径，包括：
1. 各 bit 的触发标准；
2. 离线评估脚本如何使用这些 bit 判分。

## 1. 状态位定义（UInt32）

- bit0 `IDLE`：当前主状态为 `idle`
- bit1 `GRASPING`：当前主状态为 `grasping`
- bit2 `GRASPED`：当前主状态为 `grasped`
- bit3 `DROP_EVENT`：`grasped -> 非grasped` 的边沿事件（短时锁存）
- bit4 `COLLISION`：当前检测到碰撞（电平位）
- bit5 `JOINT_LIMIT`：当前检测到关节越限（电平位）
- bit6 `PREMATURE_DROP_EVENT`：未到目标区时发生掉落（边沿事件，短时锁存）
- bit7 `AT_TARGET_ZONE`：目标物体进入固定目标区（电平位）
- bit8 `LIFTED_FROM_PLANE`：目标物体完全抬离抓取平面（电平位）
- bit9 `PLACED_STABLE`：放置后稳定且无明显滚动翻倒（电平位）

## 2. 触发标准（主程序）

来源：`main/main.py` + `main/gripper_status_bits.py`。

### 2.1 主状态位（bit0~bit2）

三态互斥，每帧只会置位一个：
- `grasped`：夹爪闭合且左右手指与同一可抓取物体接触，接触法向力超过阈值；
- `grasping`：夹爪闭合但未满足 `grasped`；
- `idle`：其余情况。

### 2.2 掉落相关事件位（bit3/bit6）

- `DROP_EVENT`：检测到 `grasped -> 非grasped` 上升沿后置位，并锁存若干发布周期；
- `PREMATURE_DROP_EVENT`：若上述掉落发生时 `AT_TARGET_ZONE` 不成立，则同时置位并锁存。

### 2.3 目标区位（bit7）

设固定目标点为 `p_target=[x,y,z]`，目标物体质心位置为 `p_obj`，半径阈值为 `r_target`：

`||p_obj - p_target|| <= r_target` 时，`AT_TARGET_ZONE=1`。

### 2.4 抬离平面位（bit8）

设抓取平面高度 `z_plane`，净空阈值 `clearance`，物体底部高度为：

`z_bottom = z_obj - h_half`

当 `z_bottom >= z_plane + clearance`，`LIFTED_FROM_PLANE=1`。

### 2.5 放置稳定位（bit9）

满足以下条件并连续 `N` 帧后置位：
- 已在目标区（bit7 条件成立）；
- 已释放（当前非 `grasped`）；
- 已回到平面附近：`|z_bottom - z_plane| <= plane_tolerance`；
- 物体线速度低：`v = ||p_t - p_{t-1}|| / Δt <= v_max`；
- 倾角较小（无翻倒）：`θ = arccos(z_local · z_world) <= θ_max`；
- 且当前不处于抬离状态（bit8 条件不成立）。

## 3. 离线评估如何用这些位

来源：`main/trajectory_ros2_evaluator.py`。

### 3.1 事件计数与占比

- `bit` 事件数：按上升沿计数；
- `bit` 占比：`on_frame_count / total_frame_count`。

输出字段包括：
- `at_target_event_count` / `at_target_frame_ratio`
- `lifted_event_count` / `lifted_frame_ratio`
- `placed_stable_event_count` / `placed_stable_frame_ratio`
- `drop_count` / `premature_drop_count`

### 3.2 惩罚项

- 掉落：`drop_count * 30`
- 碰撞：`collision_count * 1`
- 越限：`joint_limit_count * 1`
- 剧烈抖动：`jitter_count * 10`

总惩罚：

`total_penalty = jitter*10 + drop*30 + collision*1 + joint_limit*1`

总分：

`overall_score = max(0, overall_raw - total_penalty)`

若发生掉落（`drop_count > 0`），封顶：`overall_score <= 40`。

### 3.3 基本技能成功率（40 分）映射

评估脚本新增 6 项布尔判定：
1. `stable_contact`：出现过连续 `MIN_GRASPED_CONSEC_FRAMES` 帧 `grasped`；
2. `lifted_from_plane`：`lifted_event_count > 0`；
3. `grasp_hold_stable`：满足 1 且无掉落且正常释放；
4. `no_drop`：`drop_count == 0`；
5. `placed_in_target_zone`：`at_target_event_count > 0`；
6. `placed_stable_no_rollover`：`placed_stable_event_count > 0`。

子分计算：

`basic_skill_score_40 = 40 * (命中项数 / 6)`

相关输出字段：
- `basic_skill_criteria`
- `basic_skill_hit_count`
- `basic_skill_score_40`

## 4. 任务效率（20分）

离线评估新增：
- 目标时间参数：`--efficiency_target_time_s`（默认 30s）
- 实际完成时间：
  - 起点：首帧时间；
  - 终点：首个 `bit9(PLACED_STABLE)` 置位时间；若未出现 bit9，则使用末帧时间。

按你的要求，先计算效率比：

`efficiency_ratio = operation_time_s / efficiency_target_time_s`

同时输出 20 分制效率分（用于汇总）：

`efficiency_score_20 = clamp(20 / efficiency_ratio, 0, 20)`

等价写法：

`efficiency_score_20 = clamp(20 * efficiency_target_time_s / operation_time_s, 0, 20)`

## 5. 参数建议

主程序可调参数：
- `--release-target-x/y/z`
- `--release-target-radius`
- `--grasp-plane-z`
- `--lift-clearance`
- `--placed-stable-min-frames`
- `--placed-stable-max-lin-vel`
- `--placed-stable-max-tilt-deg`
- `--placed-plane-tolerance`

建议先固定场景，再做阈值标定，避免不同场景阈值不可比。
