# 机械臂仿真比赛说明（选手版）

## 1. 文档目的

本文档面向比赛选手，说明以下内容：

- 比赛任务目标与完成标准
- 场景与机械臂模型定义位置
- 控制接口（输入话题）与可观测状态（输出话题）
- 录制与提交数据的建议流程
- 评分口径（规则说明）

说明：

- 本文档不提供离线评估脚本的使用教程。
- 评估脚本属于内部裁判工具，不会公布。

---

## 2. 核心任务定义

### 2.1 任务目标

你需要控制机械臂完成一次完整的抓取-搬运-放置流程：

1. 接近并抓取可操作容器物体。
2. 将物体搬运到目标放置区域（三个圆形目标盘）。
3. 根据温度规则完成放置。
4. 放置后物体应稳定，无掉落、无明显翻倒。

### 2.2 温度放置规则（当前比赛定义）

每次仿真启动时：

- 三个可搬运物体会被赋予随机温度、材质、位置。
- 三个目标盘也会被赋予随机温度。
- 启动日志会打印物体温度和目标盘温度。

当前有效判定规则为反序温度匹配：

- 高温物体 -> 低温目标盘
- 中温物体 -> 中温目标盘
- 低温物体 -> 高温目标盘

补充说明：

- 圆盘颜色仅作视觉参考，不代表固定温度标签。
- 温度是每次运行随机初始化的，策略应依赖实时信息，而不是记忆固定颜色映射。

---

## 3. 模型与场景定义位置

### 3.1 主程序默认加载模型

主程序入口：

- mujoco_env/main/main.py

默认模型文件：

- mujoco_env/robot_model/exp/env_robot_torque_tactile.xml

### 3.2 机械臂与夹爪定义

机械臂本体与关节链路（含触觉版夹爪挂载）：

- mujoco_env/robot_model/robot/robot_arm/robot_arm620_robot_iq_tactile_sensor.xml

机械臂驱动与关节力矩传感器：

- mujoco_env/robot_model/robot/robot_arm/robot_arm620_torque.xml

### 3.3 场景物体定义

实验室环境、桌面、目标盘和可操作容器等主要在：

- mujoco_env/robot_model/env/env_worldbody.xml

其中可操作目标物体包含：

- beaker1
- graduated_cylinder
- erlenmeyer_flask

当前这三个物体在场景中显示为简化实验器皿：

- `beaker1`：烧杯外观，包含透明玻璃和蓝色液体
- `graduated_cylinder`：量筒外观，包含透明筒体、底座和黄色液体
- `erlenmeyer_flask`：锥形瓶外观，包含透明瓶身、瓶颈和紫色液体

目标放置盘为：

- target_place_table_1
- target_place_table_2
- target_place_table_3

### 3.4 场景随机化说明

每次 `episode/reset` 时，场景中的任务相关要素会做随机化。选手不应假设固定的物体位置、材质参数或温度映射。

当前会随机化的内容如下：

- 可抓取物体位置随机化
  - 对象：
    - `beaker1`
    - `graduated_cylinder`
    - `erlenmeyer_flask`
  - 随机项目：
    - 平面位置 `x/y`
    - 采样中心：`[1.1, 0.3]`
    - 采样区域：中心点附近 `30cm x 30cm`
    - 采样公式：`x = x0 + U(-0.15, 0.15)`，`y = y0 + U(-0.15, 0.15)`
    - 高度 `z`
    - 物体之间最小间距约束
    - 避开目标放置台区域
    - 保证采样位置仍在主工作台台面上

- 可抓取物体材质与接触参数随机化
  - 对象：
    - `beaker1`
    - `graduated_cylinder`
    - `erlenmeyer_flask`
  - 候选材质：
    - `metal_mat`
    - `wood_mat`
    - `glass_mat`
  - 随机项目：
    - 材质 `material`
    - 摩擦参数 `friction = [slide, torsional, rolling]`
    - 接触恢复参数 `solref`
    - 接触阻抗参数 `solimp`

- 温度随机化
  - 对象：
    - `beaker1`
    - `graduated_cylinder`
    - `erlenmeyer_flask`
    - `target_place_table_1`
    - `target_place_table_2`
    - `target_place_table_3`
  - 随机项目：
    - 三个可抓取物体的液体温度
    - 三个目标放置盘的温度
    - 物体温度与目标盘温度之间的对应关系

补充说明：

- 当前主要随机化的是任务相关物体，而不是整套实验室布局。
- 机器人本体、外部相机、实验台等基础场景默认不做随机位姿变化。

### 3.5 触觉可视化相关

触觉可视化窗口脚本：

- mujoco_env/sensors/tactile_visualizer.py

启动主程序时可加参数启用触觉 UI：

- --tactile-ui

补充说明：

- `/ee_target` 对应的 mocap body 仍保留，用于目标位姿逻辑。
- 当前 MuJoCo viewer 中暂时隐藏 `ee_pose_target_site` 球体和三根方向轴，不再显示橙色目标球。

---

## 4. 仿真启动与运行

在项目根目录执行：

    source /opt/ros/humble/setup.bash
    cd mujoco_env
    /usr/bin/python3 main/main.py

或者直接在仓库根目录执行：

    ./run_system_python.sh

可选：启用触觉力独立可视化窗口：

    /usr/bin/python3 main/main.py --tactile-ui

说明：

- 默认启用 ROS 控制输入订阅。
- 默认支持 teaching 数据记录与 rosbag 录制触发机制。
- 当前仓库建议统一使用系统 Python：`/usr/bin/python3`
- 如果当前终端里的 `python3` 指向 conda，请不要直接用 `python3 main/main.py`

### 4.1 主程序启动参数

主程序完整入口如下：

    /usr/bin/python3 mujoco_env/main/main.py [options]

当前 `main.py` 支持的命令行参数如下。

#### 路径与运行模式

- `--model`
  - 类型：`str`
  - 默认值：`None`
  - 含义：指定 MuJoCo 场景 XML 路径；不传时使用主程序内部默认模型。

- `--headless`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：无可视化模式运行；传入后不启动 GUI。

#### ROS 与录制控制

- `--no-ros-control`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：禁用 ROS 控制输入订阅。

- `--no-rosbag`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：禁用 rosbag 录制。

- `--no-topic-rename`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：禁用 rosbag 录制完成后的话题重命名步骤。

#### 图像、热成像与触觉相关

- `--no-image-publish`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：禁用 RGBD ROS2 发布。

- `--no-depth-render`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：禁用 RGBD 相机深度渲染。

- `--no-thermal`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：禁用热成像发布。

- `--tactile-ui`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：启用独立触觉可视化窗口，显示热力图和切向箭头。

#### 调试与日志

- `--print-task-snapshot-info`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：在终端打印实时任务快照信息（Live Task Snapshot）。

- `--profile-logs`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：输出 Render / Dispatch / Thermal profile 的 `info` 级日志。

#### 频率参数

- `--control-loop-hz`
  - 类型：`float`
  - 默认值：`15.0`
  - 含义：主循环频率，单位 Hz。

- `--image-publish-hz`
  - 类型：`float`
  - 默认值：`30.0`
  - 含义：RGBD 图像定时发布频率，单位 Hz。

- `--thermal-render-hz`
  - 类型：`float`
  - 默认值：`30.0`
  - 含义：热成像渲染频率，单位 Hz。

#### 扩展录制项

- `--record-depth-topics`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：录制深度图 topic，并在归档时导出深度可视化 `MP4`。

- `--record-thermal-topic`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：录制红外 `/thermal_camera/image` topic，并在归档时导出 `MP4`。

- `--record-tactile-topics`
  - 类型：开关参数
  - 默认值：关闭
  - 含义：录制触觉相关 topic。

### 4.2 常用启动示例

1. 默认方式启动：

    /usr/bin/python3 mujoco_env/main/main.py

2. 指定模型文件启动：

    /usr/bin/python3 mujoco_env/main/main.py --model mujoco_env/robot_model/exp/env_robot_torque_tactile.xml

3. 无界面运行：

    /usr/bin/python3 mujoco_env/main/main.py --headless

4. 启用触觉独立可视化窗口：

    /usr/bin/python3 mujoco_env/main/main.py --tactile-ui

5. 禁用 ROS 控制与 rosbag：

    /usr/bin/python3 mujoco_env/main/main.py --no-ros-control --no-rosbag

6. 启用性能日志并调整频率：

    /usr/bin/python3 mujoco_env/main/main.py --profile-logs --control-loop-hz 30 --image-publish-hz 15 --thermal-render-hz 15

7. 启用全部扩展录制项：

    /usr/bin/python3 mujoco_env/main/main.py --record-depth-topics --record-thermal-topic --record-tactile-topics

### 4.3 参数使用补充说明

- `--headless` 适合服务器或无显示环境运行，但此时不会弹出 MuJoCo GUI。
- `--no-image-publish` 与 `--no-depth-render` 不同：
  - `--no-image-publish` 是不发布 RGBD ROS2 话题。
  - `--no-depth-render` 是不生成深度图渲染结果。
- `--no-rosbag` 关闭后，`/teaching_status` 仍可发布，但不会触发 bag 录制。
- `--record-depth-topics`、`--record-thermal-topic`、`--record-tactile-topics` 都是在默认录制项基础上的增量开关，不会替代默认录制内容。





---

## 5. 控制输入接口（选手需要关注）

### 5.1 关节控制话题

话题名：/joint_target

消息类型：sensor_msgs/msg/JointState

约定：

- 前 6 个 position 对应机械臂 6 关节目标。
- 可选第 7 个值用于夹爪指令（gripper）。

示例：

    ros2 topic pub /joint_target sensor_msgs/msg/JointState \
    '{name: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"], position: [0.0, 0.5, -1.0, 0.0, -0.5, 0.0, 1.0]}'

### 5.2 记录控制话题

话题名：/teaching_status

消息类型：std_msgs/msg/String

控制值：

- start_teaching：开始记录
- end_teaching：结束记录

示例：

    ros2 topic pub -1 /teaching_status std_msgs/msg/String '{data: "start_teaching"}'
    ros2 topic pub -1 /teaching_status std_msgs/msg/String '{data: "end_teaching"}'

### 5.3 自动录制与归档说明

当前系统支持：

1. 发布 `/teaching_status = start_teaching` 后自动开始录制 `mcap`
2. 发布 `/teaching_status = end_teaching` 后自动停止录制
3. 停止录制后自动执行后处理，导出视频归档

默认输出目录：

- 原始 rosbag/mcap：
  - `rosbag_data/teaching_YYYYMMDD_HHMMSS/`
- 自动导出归档：
  - `rosbag_archive/teaching_YYYYMMDD_HHMMSS/`

默认归档行为：

- RGB 图像会自动导出为 `mp4`
- 同时保留逐帧时间戳索引文件
- 原始 `mcap` 不会删除

归档目录中常见文件包括：

- `video.mp4`
- `frames.jsonl.gz`
- `messages.jsonl.gz`
- `manifest.json`
- `initial_scene.json`

说明：

- `video.mp4` 用于压缩存储与直接回放
- `frames.jsonl.gz` 保存每一帧对应的精确时间戳与 ROS header 信息
- `messages.jsonl.gz` 保存非图像 topic 的压缩索引
- `manifest.json` 保存该归档的总体说明与 topic 清单
- `initial_scene.json` 保存录制开始时的物体位置、姿态与材质，用于 MuJoCo replay 恢复录制场景

### 5.4 采集完成后的回放与审核

采集结束后，系统会在 `rosbag_archive/` 下生成同名归档目录。可以使用 `mujoco_env/main/replay.py` 在 MuJoCo 中重新播放采集到的动作。

常用方式：

1. 回放最新一条归档：

    ./review_latest_replay.sh

2. 指定某一条归档回放：

    /usr/bin/python3 mujoco_env/main/replay.py --archive rosbag_archive/teaching_YYYYMMDD_HHMMSS

3. 进入审核模式回放：

    /usr/bin/python3 mujoco_env/main/replay.py --review --archive rosbag_archive/teaching_YYYYMMDD_HHMMSS

审核模式下，回放结束后终端会等待输入：

- `Enter` 或 `k`：保留该条数据
- `d`：删除该条数据
- `r`：重新回放一次
- `q`：退出审核并保留数据

回放时的场景恢复规则：

- 如果归档目录中存在 `initial_scene.json`，replay 会恢复录制开始时的物体位置、姿态与材质。
- 如果归档目录中没有 `initial_scene.json`，replay 会使用一次新的随机 reset 场景，因此可能和录制时看到的物体布局不同。

常用参数：

- `--archive-root`
  - 默认值：`rosbag_archive`
  - 含义：归档根目录。不传 `--archive` 时，会从该目录中选择最新的 `teaching_*` 归档。

- `--archive`
  - 含义：指定要回放的归档目录，可以传完整路径，也可以传 `teaching_YYYYMMDD_HHMMSS` 目录名。

- `--source`
  - 默认值：`auto`
  - 含义：选择回放数据来源。通常保持默认即可。

- `--speed`
  - 默认值：`1.0`
  - 含义：回放速度倍率。例如 `--speed 2.0` 表示 2 倍速。

- `--render-fps`
  - 默认值：`30.0`
  - 含义：GUI 渲染帧率。数据采样频率较高时，可适当降低该值保证回放流畅。

- `--headless`
  - 含义：无 GUI 回放，适合只检查数据是否可解析。

- `--no-realtime`
  - 含义：不按真实时间等待，尽快跑完回放。

- `--settle-seconds`
  - 默认值：`0.5`
  - 含义：回放结束后保持最后一个动作的时间，便于观察最终状态。

删除说明：

- 在 `--review` 审核模式下选择 `d` 时，会删除当前归档目录。
- 如果存在同名原始 rosbag 目录，也会同时删除 `rosbag_data/teaching_YYYYMMDD_HHMMSS/`。
- 删除逻辑只允许删除 `teaching_*` 数据目录，避免误删其他路径。

### 5.5 Web 回放 GUI

如果需要在浏览器里选择数据集、启动回放、重播、停止或删除不合格数据，可以启动本地 Web GUI：

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 mujoco_env/main/replay_gui.py

默认访问地址：

    http://127.0.0.1:8765

也可以启动后自动打开浏览器：

    /usr/bin/python3 mujoco_env/main/replay_gui.py --open-browser

GUI 支持的操作：

- 浏览 `rosbag_archive/` 下的 `teaching_*` 归档
- 按名称搜索、按时间或名称排序
- 选择数据源：`auto`、`joint_cmd`、`joint_states`、`joint_states_R`
- 设置回放速度、渲染帧率、末帧保持时间
- 开启 `headless` 或 `no-realtime`
- 播放、重播、停止当前回放
- 标记保留数据
- 删除不合格数据

说明：

- Web GUI 只负责选择和管理数据集，MuJoCo 画面仍会作为独立窗口打开。
- 同一时间只允许一个 replay 子进程运行。
- 删除数据时会沿用安全删除规则，只允许删除 `teaching_*` 目录。
- 默认只监听 `127.0.0.1`，不会暴露到局域网。
- 如需修改监听地址或端口，可使用 `--host` 和 `--port`。

### 5.6 录制项参数配置

默认情况下，不传任何额外参数时，录制项保持当前基础配置，即默认录制：

- `/ee_camera/rgb/image_raw`
- `/external_camera/rgb/image_raw`
- `/ee_camera/camera_info`
- `/external_camera/camera_info`
- `/ee_pose`
- `/ee_pose_gripper`
- `/ee_target`
- `/grasp_distance`
- `/joint_states_R`
- `/joint_states_sim`
- `/joint_target`
- `/gripper_status`
- `/teaching_status`
- `/parameter_events`
- `/rosout`

额外可选录制项如下：

- `--record-depth-topics`
  - 录制深度 topic：
    - `/ee_camera/depth/image_raw`
    - `/external_camera/depth/image_raw`
  - 归档时会额外导出深度可视化 `mp4`

- `--record-thermal-topic`
  - 录制红外 topic：
    - `/thermal_camera/image`
  - 归档时会导出红外 `mp4`

- `--record-tactile-topics`
  - 录制触觉 topic：
    - `/gripper_tactile/left/vector`
    - `/gripper_tactile/right/vector`
    - `/gripper_tactile/left/tangential`
    - `/gripper_tactile/right/tangential`
    - `/gripper_tactile/left/pad`
    - `/gripper_tactile/right/pad`

使用示例：

1. 仅使用当前默认录制项：

    /usr/bin/python3 mujoco_env/main/main.py

2. 在默认基础上增加深度录制：

    /usr/bin/python3 mujoco_env/main/main.py --record-depth-topics

3. 在默认基础上增加深度和红外录制：

    /usr/bin/python3 mujoco_env/main/main.py --record-depth-topics --record-thermal-topic

4. 录制全部扩展项：

    /usr/bin/python3 mujoco_env/main/main.py --record-depth-topics --record-thermal-topic --record-tactile-topics

补充说明：

- 深度图归档中的 `mp4` 是“可视化视频”，用于查看和压缩存储。
- 深度的原始数值母版仍保留在 `mcap` 中。
- 触觉数据不是图像，因此不会导出为 `mp4`，而是保存在原始 `mcap` 和归档索引中。

---

## 6. 可观测输出接口（建议重点监控）

### 6.1 状态类话题

- /joint_states
- /gripper_status
- /grasp_distance
- /ee_pose
- /task_scene_snapshot

说明：

- 选手可以订阅当前公开的所有 topic。
- `/task_scene_snapshot` 仅用于训练、调试和离线分析。
- 正式测评时不会发布 `/task_scene_snapshot`。
- 因此参赛方案不能依赖 `/task_scene_snapshot` 作为算法的在线输入。

其中 /gripper_status 为 UInt32 位掩码，核心位含义：

- bit0: idle
- bit1: grasping
- bit2: grasped
- bit3: drop_event
- bit4: collision
- bit5: joint_limit
- bit6: premature_drop_event
- bit7: at_target_zone
- bit8: lifted_from_plane
- bit9: placed_stable

### 6.2 图像与传感器类话题

- /ee_camera/rgb/image_raw
- /ee_camera/depth/image_raw
- /external_camera/rgb/image_raw
- /external_camera/depth/image_raw
- /thermal_camera/image
- /gripper_tactile/left/vector
- /gripper_tactile/right/vector
- /gripper_tactile/left/pad
- /gripper_tactile/right/pad
- /gripper_tactile/left/tangential
- /gripper_tactile/right/tangential

建议：

- 图像类优先用 ros2 topic hz 或可视化工具查看，不建议直接 echo 大数据流。

### 6.3 触觉力数据对应关系（重点）

当前夹爪触觉数据是 10x5 taxel 网格（rows=10, cols=5）。

每个触觉点有三轴力分量：

- fx: 局部 x 方向力
- fy: 局部 y 方向力
- fz: 局部 z 方向力（带符号）

在本项目中：

- 切向力（tangential）定义为平面内合力：

    tangential = sqrt(fx^2 + fy^2)

- 法向力（normal）定义为 z 向绝对值：

    normal = abs(fz)

话题与含义一一对应如下：

- /gripper_tactile/left/vector 和 /gripper_tactile/right/vector
    - 数据语义：三轴力向量
    - 原始形状：10x5x3
    - 第三个维度顺序：[fx, fy, fz_signed]

- /gripper_tactile/left/tangential 和 /gripper_tactile/right/tangential
    - 数据语义：切向力标量图
    - 原始形状：10x5
    - 数值来自 sqrt(fx^2 + fy^2)

- /gripper_tactile/left/pad 和 /gripper_tactile/right/pad
    - 数据语义：法向力标量图（pad 压力图）
    - 原始形状：10x5
    - 数值来自 abs(fz)

传输格式说明：

- ROS2 话题类型使用 std_msgs/Float32MultiArray。
- data 字段是线性一维数组，但 layout.dim 明确保留原始维度（rows, cols, xyz）。
- 也就是说语义上是二维/三维网格，只是在传输时做了线性化存储，不是丢失维度。

二维 10x5 到一维的展开规则：

- 行优先（row-major）展开。
- 映射公式：i = r * cols + c，其中 cols=5。
- 反解公式：r = i // 5，c = i % 5。

三维 10x5x3（vector）到一维的展开规则：

- 行优先 + 通道连续。
- 映射公式：i = ((r * cols) + c) * 3 + k，其中 k∈{0,1,2} 对应 fx,fy,fz。
- 反解公式：
    - rc = i // 3
    - k = i % 3
    - r = rc // 5
    - c = rc % 5

一致性关系（可用于校验）：

- tangential[r,c] 与 vector[r,c,0], vector[r,c,1] 满足
    tangential[r,c] = sqrt(vector[r,c,0]^2 + vector[r,c,1]^2)
- pad[r,c] 与 vector[r,c,2] 满足
    pad[r,c] = abs(vector[r,c,2])

完整恢复方法（订阅端可直接照抄）：

1. 先读取 `msg.layout.dim`，确认 rows、cols、xyz。
2. 再把 `msg.data` 按行优先顺序 reshape 回原始网格。
3. 对 `vector` 恢复成 `(10, 5, 3)`，对 `tangential/pad` 恢复成 `(10, 5)`。

示例代码（Python）：

    import numpy as np

    def restore_tactile(msg):
        rows = int(msg.layout.dim[0].size)
        cols = int(msg.layout.dim[1].size)
        data = np.asarray(msg.data, dtype=np.float32)

        # 如果是 vector 话题，layout.dim 里会有第三维 xyz
        if len(msg.layout.dim) == 3:
            xyz = int(msg.layout.dim[2].size)
            return data.reshape(rows, cols, xyz)

        # 如果是 pad / tangential 话题，恢复成二维网格
        return data.reshape(rows, cols)

物理意义总结：

- `vector`：每个触觉点的完整受力矢量，表示该 taxel 在局部坐标系下受到的接触力方向和大小。
  - `fx` / `fy`：平面内受力，反映横向摩擦、侧向挤压、滑移趋势。
  - `fz_signed`：法向受力分量，正负号保留了方向信息，可判断是向内压还是向外拉。
- `tangential`：切向力大小，只表示“平面内推/摩擦有多大”，不带方向。
- `pad`：法向力大小，只表示“按压有多重”，不带方向。

所以：

- 需要方向判断时，用 `vector`。
- 需要接触强弱或热力图时，用 `tangential` 或 `pad`。
- 如果要看接触“是否压到指面上”，优先看 `pad`。
- 如果要看“是否发生横向滑动”，优先看 `tangential`。

---

## 7. 评分口径说明（规则公开版）

以下是比赛规则口径，便于选手理解优化方向。评估实现由内部裁判工具执行。

### 7.1 阶段划分

根据 gripper 状态自动划分 4 阶段：

1. approach：接近阶段
2. grasping：抓取阶段
3. transport：搬运阶段
4. release：放置阶段

### 7.2 阶段内评分项

每阶段计算三项分数（0 到 100），再加权得到阶段分：

- 平滑度权重 0.40
- 稳定性权重 0.35
- 力矩效率权重 0.25

参考公式：

- 平滑度分 = max(0, 100 * (1 - overall_jerk / 150.0))
- 稳定性分 = max(0, 100 * (1 - end_vel_norm / 0.3))
- 力矩效率分 = max(0, 100 * (1 - effort_rms_mean / 80.0))

### 7.3 跨阶段加权

总体原始分 overall_score_raw 由各阶段加权汇总：

- approach: 0.10
- grasping: 0.15
- transport: 0.50
- release: 0.25

若某阶段无有效数据，其权重会按比例分配给有数据阶段。

### 7.4 惩罚项

总惩罚：

total_penalty = jitter_penalty + drop_penalty + collision_penalty + joint_limit_penalty

当前口径：

- 剧烈抖动惩罚：每阶段若出现严重抖动，按 5 分/次计入
- 掉落惩罚：10 分/次
- 碰撞惩罚：1 分/次
- 关节越限惩罚：1 分/次

总体分：

overall_score = max(0, overall_score_raw - total_penalty)

若发生掉落事件，最终分数存在上限约束（当前为 80）。

### 7.5 基本技能成功率（40分维度）

统计以下 6 项是否达成：

1. 稳定接触
2. 完全抬离平面
3. 抓取保持稳定
4. 全程无掉落
5. 放置到目标区
6. 放置后稳定无翻倒

基础技能分按命中项比例折算到 40 分维度。

### 7.6 任务效率（20分维度）

以操作时长与目标时长比值计算效率分，时长越短分越高，上限 20 分。

### 7.7 温度放置得分融合

放置任务会根据物体最终最近目标盘关系计算温度匹配得分，并按权重融合到总体分。

当前规则重点是反序温度匹配（高温放低温盘、低温放高温盘）。

---

## 8. 任务成功判定

一次任务通常需同时满足：

- 发生过有效 grasped
- 到达目标区域
- 放置后稳定
- 无掉落
- 存在正常 release

任一关键条件缺失都可能导致任务失败或显著扣分。

---

## 9. 选手提交与开发建议

### 9.1 建议提交内容

- 控制策略代码
- 运行配置说明
- 关键参数说明

### 9.2 建议保留的运行产物

- rosbag 记录目录（用于复核）
- 关键日志（温度初始化、抓取状态、失败原因）

### 9.3 稳定性建议

- 优先减少 transport 阶段抖动与速度突变。
- 优先避免掉落、碰撞和关节越限。
- 放置末段适当减速，确保 placed_stable 可持续成立。

---

## 10. 版本与发布说明

- main.py 属于运行入口，会在比赛环境提供。
- 内部评估实现不作为 release 分支公开交付内容。
- 选手请以本 README 的规则说明为准进行策略开发。
