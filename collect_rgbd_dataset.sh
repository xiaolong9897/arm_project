#!/usr/bin/env bash
# 每组数据都重新启动一次仿真，使 main.py 在启动时执行随机 reset。
# 用法：./collect_rgbd_dataset.sh [采集次数] [启动等待秒数] [每轮间隔秒数]
# 示例：bash collect_rgbd_dataset.sh 100 3 5

# ROS 的 setup.bash 会访问可能尚未定义的环境变量，故先不开启 -u。
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_COUNT="${1:-10}"
STARTUP_WAIT_SECONDS="${2:-5}"
ROUND_INTERVAL_SECONDS="${3:-5}"
SYSTEM_PYTHON="/usr/bin/python3"

if ! [[ "$SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "采集次数必须是正整数，当前值：$SAMPLE_COUNT" >&2
    exit 2
fi

if ! [[ "$STARTUP_WAIT_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "启动等待秒数必须是非负数字，当前值：$STARTUP_WAIT_SECONDS" >&2
    exit 2
fi

if ! [[ "$ROUND_INTERVAL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "每轮间隔秒数必须是非负数字，当前值：$ROUND_INTERVAL_SECONDS" >&2
    exit 2
fi

source /opt/ros/humble/setup.bash
set -u
cd "$PROJECT_ROOT"

# --headless 仍需要离屏渲染 external RGB-D；EGL 不依赖 X11/GUI 窗口。
export MUJOCO_GL=egl

SIM_PID=""

stop_simulation() {
    if [[ -z "$SIM_PID" ]] || ! kill -0 "$SIM_PID" 2>/dev/null; then
        SIM_PID=""
        return
    fi
    echo "请求 MuJoCo 正常退出（Ctrl+C，PID: $SIM_PID）..."
    kill -INT "$SIM_PID" 2>/dev/null || true
    for _ in {1..10}; do
        if ! kill -0 "$SIM_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$SIM_PID" 2>/dev/null; then
        echo "正常退出超时，改用 SIGTERM 停止仿真..." >&2
        kill -TERM "$SIM_PID" 2>/dev/null || true
    fi
    wait "$SIM_PID" 2>/dev/null || true
    SIM_PID=""
}
trap stop_simulation EXIT INT TERM

for ((index = 1; index <= SAMPLE_COUNT; index++)); do
    echo "[$index/$SAMPLE_COUNT] 启动无 GUI 仿真（本次会随机 reset）..."
    (
        cd "$PROJECT_ROOT/mujoco_env"
        exec "$SYSTEM_PYTHON" main/main.py --headless --no-rosbag
    ) &
    SIM_PID=$!

    echo "[$index/$SAMPLE_COUNT] 等待 $STARTUP_WAIT_SECONDS 秒，让仿真和 ROS 话题就绪..."
    sleep "$STARTUP_WAIT_SECONDS"
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        echo "仿真启动失败，未开始采集。请查看上方 MuJoCo 日志。" >&2
        exit 1
    fi

    echo "[$index/$SAMPLE_COUNT] 开始采集..."
    "$SYSTEM_PYTHON" mujoco_env/main/xiaoman1/extel_rgb_depth.py
    echo "[$index/$SAMPLE_COUNT] 采集完成。"
    stop_simulation
    if (( index < SAMPLE_COUNT )); then
        echo "[$index/$SAMPLE_COUNT] 等待 $ROUND_INTERVAL_SECONDS 秒后开始下一轮..."
        sleep "$ROUND_INTERVAL_SECONDS"
    fi
done

echo "全部 $SAMPLE_COUNT 组数据采集完成，每组均经过独立随机 reset。"
