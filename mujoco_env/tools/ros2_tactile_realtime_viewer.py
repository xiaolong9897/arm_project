#!/usr/bin/python3
"""ROS2 tactile topic realtime viewer.

Subscribe to:
- /gripper_tactile/left/vector   (10x5x3)
- /gripper_tactile/right/vector  (10x5x3)
- /gripper_tactile/left/tangential (10x5, optional)
- /gripper_tactile/right/tangential (10x5, optional)

And render:
- 2D heatmap: normal force |fz|
- 2D quiver : tangential direction (fx, fy)
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError("matplotlib is required for ros2_tactile_realtime_viewer.py") from exc

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
except Exception as exc:
    raise RuntimeError("ROS2 Python dependencies are required (rclpy, std_msgs)") from exc


@dataclass
class SideState:
    vector: np.ndarray
    tangential: np.ndarray
    has_vector: bool = False
    has_tangential: bool = False


class TactileViewerNode(Node):
    def __init__(self, rows: int, cols: int, use_tangential_topic: bool):
        super().__init__("ros2_tactile_realtime_viewer")
        self.rows = rows
        self.cols = cols
        self.use_tangential_topic = use_tangential_topic

        self._lock = threading.Lock()
        self.left = SideState(
            vector=np.zeros((rows, cols, 3), dtype=np.float32),
            tangential=np.zeros((rows, cols), dtype=np.float32),
        )
        self.right = SideState(
            vector=np.zeros((rows, cols, 3), dtype=np.float32),
            tangential=np.zeros((rows, cols), dtype=np.float32),
        )

        self.create_subscription(
            Float32MultiArray,
            "/gripper_tactile/left/vector",
            self._on_left_vector,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/gripper_tactile/right/vector",
            self._on_right_vector,
            10,
        )

        if use_tangential_topic:
            self.create_subscription(
                Float32MultiArray,
                "/gripper_tactile/left/tangential",
                self._on_left_tangential,
                10,
            )
            self.create_subscription(
                Float32MultiArray,
                "/gripper_tactile/right/tangential",
                self._on_right_tangential,
                10,
            )

        self.get_logger().info("Subscribed tactile topics")

    def _reshape_vector(self, msg: Float32MultiArray) -> np.ndarray | None:
        arr = np.asarray(msg.data, dtype=np.float32)
        expected = self.rows * self.cols * 3
        if arr.size != expected:
            return None
        return arr.reshape(self.rows, self.cols, 3)

    def _reshape_grid(self, msg: Float32MultiArray) -> np.ndarray | None:
        arr = np.asarray(msg.data, dtype=np.float32)
        expected = self.rows * self.cols
        if arr.size != expected:
            return None
        return arr.reshape(self.rows, self.cols)

    def _on_left_vector(self, msg: Float32MultiArray):
        data = self._reshape_vector(msg)
        if data is None:
            return
        with self._lock:
            self.left.vector = data
            self.left.has_vector = True

    def _on_right_vector(self, msg: Float32MultiArray):
        data = self._reshape_vector(msg)
        if data is None:
            return
        with self._lock:
            self.right.vector = data
            self.right.has_vector = True

    def _on_left_tangential(self, msg: Float32MultiArray):
        data = self._reshape_grid(msg)
        if data is None:
            return
        with self._lock:
            self.left.tangential = data
            self.left.has_tangential = True

    def _on_right_tangential(self, msg: Float32MultiArray):
        data = self._reshape_grid(msg)
        if data is None:
            return
        with self._lock:
            self.right.tangential = data
            self.right.has_tangential = True

    def snapshot(self):
        with self._lock:
            return (
                self.left.vector.copy(),
                self.left.tangential.copy(),
                self.left.has_vector,
                self.left.has_tangential,
                self.right.vector.copy(),
                self.right.tangential.copy(),
                self.right.has_vector,
                self.right.has_tangential,
            )


def build_ui(rows: int, cols: int):
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), squeeze=False)
    axes = axes[0]

    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    views = []

    for idx, side in enumerate(("left", "right")):
        ax = axes[idx]
        heat = ax.imshow(
            np.zeros((rows, cols), dtype=np.float32),
            cmap="inferno",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        fig.colorbar(heat, ax=ax, fraction=0.042, pad=0.02)
        quiver = ax.quiver(
            x_coords,
            y_coords,
            np.zeros((rows, cols), dtype=np.float32),
            np.zeros((rows, cols), dtype=np.float32),
            color="#38bdf8",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="mid",
            width=0.006,
            headwidth=3.0,
            headlength=4.0,
        )
        status = ax.text(
            0.98,
            -0.12,
            "waiting data...",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="right",
        )
        ax.set_title(f"{side} tactile")
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_box_aspect(rows / cols)
        views.append((heat, quiver, status))

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.98])
    return fig, views


def main():
    parser = argparse.ArgumentParser(description="ROS2 realtime tactile viewer")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--hz", type=float, default=20.0, help="UI refresh frequency")
    parser.add_argument(
        "--ignore-tangential-topic",
        action="store_true",
        help="Compute tangential from vector topic only",
    )
    args = parser.parse_args()

    rclpy.init()
    node = TactileViewerNode(
        rows=args.rows,
        cols=args.cols,
        use_tangential_topic=not args.ignore_tangential_topic,
    )

    fig, views = build_ui(args.rows, args.cols)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        sleep_dt = 1.0 / max(args.hz, 1e-3)
        while plt.fignum_exists(fig.number):
            (
                left_vec,
                left_tan,
                left_ok,
                left_tan_ok,
                right_vec,
                right_tan,
                right_ok,
                right_tan_ok,
            ) = node.snapshot()

            for side_idx, (vec, tan, has_vec, has_tan) in enumerate(
                (
                    (left_vec, left_tan, left_ok, left_tan_ok),
                    (right_vec, right_tan, right_ok, right_tan_ok),
                )
            ):
                heat, quiver, status = views[side_idx]
                if not has_vec:
                    status.set_text("waiting vector topic...")
                    continue

                fx = vec[:, :, 0]
                fy = vec[:, :, 1]
                fz = vec[:, :, 2]
                normal = np.abs(fz)

                if not has_tan or args.ignore_tangential_topic:
                    tangential = np.sqrt(np.square(fx) + np.square(fy))
                else:
                    tangential = tan

                max_n = float(np.max(normal))
                max_t = float(np.max(tangential))
                heat.set_data(normal)
                heat.set_clim(0.0, max(1.0, max_n))

                scale = max(1.0, max_t)
                deadzone = 0.01
                mag = np.sqrt(np.square(fx) + np.square(fy))
                fx_plot = np.where(mag >= deadzone, fx / scale, 0.0)
                fy_plot = np.where(mag >= deadzone, fy / scale, 0.0)
                quiver.set_UVC(fx_plot, fy_plot)
                status.set_text(f"maxN={max_n:.4f} | maxT={max_t:.4f}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(sleep_dt)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close(fig)


if __name__ == "__main__":
    main()
