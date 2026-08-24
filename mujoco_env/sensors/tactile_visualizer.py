"""
MuJoCo 夹爪触觉可视化器。

从旧的 Touch_Sensor/run_gripper_tactile_sensor.py 收敛到 sensors 模块中，
这样主程序不再需要通过动态加载脚本的方式接入触觉可视化。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _PanelCfg:
    prefix: str
    rows: int
    cols: int
    site_id_grid: object
    sensor_adr_grid: object
    geom_id_grid: object
    geom_ids: set[int]
    geom_to_rc: dict[int, tuple[int, int]]
    normal_heatmap: object | None = None
    tangent_quiver: object | None = None
    status: object | None = None
    tangential_scale_ema: float | None = None


class GripperTactileVisualizer:
    """左右夹爪触觉阵列可视化器。"""

    def __init__(
        self,
        model,
        data,
        *,
        prefixes: tuple[str, ...] = ("left", "right"),
        rows: int = 10,
        cols: int = 5,
        update_every: int = 5,
        arrow_width: float = 0.0009,
        arrow_length_scale: float = 0.08,
        arrow_min_length: float = 0.006,
        tangential_deadzone: float = 0.01,
        tangential_ema_alpha: float = 0.8,
        quiver_scale: float = 1.2,
    ):
        import matplotlib.pyplot as plt
        import mujoco
        import numpy as np

        self.model = model
        self.data = data
        self.plt = plt
        self.np = np
        self.mujoco = mujoco
        self.rows = rows
        self.cols = cols
        self.update_every = update_every
        self.arrow_width = arrow_width
        self.arrow_length_scale = arrow_length_scale
        self.arrow_min_length = arrow_min_length
        self.tangential_deadzone = tangential_deadzone
        self.tangential_ema_alpha = tangential_ema_alpha
        self.quiver_scale = quiver_scale

        self._step = 0
        self.panels: list[_PanelCfg] = [self._build_panel(prefix) for prefix in prefixes]

        self.plt.ion()
        panel_count = len(self.panels)
        fig_w = max(7.2, panel_count * 4.8)
        self.fig, axes = self.plt.subplots(1, panel_count, figsize=(fig_w, 6.2), squeeze=False)
        axes = axes[0]

        for index, panel in enumerate(self.panels):
            ax = axes[index]
            heat = ax.imshow(
                self.np.zeros((rows, cols), dtype=float),
                cmap="inferno",
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
            )
            self.fig.colorbar(heat, ax=ax, fraction=0.042, pad=0.02)
            ax.set_title(f"{panel.prefix} tactile")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_xticks(self.np.arange(cols))
            ax.set_yticks(self.np.arange(rows))
            ax.set_box_aspect(rows / cols)

            x_coords, y_coords = self.np.meshgrid(self.np.arange(cols), self.np.arange(rows))
            quiver = ax.quiver(
                x_coords,
                y_coords,
                self.np.zeros((rows, cols)),
                self.np.zeros((rows, cols)),
                color="#38bdf8",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                pivot="mid",
                width=0.0055,
                headwidth=2.8,
                headlength=4.2,
                headaxislength=3.8,
            )
            status = ax.text(
                0.98,
                -0.14,
                "maxN=0.0000 | maxT=0.0000 | maxV=0.0000",
                transform=ax.transAxes,
                fontsize=10,
                va="bottom",
                ha="right",
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="none"),
            )

            panel.normal_heatmap = heat
            panel.tangent_quiver = quiver
            panel.status = status

        self.fig.tight_layout(rect=[0.0, 0.10, 1.0, 0.98])
        self.fig.subplots_adjust(left=0.05, right=0.99, bottom=0.14, wspace=0.14)
        self.plt.show(block=False)

    def _build_panel(self, prefix: str) -> _PanelCfg:
        site_id_grid = []
        sensor_adr_grid = []
        geom_id_grid = []
        geom_ids: set[int] = set()
        geom_to_rc: dict[int, tuple[int, int]] = {}

        for row in range(self.rows):
            row_sites = []
            row_adrs = []
            row_geoms = []
            for col in range(self.cols):
                site_name = f"{prefix}_site_{row}_{col}"
                sensor_name = f"{prefix}_force_{row}_{col}"
                geom_name = f"{prefix}_cell_geom_{row}_{col}"

                site_id = self.model.site(site_name).id
                sensor_id = self.model.sensor(sensor_name).id
                sensor_adr = self.model.sensor_adr[sensor_id]

                row_sites.append(site_id)
                row_adrs.append(sensor_adr)

                try:
                    geom_id = self.model.geom(geom_name).id
                except KeyError:
                    geom_id = -1
                row_geoms.append(geom_id)
                if geom_id >= 0:
                    geom_ids.add(geom_id)
                    geom_to_rc[geom_id] = (row, col)

            site_id_grid.append(row_sites)
            sensor_adr_grid.append(row_adrs)
            geom_id_grid.append(row_geoms)

        return _PanelCfg(
            prefix=prefix,
            rows=self.rows,
            cols=self.cols,
            site_id_grid=site_id_grid,
            sensor_adr_grid=sensor_adr_grid,
            geom_id_grid=geom_id_grid,
            geom_ids=geom_ids,
            geom_to_rc=geom_to_rc,
        )

    def _read_panel_grids(self, panel: _PanelCfg):
        normal = self.np.zeros((panel.rows, panel.cols), dtype=float)
        tangential = self.np.zeros((panel.rows, panel.cols), dtype=float)
        fx_grid = self.np.zeros((panel.rows, panel.cols), dtype=float)
        fy_grid = self.np.zeros((panel.rows, panel.cols), dtype=float)
        fz_signed_grid = self.np.zeros((panel.rows, panel.cols), dtype=float)

        for row in range(panel.rows):
            for col in range(panel.cols):
                adr = panel.sensor_adr_grid[row][col]
                fx = float(self.data.sensordata[adr + 0])
                fy = float(self.data.sensordata[adr + 1])
                fz = float(self.data.sensordata[adr + 2])

                fx_grid[row, col] = -fx
                fy_grid[row, col] = -fy
                fz_signed_grid[row, col] = -fz
                normal[row, col] = abs(fz_signed_grid[row, col])
                tangential[row, col] = math.sqrt(fx * fx + fy * fy)

        vector = self.np.sqrt(
            fx_grid * fx_grid + fy_grid * fy_grid + fz_signed_grid * fz_signed_grid
        )
        if float(self.np.max(vector)) <= 1e-9:
            return self._read_panel_grids_from_contacts(panel)
        return {
            "normal_grid": normal,
            "tangential_grid": tangential,
            "fx_grid": fx_grid,
            "fy_grid": fy_grid,
            "fz_signed_grid": fz_signed_grid,
            "vector_grid": vector,
        }

    def _read_panel_grids_from_contacts(self, panel: _PanelCfg):
        fx_grid = self.np.zeros((panel.rows, panel.cols), dtype=float)
        fy_grid = self.np.zeros((panel.rows, panel.cols), dtype=float)
        fz_signed_grid = self.np.zeros((panel.rows, panel.cols), dtype=float)

        wrench = self.np.zeros(6, dtype=float)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in panel.geom_to_rc and geom2 not in panel.geom_to_rc:
                continue

            self.mujoco.mj_contactForce(self.model, self.data, index, wrench)
            frame = contact.frame.reshape(3, 3)
            f_contact = self.np.array([wrench[0], wrench[1], wrench[2]], dtype=float)
            f_world = frame.T @ f_contact

            if geom2 in panel.geom_to_rc:
                row, col = panel.geom_to_rc[geom2]
                site_id = panel.site_id_grid[row][col]
                rot = self.data.site_xmat[site_id].reshape(3, 3)
                f_local = rot.T @ f_world
                fx_grid[row, col] += -f_local[0]
                fy_grid[row, col] += -f_local[1]
                fz_signed_grid[row, col] += -f_local[2]
            if geom1 in panel.geom_to_rc:
                row, col = panel.geom_to_rc[geom1]
                site_id = panel.site_id_grid[row][col]
                rot = self.data.site_xmat[site_id].reshape(3, 3)
                f_local = rot.T @ (-f_world)
                fx_grid[row, col] += -f_local[0]
                fy_grid[row, col] += -f_local[1]
                fz_signed_grid[row, col] += -f_local[2]

        normal = self.np.abs(fz_signed_grid)
        tangential = self.np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid)
        vector = self.np.sqrt(
            fx_grid * fx_grid + fy_grid * fy_grid + fz_signed_grid * fz_signed_grid
        )
        return {
            "normal_grid": normal,
            "tangential_grid": tangential,
            "fx_grid": fx_grid,
            "fy_grid": fy_grid,
            "fz_signed_grid": fz_signed_grid,
            "vector_grid": vector,
        }

    def get_sensor_grids(self):
        return {panel.prefix: self._read_panel_grids(panel) for panel in self.panels}

    def _get_contact_geoms(self):
        contact_geoms = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            contact_geoms.add(int(contact.geom1))
            contact_geoms.add(int(contact.geom2))
        return contact_geoms

    @staticmethod
    def _lerp(low_rgba, high_rgba, alpha):
        alpha = max(0.0, min(1.0, float(alpha)))
        return [lo + (hi - lo) * alpha for lo, hi in zip(low_rgba, high_rgba)]

    def update(self, viewer=None):
        self._step += 1
        if self._step % self.update_every != 0:
            return

        if viewer is not None:
            viewer.user_scn.ngeom = 0
            contact_geoms = self._get_contact_geoms()
        else:
            contact_geoms = set()

        grids = self.get_sensor_grids()

        for panel in self.panels:
            grid = grids[panel.prefix]
            normal = grid["normal_grid"]
            tangential = grid["tangential_grid"]
            fx_grid = grid["fx_grid"]
            fy_grid = grid["fy_grid"]
            fz_signed_grid = grid["fz_signed_grid"]
            vector = grid["vector_grid"]

            max_n = float(self.np.max(normal))
            max_t = float(self.np.max(tangential))
            max_v = float(self.np.max(vector))

            scale_now = max(1.0, max_t)
            if panel.tangential_scale_ema is None:
                panel.tangential_scale_ema = scale_now
            else:
                panel.tangential_scale_ema = (
                    self.tangential_ema_alpha * scale_now
                    + (1.0 - self.tangential_ema_alpha) * panel.tangential_scale_ema
                )

            mag = self.np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid)
            fx_plot = self.np.where(mag >= self.tangential_deadzone, fx_grid, 0.0)
            fy_plot = self.np.where(mag >= self.tangential_deadzone, fy_grid, 0.0)

            panel.normal_heatmap.set_data(normal)
            panel.normal_heatmap.set_clim(0.0, max(1.0, max_n))
            panel.tangent_quiver.set_UVC(
                fx_plot / panel.tangential_scale_ema * self.quiver_scale,
                fy_plot / panel.tangential_scale_ema * self.quiver_scale,
            )
            panel.status.set_text(f"maxN={max_n:.4f} | maxT={max_t:.4f} | maxV={max_v:.4f}")

            if viewer is None:
                continue

            v_scale = max(1.0, max_v)
            for row in range(panel.rows):
                for col in range(panel.cols):
                    geom_id = panel.geom_id_grid[row][col]
                    site_id = panel.site_id_grid[row][col]
                    n_level = normal[row, col] / max(1.0, max_n)

                    if geom_id >= 0:
                        self.model.geom_rgba[geom_id] = self._lerp(
                            [0.88, 0.88, 0.88, 0.90], [0.96, 0.25, 0.10, 1.0], n_level
                        )
                    self.model.site_rgba[site_id] = self._lerp(
                        [0.30, 0.85, 1.00, 0.12], [0.30, 0.85, 1.00, 0.55], n_level
                    )

                    if geom_id >= 0 and geom_id not in contact_geoms:
                        continue

                    fx = float(fx_grid[row, col])
                    fy = float(fy_grid[row, col])
                    fz = float(fz_signed_grid[row, col])
                    vmag = math.sqrt(fx * fx + fy * fy + fz * fz)
                    if vmag <= 1e-10:
                        continue

                    direction_local = self.np.array([fx, fy, fz], dtype=float)
                    rot = self.data.site_xmat[site_id].reshape(3, 3)
                    direction_world = rot @ direction_local
                    direction_world /= (self.np.linalg.norm(direction_world) + 1e-12)

                    start = self.data.site_xpos[site_id].copy()
                    level = vmag / v_scale
                    length = self.arrow_min_length + self.arrow_length_scale * level
                    end = start + direction_world * length

                    arrow_id = viewer.user_scn.ngeom
                    if arrow_id >= viewer.user_scn.maxgeom:
                        break
                    self.mujoco.mjv_connector(
                        viewer.user_scn.geoms[arrow_id],
                        self.mujoco.mjtGeom.mjGEOM_ARROW,
                        self.arrow_width,
                        start,
                        end,
                    )
                    viewer.user_scn.geoms[arrow_id].rgba[:] = self._lerp(
                        [0.25, 0.85, 1.00, 0.72], [0.10, 0.95, 0.35, 0.95], level
                    )
                    viewer.user_scn.ngeom += 1

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        self.plt.ioff()
        self.plt.close(self.fig)
