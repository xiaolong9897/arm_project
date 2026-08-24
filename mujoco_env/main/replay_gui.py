#!/usr/bin/python3
"""
Local Web GUI for reviewing MuJoCo teaching replay archives.

The GUI is intentionally implemented with the Python standard library so it can
run in the same system Python environment as replay.py without extra packages.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PYTHON = "/usr/bin/python3"
DEFAULT_MODEL = PROJECT_ROOT / "mujoco_env/robot_model/exp/env_robot_torque_tactile.xml"
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "rosbag_archive"
DEFAULT_BAG_ROOT = PROJECT_ROOT / "rosbag_data"
REPLAY_SCRIPT = PROJECT_ROOT / "mujoco_env/main/replay.py"
JOINT_TOPICS = ("joint_cmd", "joint_states", "joint_states_R")


def _enforce_system_python() -> None:
    current_python = os.path.realpath(sys.executable)
    expected_python = os.path.realpath(SYSTEM_PYTHON)
    if current_python != expected_python:
        raise SystemExit(
            "\nreplay_gui.py 必须使用系统 Python 运行。\n"
            f"   当前: {current_python}\n"
            f"   要求: {expected_python}\n"
            f"   请改用: {SYSTEM_PYTHON} mujoco_env/main/replay_gui.py\n"
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_rmtree(path: Path, root: Path) -> bool:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if not path.exists():
        return False
    if not path.is_dir():
        raise ValueError(f"拒绝删除非目录路径: {path}")
    if path == root or not _is_relative_to(path, root):
        raise ValueError(f"拒绝删除根目录外路径: {path} (root={root})")
    shutil.rmtree(path)
    return True


def delete_recording_pair(archive_dir: Path, archive_root: Path, bag_root: Path) -> list[Path]:
    archive_dir = archive_dir.expanduser().resolve()
    archive_root = archive_root.expanduser().resolve()
    bag_root = bag_root.expanduser().resolve()

    if not archive_dir.name.startswith("teaching_"):
        raise ValueError(f"拒绝删除非 teaching_* 目录: {archive_dir}")

    deleted: list[Path] = []
    if _safe_rmtree(archive_dir, archive_root):
        deleted.append(archive_dir)

    raw_bag_dir = bag_root / archive_dir.name
    if _safe_rmtree(raw_bag_dir, bag_root):
        deleted.append(raw_bag_dir)

    return deleted


def _topic_dir_name(topic_name: str) -> str:
    return topic_name.strip("/").replace("/", "__") or "root_topic"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


class ReplayGuiState:
    def __init__(self, archive_root: Path, bag_root: Path, model_path: Path) -> None:
        self.archive_root = archive_root.expanduser().resolve()
        self.bag_root = bag_root.expanduser().resolve()
        self.model_path = model_path.expanduser().resolve()
        self.lock = threading.RLock()
        self.process: Optional[subprocess.Popen[str]] = None
        self.status = "idle"
        self.current_archive: Optional[Path] = None
        self.current_command: list[str] = []
        self.exit_code: Optional[int] = None
        self.stopping = False
        self.logs: deque[str] = deque(maxlen=500)
        self.reviewed_keep: set[str] = set()

    def add_log(self, line: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.logs.append(f"[{timestamp}] {line.rstrip()}")

    def resolve_archive(self, value: str) -> Path:
        raw = Path(str(value).strip())
        if not str(raw):
            raise ValueError("缺少归档目录")

        if raw.is_absolute():
            archive_dir = raw.expanduser().resolve()
        else:
            archive_dir = (self.archive_root / raw).expanduser().resolve()

        if not _is_relative_to(archive_dir, self.archive_root):
            raise ValueError(f"归档目录必须位于 archive root 内: {self.archive_root}")
        if not archive_dir.name.startswith("teaching_"):
            raise ValueError(f"只允许操作 teaching_* 归档: {archive_dir.name}")
        if not (archive_dir / "manifest.json").exists():
            raise FileNotFoundError(f"归档目录缺少 manifest.json: {archive_dir}")
        return archive_dir

    def list_archives(self) -> list[dict[str, Any]]:
        if not self.archive_root.exists():
            return []

        archives: list[dict[str, Any]] = []
        for path in self.archive_root.iterdir():
            if not path.is_dir() or not path.name.startswith("teaching_"):
                continue
            manifest_path = path / "manifest.json"
            if not manifest_path.exists():
                continue

            manifest = _read_json(manifest_path)
            topics = manifest.get("topics", [])
            available_topics: list[str] = []
            message_counts: dict[str, int] = {}
            first_ns: Optional[int] = None
            last_ns: Optional[int] = None

            if isinstance(topics, list):
                for topic_info in topics:
                    if not isinstance(topic_info, dict):
                        continue
                    topic_name = str(topic_info.get("topic_name", "")).strip("/")
                    if topic_name in JOINT_TOPICS:
                        available_topics.append(topic_name)
                        count = int(topic_info.get("message_count") or 0)
                        message_counts[topic_name] = count
                        for key, current in (("first_bag_time_ns", "first"), ("last_bag_time_ns", "last")):
                            value = topic_info.get(key)
                            if value is None:
                                continue
                            value = int(value)
                            if current == "first":
                                first_ns = value if first_ns is None else min(first_ns, value)
                            else:
                                last_ns = value if last_ns is None else max(last_ns, value)

            topics_dir = path / "topics"
            for topic in JOINT_TOPICS:
                topic_path = topics_dir / _topic_dir_name(f"/{topic}") / "messages.jsonl.gz"
                if topic_path.exists() and topic not in available_topics:
                    available_topics.append(topic)

            duration_s = None
            if first_ns is not None and last_ns is not None and last_ns >= first_ns:
                duration_s = (last_ns - first_ns) / 1e9

            stat = path.stat()
            archives.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "mtime": stat.st_mtime,
                    "mtime_text": _format_timestamp(stat.st_mtime),
                    "has_scene": (path / "initial_scene.json").exists(),
                    "available_topics": sorted(available_topics),
                    "message_counts": message_counts,
                    "duration_s": duration_s,
                    "raw_bag_exists": (self.bag_root / path.name).exists(),
                    "kept": path.name in self.reviewed_keep,
                }
            )

        archives.sort(key=lambda item: float(item["mtime"]), reverse=True)
        return archives

    def start_replay(self, payload: dict[str, Any]) -> dict[str, Any]:
        archive_dir = self.resolve_archive(str(payload.get("archive", "")))
        source = str(payload.get("source") or "auto")
        if source not in ("auto", *JOINT_TOPICS):
            raise ValueError(f"未知数据源: {source}")

        speed = max(float(payload.get("speed") or 1.0), 1e-6)
        render_fps = max(float(payload.get("render_fps") or 30.0), 1.0)
        settle_seconds = max(float(payload.get("settle_seconds") or 0.5), 0.0)
        headless = bool(payload.get("headless"))
        no_realtime = bool(payload.get("no_realtime"))

        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("已有 replay 正在运行，请先停止当前回放")

            command = [
                SYSTEM_PYTHON,
                str(REPLAY_SCRIPT),
                "--archive",
                str(archive_dir),
                "--model",
                str(self.model_path),
                "--source",
                source,
                "--speed",
                str(speed),
                "--render-fps",
                str(render_fps),
                "--settle-seconds",
                str(settle_seconds),
            ]
            if headless:
                command.append("--headless")
            if no_realtime:
                command.append("--no-realtime")

            self.logs.clear()
            self.current_archive = archive_dir
            self.current_command = command
            self.exit_code = None
            self.stopping = False
            self.status = "running"
            self.add_log("启动回放: " + " ".join(command))

            self.process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            threading.Thread(target=self._read_process_output, daemon=True).start()
            threading.Thread(target=self._watch_process, daemon=True).start()
            return self.get_status_locked()

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self.add_log(line)
        except Exception as exc:
            self.add_log(f"读取 replay 日志失败: {exc}")

    def _watch_process(self) -> None:
        process = self.process
        if process is None:
            return
        exit_code = process.wait()
        with self.lock:
            self.exit_code = exit_code
            if self.stopping:
                self.status = "stopped"
            elif exit_code == 0:
                self.status = "completed"
            else:
                self.status = "failed"
            self.add_log(f"replay 进程结束，退出码: {exit_code}")

    def stop_replay(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                self.status = "idle" if self.current_archive is None else self.status
                return self.get_status_locked()
            self.stopping = True
            self.status = "stopping"
            self.add_log("正在停止 replay 进程")
            process.terminate()

        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.add_log("terminate 超时，执行 kill")
            process.kill()
            process.wait(timeout=5.0)

        with self.lock:
            return self.get_status_locked()

    def keep_archive(self, payload: dict[str, Any]) -> dict[str, Any]:
        archive_dir = self.resolve_archive(str(payload.get("archive", "")))
        with self.lock:
            self.reviewed_keep.add(archive_dir.name)
            self.add_log(f"已标记保留: {archive_dir.name}")
        return {"kept": archive_dir.name}

    def delete_archive(self, payload: dict[str, Any]) -> dict[str, Any]:
        archive_dir = self.resolve_archive(str(payload.get("archive", "")))
        with self.lock:
            if (
                self.process is not None
                and self.process.poll() is None
                and self.current_archive is not None
                and self.current_archive.resolve() == archive_dir.resolve()
            ):
                raise RuntimeError("当前数据正在回放，删除前请先停止")

        deleted = delete_recording_pair(archive_dir, self.archive_root, self.bag_root)
        with self.lock:
            self.reviewed_keep.discard(archive_dir.name)
            self.add_log("已删除数据: " + ", ".join(str(path) for path in deleted))
        return {"deleted": [str(path) for path in deleted]}

    def get_status_locked(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "archive": self.current_archive.name if self.current_archive else None,
            "archive_path": str(self.current_archive) if self.current_archive else None,
            "exit_code": self.exit_code,
            "command": self.current_command,
            "logs": list(self.logs),
        }

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            return self.get_status_locked()


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuJoCo 数据回放 GUI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #172033;
      --muted: #667085;
      --accent: #1769aa;
      --accent-dark: #0f4f82;
      --danger: #b42318;
      --success: #067647;
      --warn: #b54708;
      --shadow: 0 1px 3px rgba(16, 24, 40, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      font-size: 14px;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 650; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) 390px;
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 58px);
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    input, select {
      width: 100%;
      min-height: 36px;
      border: 1px solid #c9d1df;
      border-radius: 6px;
      padding: 7px 9px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    button {
      min-height: 36px;
      border: 1px solid #b9c3d3;
      border-radius: 6px;
      padding: 7px 11px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      white-space: nowrap;
    }
    button:hover { border-color: var(--accent); color: var(--accent-dark); }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    button.primary:hover { background: var(--accent-dark); color: #fff; }
    button.danger { border-color: #f1b4ae; color: var(--danger); }
    button.danger:hover { background: #fff3f1; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .archive-list { overflow: auto; max-height: calc(100vh - 152px); }
    .archive-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 12px;
      border-bottom: 1px solid #eef1f5;
      cursor: pointer;
    }
    .archive-row:hover, .archive-row.selected { background: #eef6fc; }
    .archive-title { font-weight: 650; overflow-wrap: anywhere; }
    .archive-meta { color: var(--muted); font-size: 12px; margin-top: 5px; line-height: 1.55; }
    .badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 5px; }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid #d4dbe7;
      background: #f9fafb;
      color: #344054;
      font-size: 12px;
    }
    .badge.ok { color: var(--success); border-color: #9bd6bd; background: #edfdf5; }
    .badge.warn { color: var(--warn); border-color: #f7cf9d; background: #fff7ed; }
    .side { padding: 14px; display: flex; flex-direction: column; gap: 14px; }
    .selected-name { font-size: 16px; font-weight: 650; min-height: 24px; overflow-wrap: anywhere; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    label { display: flex; flex-direction: column; gap: 5px; color: var(--muted); font-size: 12px; }
    label span { color: var(--muted); }
    .checks { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .check {
      flex-direction: row;
      align-items: center;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 36px;
    }
    .check input { width: auto; min-height: 0; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .actions .wide { grid-column: 1 / -1; }
    .status {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
    }
    .status-line { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .status strong { text-transform: uppercase; letter-spacing: 0; }
    pre {
      margin: 0;
      padding: 10px;
      min-height: 220px;
      max-height: calc(100vh - 520px);
      overflow: auto;
      border: 1px solid #202938;
      border-radius: 8px;
      background: #101828;
      color: #d0d5dd;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .empty {
      padding: 28px 12px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .archive-list { max-height: 46vh; }
      pre { max-height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MuJoCo 数据回放 GUI</h1>
    <button id="refreshBtn">刷新数据集</button>
  </header>
  <main>
    <section>
      <div class="toolbar">
        <input id="searchInput" placeholder="搜索 teaching_ 数据集">
        <select id="sortSelect">
          <option value="mtime_desc">时间从新到旧</option>
          <option value="mtime_asc">时间从旧到新</option>
          <option value="name_asc">名称 A-Z</option>
          <option value="name_desc">名称 Z-A</option>
        </select>
        <button id="clearBtn">清空</button>
      </div>
      <div id="archiveList" class="archive-list"></div>
    </section>
    <aside class="side">
      <div>
        <div class="selected-name" id="selectedName">未选择数据集</div>
        <div class="archive-meta" id="selectedMeta">请从左侧选择一个 teaching_* 归档。</div>
      </div>
      <div class="form-grid">
        <label><span>数据源</span>
          <select id="sourceInput">
            <option value="auto">auto</option>
            <option value="joint_cmd">joint_cmd</option>
            <option value="joint_states">joint_states</option>
            <option value="joint_states_R">joint_states_R</option>
          </select>
        </label>
        <label><span>速度倍率</span><input id="speedInput" type="number" min="0.1" step="0.1" value="1.0"></label>
        <label><span>渲染 FPS</span><input id="fpsInput" type="number" min="1" step="1" value="30"></label>
        <label><span>末帧保持秒数</span><input id="settleInput" type="number" min="0" step="0.1" value="0.5"></label>
      </div>
      <div class="checks">
        <label class="check"><input id="headlessInput" type="checkbox"> headless</label>
        <label class="check"><input id="noRealtimeInput" type="checkbox"> no-realtime</label>
      </div>
      <div class="actions">
        <button class="primary" id="playBtn">播放</button>
        <button id="replayBtn">重播</button>
        <button id="stopBtn">停止</button>
        <button id="keepBtn">保留</button>
        <button class="danger wide" id="deleteBtn">删除当前数据</button>
      </div>
      <div class="status">
        <div class="status-line">
          <span>状态</span>
          <strong id="statusText">idle</strong>
        </div>
        <div class="archive-meta" id="runningText"></div>
      </div>
      <pre id="logBox"></pre>
    </aside>
  </main>
  <script>
    let archives = [];
    let selected = null;

    const $ = (id) => document.getElementById(id);

    function fmtDuration(value) {
      if (value === null || value === undefined) return "未知时长";
      return `${Number(value).toFixed(2)}s`;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch]));
    }

    function filteredArchives() {
      const q = $("searchInput").value.trim().toLowerCase();
      let rows = archives.filter((item) => item.name.toLowerCase().includes(q));
      const mode = $("sortSelect").value;
      rows.sort((a, b) => {
        if (mode === "mtime_asc") return a.mtime - b.mtime;
        if (mode === "name_asc") return a.name.localeCompare(b.name);
        if (mode === "name_desc") return b.name.localeCompare(a.name);
        return b.mtime - a.mtime;
      });
      return rows;
    }

    function renderArchives() {
      const list = $("archiveList");
      const rows = filteredArchives();
      if (!rows.length) {
        list.innerHTML = '<div class="empty">没有找到可回放的数据集</div>';
        return;
      }
      list.innerHTML = rows.map((item) => {
        const counts = Object.entries(item.message_counts || {}).map(([k, v]) => `${k}:${v}`).join(" / ") || "无 joint 计数";
        const topics = (item.available_topics || []).join(", ") || "无可用 joint topic";
        const selectedClass = selected && selected.name === item.name ? " selected" : "";
        const safeName = escapeHtml(item.name);
        const safeTime = escapeHtml(item.mtime_text);
        const safeCounts = escapeHtml(counts);
        const safeTopics = escapeHtml(topics);
        return `
          <div class="archive-row${selectedClass}" data-name="${safeName}">
            <div>
              <div class="archive-title">${safeName}</div>
              <div class="archive-meta">${safeTime} · ${fmtDuration(item.duration_s)} · ${safeCounts}<br>${safeTopics}</div>
            </div>
            <div class="badges">
              <span class="badge ${item.has_scene ? "ok" : "warn"}">${item.has_scene ? "scene" : "no scene"}</span>
              <span class="badge ${item.raw_bag_exists ? "ok" : ""}">${item.raw_bag_exists ? "raw" : "archive"}</span>
              ${item.kept ? '<span class="badge ok">kept</span>' : ''}
            </div>
          </div>`;
      }).join("");
      list.querySelectorAll(".archive-row").forEach((row) => {
        row.addEventListener("click", () => selectArchive(row.dataset.name));
      });
    }

    function selectArchive(name) {
      selected = archives.find((item) => item.name === name) || null;
      if (selected) {
        $("selectedName").textContent = selected.name;
        $("selectedMeta").textContent = `${selected.mtime_text} · ${fmtDuration(selected.duration_s)} · ${selected.has_scene ? "可恢复初始场景" : "没有 initial_scene.json"}`;
      }
      renderArchives();
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || `请求失败: ${response.status}`);
      }
      return payload;
    }

    function replayPayload() {
      if (!selected) throw new Error("请先选择数据集");
      return {
        archive: selected.name,
        source: $("sourceInput").value,
        speed: Number($("speedInput").value || 1.0),
        render_fps: Number($("fpsInput").value || 30),
        settle_seconds: Number($("settleInput").value || 0.5),
        headless: $("headlessInput").checked,
        no_realtime: $("noRealtimeInput").checked,
      };
    }

    async function loadArchives() {
      const payload = await requestJson("/api/archives");
      archives = payload.archives || [];
      if (selected && !archives.find((item) => item.name === selected.name)) {
        selected = null;
        $("selectedName").textContent = "未选择数据集";
        $("selectedMeta").textContent = "请从左侧选择一个 teaching_* 归档。";
      }
      renderArchives();
    }

    async function updateStatus() {
      try {
        const payload = await requestJson("/api/status");
        const status = payload.status || "idle";
        $("statusText").textContent = status;
        $("runningText").textContent = payload.archive ? `当前数据: ${payload.archive}` : "";
        $("logBox").textContent = (payload.logs || []).join("\n");
        $("logBox").scrollTop = $("logBox").scrollHeight;
      } catch (err) {
        $("statusText").textContent = "error";
        $("runningText").textContent = err.message;
      }
    }

    async function startReplay() {
      await requestJson("/api/replay/start", { method: "POST", body: JSON.stringify(replayPayload()) });
      await updateStatus();
    }

    $("refreshBtn").addEventListener("click", loadArchives);
    $("clearBtn").addEventListener("click", () => { $("searchInput").value = ""; renderArchives(); });
    $("searchInput").addEventListener("input", renderArchives);
    $("sortSelect").addEventListener("change", renderArchives);
    $("playBtn").addEventListener("click", async () => {
      try { await startReplay(); } catch (err) { alert(err.message); }
    });
    $("replayBtn").addEventListener("click", async () => {
      try {
        await requestJson("/api/replay/stop", { method: "POST", body: "{}" });
        await startReplay();
      } catch (err) { alert(err.message); }
    });
    $("stopBtn").addEventListener("click", async () => {
      try {
        await requestJson("/api/replay/stop", { method: "POST", body: "{}" });
        await updateStatus();
      } catch (err) { alert(err.message); }
    });
    $("keepBtn").addEventListener("click", async () => {
      try {
        if (!selected) throw new Error("请先选择数据集");
        await requestJson("/api/archive/keep", { method: "POST", body: JSON.stringify({ archive: selected.name }) });
        await loadArchives();
      } catch (err) { alert(err.message); }
    });
    $("deleteBtn").addEventListener("click", async () => {
      try {
        if (!selected) throw new Error("请先选择数据集");
        if (!confirm(`确认删除 ${selected.name}？该操作会删除 archive 和同名 raw rosbag。`)) return;
        await requestJson("/api/archive/delete", { method: "POST", body: JSON.stringify({ archive: selected.name }) });
        selected = null;
        await loadArchives();
        await updateStatus();
      } catch (err) { alert(err.message); }
    });

    loadArchives().catch((err) => alert(err.message));
    updateStatus();
    setInterval(updateStatus, 1000);
  </script>
</body>
</html>
"""


class ReplayGuiHandler(BaseHTTPRequestHandler):
    state: ReplayGuiState

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        if not data:
            return {}
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON object")
        return payload

    def _handle_api(self, method: str, path: str) -> None:
        try:
            if method == "GET" and path == "/api/archives":
                self._send_json({"ok": True, "archives": self.state.list_archives()})
                return
            if method == "GET" and path == "/api/status":
                self._send_json({"ok": True, **self.state.get_status()})
                return
            if method == "POST" and path == "/api/replay/start":
                self._send_json({"ok": True, **self.state.start_replay(self._read_payload())})
                return
            if method == "POST" and path == "/api/replay/stop":
                self._send_json({"ok": True, **self.state.stop_replay()})
                return
            if method == "POST" and path == "/api/archive/delete":
                self._send_json({"ok": True, **self.state.delete_archive(self._read_payload())})
                return
            if method == "POST" and path == "/api/archive/keep":
                self._send_json({"ok": True, **self.state.keep_archive(self._read_payload())})
                return
            self._send_json({"ok": False, "error": "未知 API"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(HTML_PAGE)
            return
        if path.startswith("/api/"):
            self._handle_api("GET", path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._handle_api("POST", path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 MuJoCo teaching 数据回放 Web GUI")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT, help="归档根目录")
    parser.add_argument("--bag-root", type=Path, default=DEFAULT_BAG_ROOT, help="原始 rosbag 根目录")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo XML 模型路径")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="监听端口，默认 8765")
    parser.add_argument("--open-browser", action="store_true", help="启动后自动打开浏览器")
    return parser


def main() -> None:
    _enforce_system_python()
    args = build_arg_parser().parse_args()

    state = ReplayGuiState(
        archive_root=args.archive_root,
        bag_root=args.bag_root,
        model_path=args.model,
    )
    ReplayGuiHandler.state = state

    server = ThreadingHTTPServer((args.host, int(args.port)), ReplayGuiHandler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print("MuJoCo teaching replay Web GUI")
    print("=" * 72)
    print(f"访问地址: {url}")
    print(f"归档目录: {state.archive_root}")
    print(f"原始 rosbag: {state.bag_root}")
    print(f"模型文件: {state.model_path}")
    print("按 Ctrl+C 退出")
    print("=" * 72)

    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n用户中断 GUI")
    finally:
        state.stop_replay()
        server.server_close()


if __name__ == "__main__":
    main()
