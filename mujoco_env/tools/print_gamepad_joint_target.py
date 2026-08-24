#!/usr/bin/env python3
"""
Read and print Xbox/Linux joystick data.

This tool uses the Linux joystick API directly (/dev/input/js*) and does not
require pygame, evdev, ROS2, or any third-party package.

Examples:
    python3 mujoco_env/tools/print_gamepad_joint_target.py --list
    python3 mujoco_env/tools/print_gamepad_joint_target.py
    python3 mujoco_env/tools/print_gamepad_joint_target.py --device /dev/input/js1 --events
"""

from __future__ import annotations

import argparse
import array
import errno
import fcntl
import glob
import os
import select
import struct
import sys
import time
from dataclasses import dataclass
from typing import Iterable


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12

DEFAULT_AXIS_NAMES = [
    "left_x",
    "left_y",
    "left_trigger",
    "right_x",
    "right_y",
    "right_trigger",
    "dpad_x",
    "dpad_y",
]

DEFAULT_BUTTON_NAMES = [
    "A",
    "B",
    "X",
    "Y",
    "LB",
    "RB",
    "View/Back",
    "Menu/Start",
    "Guide",
    "LeftStick",
    "RightStick",
]


@dataclass
class JoystickInfo:
    path: str
    name: str
    axis_count: int
    button_count: int


def _jsiocgname(length: int) -> int:
    """Build JSIOCGNAME(len), equivalent to Linux _IOC(_IOC_READ, 'j', 0x13, len)."""
    return 0x80000000 | (length << 16) | (ord("j") << 8) | 0x13


def _read_u8_ioctl(fd: int, request: int) -> int:
    buf = array.array("B", [0])
    fcntl.ioctl(fd, request, buf, True)
    return int(buf[0])


def _read_name_ioctl(fd: int) -> str:
    buf = array.array("B", [0] * 128)
    try:
        fcntl.ioctl(fd, _jsiocgname(len(buf)), buf, True)
    except OSError:
        return "Unknown joystick"

    raw = bytes(buf)
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace") or "Unknown joystick"


def discover_devices() -> list[str]:
    return sorted(glob.glob("/dev/input/js*"))


def read_joystick_info(path: str) -> JoystickInfo:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        return JoystickInfo(
            path=path,
            name=_read_name_ioctl(fd),
            axis_count=_read_u8_ioctl(fd, JSIOCGAXES),
            button_count=_read_u8_ioctl(fd, JSIOCGBUTTONS),
        )
    finally:
        os.close(fd)


def iter_joystick_infos(paths: Iterable[str]) -> Iterable[JoystickInfo | tuple[str, OSError]]:
    for path in paths:
        try:
            yield read_joystick_info(path)
        except OSError as exc:
            yield (path, exc)


def axis_name(index: int) -> str:
    if index < len(DEFAULT_AXIS_NAMES):
        return DEFAULT_AXIS_NAMES[index]
    return f"axis_{index}"


def button_name(index: int) -> str:
    if index < len(DEFAULT_BUTTON_NAMES):
        return DEFAULT_BUTTON_NAMES[index]
    return f"button_{index}"


def normalize_axis(value: int, deadzone: float) -> float:
    if value < 0:
        normalized = value / 32768.0
    else:
        normalized = value / 32767.0

    normalized = max(-1.0, min(1.0, normalized))
    if abs(normalized) < deadzone:
        return 0.0
    return normalized


def format_axis_bar(value: float, width: int = 20) -> str:
    midpoint = width // 2
    pos = int(round((value + 1.0) * midpoint))
    pos = max(0, min(width, pos))

    chars = [" "] * (width + 1)
    chars[midpoint] = "|"
    chars[pos] = "*"
    return "".join(chars)


def print_device_list() -> int:
    devices = discover_devices()
    if not devices:
        print_no_device_help()
        return 2

    for item in iter_joystick_infos(devices):
        if isinstance(item, JoystickInfo):
            print(f"{item.path}: {item.name} ({item.axis_count} axes, {item.button_count} buttons)")
        else:
            path, exc = item
            print(f"{path}: cannot read ({exc})")
    return 0


def print_no_device_help() -> None:
    print("No joystick device found under /dev/input/js*.")
    print()
    print("Check these on the machine that has the Xbox controller connected:")
    print("  ls -l /dev/input/js*")
    print("  sudo modprobe joydev")
    print("  sudo usermod -aG input $USER   # then log out and back in")
    print()
    print("If you are running inside Docker, pass the input devices into the container,")
    print("for example with --privileged or by mounting /dev/input.")


def open_selected_device(device: str | None) -> tuple[int, JoystickInfo]:
    if device is None:
        devices = discover_devices()
        if not devices:
            print_no_device_help()
            raise SystemExit(2)
        device = devices[0]

    info = read_joystick_info(device)
    fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    return fd, info


def print_state(
    info: JoystickInfo,
    axes: list[int],
    buttons: list[int],
    deadzone: float,
    last_events: list[str],
    clear: bool,
) -> None:
    if clear:
        print("\033[H\033[J", end="")

    now = time.strftime("%H:%M:%S")
    print(f"Xbox/gamepad data monitor  {now}")
    print(f"Device: {info.path}")
    print(f"Name:   {info.name}")
    print("Press Ctrl+C to exit.")
    print()

    print("Axes:")
    for index, raw_value in enumerate(axes):
        normalized = normalize_axis(raw_value, deadzone)
        bar = format_axis_bar(normalized)
        print(f"  {index:02d} {axis_name(index):>14}: raw={raw_value:6d} norm={normalized:+.3f} [{bar}]")

    pressed = [button_name(i) for i, value in enumerate(buttons) if value]
    print()
    print("Buttons:")
    print(f"  pressed: {', '.join(pressed) if pressed else '(none)'}")
    for index, value in enumerate(buttons):
        print(f"  {index:02d} {button_name(index):>14}: {value}")

    if last_events:
        print()
        print("Recent events:")
        for line in last_events[-10:]:
            print(f"  {line}")

    sys.stdout.flush()


def event_label(event_type: int, number: int, value: int, deadzone: float) -> str:
    event_kind = event_type & ~JS_EVENT_INIT
    init_prefix = "init " if event_type & JS_EVENT_INIT else ""

    if event_kind == JS_EVENT_AXIS:
        normalized = normalize_axis(value, deadzone)
        return f"{init_prefix}axis {number:02d} {axis_name(number)} raw={value} norm={normalized:+.3f}"
    if event_kind == JS_EVENT_BUTTON:
        return f"{init_prefix}button {number:02d} {button_name(number)} value={value}"
    return f"{init_prefix}unknown type=0x{event_type:02x} number={number} value={value}"


def monitor(args: argparse.Namespace) -> int:
    fd, info = open_selected_device(args.device)
    axes = [0] * info.axis_count
    buttons = [0] * info.button_count
    last_events: list[str] = []
    next_print = 0.0
    print_interval = 1.0 / max(args.rate, 1.0)
    clear = not args.no_clear and sys.stdout.isatty() and not args.events

    print(f"Opened {info.path}: {info.name} ({info.axis_count} axes, {info.button_count} buttons)")

    try:
        while True:
            timeout = max(0.0, next_print - time.monotonic())
            readable, _, _ = select.select([fd], [], [], min(timeout, 0.05))

            if readable:
                while True:
                    try:
                        data = os.read(fd, 8)
                    except BlockingIOError:
                        break
                    except OSError as exc:
                        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                            break
                        raise

                    if len(data) != 8:
                        break

                    _event_time_ms, value, event_type, number = struct.unpack("IhBB", data)
                    event_kind = event_type & ~JS_EVENT_INIT

                    if event_kind == JS_EVENT_AXIS and number < len(axes):
                        axes[number] = value
                    elif event_kind == JS_EVENT_BUTTON and number < len(buttons):
                        buttons[number] = value

                    label = event_label(event_type, number, value, args.deadzone)
                    last_events.append(label)
                    if len(last_events) > 20:
                        del last_events[:-20]

                    if args.events:
                        print(label, flush=True)

            now = time.monotonic()
            if not args.events and now >= next_print:
                print_state(info, axes, buttons, args.deadzone, last_events, clear)
                next_print = now + print_interval

    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        os.close(fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and print Xbox/Linux gamepad data.")
    parser.add_argument("--device", help="Joystick device path, for example /dev/input/js0.")
    parser.add_argument("--list", action="store_true", help="List detected /dev/input/js* devices and exit.")
    parser.add_argument("--rate", type=float, default=20.0, help="Screen refresh rate in Hz. Default: 20.")
    parser.add_argument("--deadzone", type=float, default=0.08, help="Normalized axis deadzone. Default: 0.08.")
    parser.add_argument("--events", action="store_true", help="Print one line per input event instead of refreshing a status screen.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between status frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        return print_device_list()
    return monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
