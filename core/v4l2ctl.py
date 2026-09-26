"""Read and write a V4L2 camera's controls with plain ioctls (no v4l2-ctl: the app's container has none).

core.capture snapshots the camera's controls when it first opens (after scripts/camera_setup.sh set
them) and writes them back when the camera reconnects: a UVC camera that drops off USB comes back with
its defaults (the Brio: auto exposure and autofocus on), which would drop the frame rate in dim light
and change the picture under the tracker. Linux only; elsewhere every call is a no-op.

    python -m core.v4l2ctl /dev/v4l/by-id/usb-...-video-index0     # print the controls and values
"""
from __future__ import annotations

import errno
import logging
import os
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)

_QUERYCTRL_FMT = "II32siiiiI2I"            # struct v4l2_queryctrl: 68 bytes
_CONTROL_FMT = "Ii"                         # struct v4l2_control: id, value


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_G_CTRL = _iowr(27, struct.calcsize(_CONTROL_FMT))
VIDIOC_S_CTRL = _iowr(28, struct.calcsize(_CONTROL_FMT))
VIDIOC_QUERYCTRL = _iowr(36, struct.calcsize(_QUERYCTRL_FMT))
NEXT_CTRL = 0x80000000                      # V4L2_CTRL_FLAG_NEXT_CTRL
FLAG_DISABLED, FLAG_READ_ONLY, FLAG_INACTIVE, FLAG_WRITE_ONLY = 0x1, 0x4, 0x10, 0x40
TYPE_BUTTON, TYPE_CTRL_CLASS = 4, 6
SETTABLE_TYPES = (1, 2, 3)                  # integer, boolean, menu


@dataclass
class Control:
    id: int
    name: str
    value: int


def _key(name: str) -> str:
    return name.strip().lower().replace(",", "").replace(" ", "_")


def _mode_first(c: Control) -> tuple:
    """Auto/manual switches before the values they gate: auto_exposure before exposure_time_absolute,
    focus_automatic_continuous before focus_absolute, white_balance_automatic before its temperature."""
    k = _key(c.name)
    return (0 if ("auto" in k or "dynamic" in k) else 1, c.id)


def snapshot(path: str, ioctl: Optional[Callable] = None) -> list[Control]:
    """Every settable control's current value, auto/manual switches first. [] off Linux or on error."""
    if ioctl is None:
        if not sys.platform.startswith("linux"):
            return []
        import fcntl
        ioctl = fcntl.ioctl
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError as ex:
        log.warning("camera controls: can't open %s (%s)", path, ex)
        return []
    out: list[Control] = []
    try:
        cid = NEXT_CTRL
        for _ in range(512):
            buf = bytearray(struct.pack(_QUERYCTRL_FMT, cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
            try:
                ioctl(fd, VIDIOC_QUERYCTRL, buf)
            except OSError as ex:
                if ex.errno != errno.EINVAL:
                    log.warning("camera controls: query failed on %s (%s)", path, ex)
                break                                   # EINVAL: past the last control
            qid, typ, name, _mn, _mx, _step, _default, flags, _r0, _r1 = struct.unpack(_QUERYCTRL_FMT, bytes(buf))
            cid = qid | NEXT_CTRL
            if typ not in SETTABLE_TYPES or flags & (FLAG_DISABLED | FLAG_READ_ONLY | FLAG_WRITE_ONLY):
                continue
            c = bytearray(struct.pack(_CONTROL_FMT, qid, 0))
            try:
                ioctl(fd, VIDIOC_G_CTRL, c)
            except OSError:
                continue
            out.append(Control(qid, name.split(b"\0", 1)[0].decode(errors="replace"),
                               struct.unpack(_CONTROL_FMT, bytes(c))[1]))
    finally:
        os.close(fd)
    return sorted(out, key=_mode_first)


def restore(path: str, controls: list[Control], ioctl: Optional[Callable] = None) -> int:
    """Write the snapshot back, auto/manual switches first; returns how many controls took. A control
    the camera refuses (e.g. inactive while auto is on) is skipped."""
    if not controls:
        return 0
    if ioctl is None:
        if not sys.platform.startswith("linux"):
            return 0
        import fcntl
        ioctl = fcntl.ioctl
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError as ex:
        log.warning("camera controls: can't open %s to restore (%s)", path, ex)
        return 0
    done = 0
    try:
        for c in sorted(controls, key=_mode_first):
            try:
                ioctl(fd, VIDIOC_S_CTRL, bytearray(struct.pack(_CONTROL_FMT, c.id, c.value)))
                done += 1
            except OSError:
                log.debug("camera controls: %s=%d refused", c.name, c.value)
    finally:
        os.close(fd)
    return done


def device_path(source) -> Optional[str]:
    """The device file for a camera source: a path as is, an index as /dev/videoN, else None."""
    if isinstance(source, int):
        return f"/dev/video{source}"
    if isinstance(source, str):
        return f"/dev/video{source}" if source.isdigit() else source
    return None


if __name__ == "__main__":
    for ctl in snapshot(sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"):
        print(f"{_key(ctl.name):40s} {ctl.value}")
