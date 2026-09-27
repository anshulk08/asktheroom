"""core/v4l2ctl.py with a fake ioctl: snapshot and restore a camera's controls, auto switches first."""
import errno
import struct

from core import v4l2ctl as V

AUTO_EXPOSURE, EXPOSURE, GAIN, ZOOM, BUTTON, READ_ONLY = 0x9A0901, 0x9A0902, 0x980913, 0x9A090D, 0x980921, 0x980922
FOCUS, FOCUS_AUTO = 0x9A090A, 0x9A090C          # real UVC ids: the auto switch sorts after the value


class FakeCam:
    """A control table; exposure_time_absolute refuses writes while auto_exposure is 3 (auto), as UVC does."""
    def __init__(self):
        self.ctrls = {AUTO_EXPOSURE: ("Auto Exposure", 3, 0, 1), EXPOSURE: ("Exposure Time, Absolute", 1, 0, 333),
                      GAIN: ("Gain", 1, 0, 96), ZOOM: ("Zoom, Absolute", 1, 0, 120),
                      FOCUS: ("Focus, Absolute", 1, 0, 40), FOCUS_AUTO: ("Focus, Automatic Continuous", 2, 0, 0),
                      BUTTON: ("Reset", V.TYPE_BUTTON, 0, 0), READ_ONLY: ("Status", 1, V.FLAG_READ_ONLY, 7)}
        self.writes = []

    def value(self, cid):
        return self.ctrls[cid][3]

    def __call__(self, fd, req, buf):
        if req == V.VIDIOC_QUERYCTRL:
            want = struct.unpack(V._QUERYCTRL_FMT, bytes(buf))[0] & ~V.NEXT_CTRL
            nxt = [c for c in sorted(self.ctrls) if c > want]
            if not nxt:
                raise OSError(errno.EINVAL, "end")
            name, typ, flags, _ = self.ctrls[nxt[0]]
            buf[:] = struct.pack(V._QUERYCTRL_FMT, nxt[0], typ, name.encode(), 0, 1000, 1, 0, flags, 0, 0)
        elif req == V.VIDIOC_G_CTRL:
            cid = struct.unpack(V._CONTROL_FMT, bytes(buf))[0]
            buf[:] = struct.pack(V._CONTROL_FMT, cid, self.value(cid))
        elif req == V.VIDIOC_S_CTRL:
            cid, val = struct.unpack(V._CONTROL_FMT, bytes(buf))
            if (cid == EXPOSURE and self.value(AUTO_EXPOSURE) == 3) or (cid == FOCUS and self.value(FOCUS_AUTO) == 1):
                raise OSError(errno.EACCES, "inactive")
            name, typ, flags, _ = self.ctrls[cid]
            self.ctrls[cid] = (name, typ, flags, val)
            self.writes.append(cid)


def test_a_reconnected_camera_gets_its_settings_back_auto_switch_first(tmp_path):
    dev = tmp_path / "video0"
    dev.write_bytes(b"")
    cam = FakeCam()
    snap = V.snapshot(str(dev), ioctl=cam)
    assert {c.id for c in snap} == {AUTO_EXPOSURE, EXPOSURE, GAIN, ZOOM, FOCUS, FOCUS_AUTO}   # no button/read-only
    fresh = FakeCam()                          # plugged back in: defaults, auto exposure and autofocus on
    fresh.ctrls[AUTO_EXPOSURE] = ("Auto Exposure", 3, 0, 3)
    fresh.ctrls[FOCUS_AUTO] = ("Focus, Automatic Continuous", 2, 0, 1)
    fresh.ctrls[FOCUS] = ("Focus, Absolute", 1, 0, 0)
    fresh.ctrls[GAIN] = ("Gain", 1, 0, 0)
    assert V.restore(str(dev), snap, ioctl=fresh) == 6
    assert fresh.writes.index(FOCUS_AUTO) < fresh.writes.index(FOCUS)          # manual first, or focus is refused
    assert (fresh.value(AUTO_EXPOSURE), fresh.value(EXPOSURE), fresh.value(GAIN)) == (1, 333, 96)
    assert (fresh.value(FOCUS_AUTO), fresh.value(FOCUS)) == (0, 40)


def test_a_missing_device_gives_an_empty_snapshot_and_restores_nothing(tmp_path):
    assert V.snapshot(str(tmp_path / "nope"), ioctl=FakeCam()) == []
    assert V.restore(str(tmp_path / "nope"), [V.Control(GAIN, "Gain", 96)], ioctl=FakeCam()) == 0


def test_device_path_for_indexes_and_paths():
    assert V.device_path(0) == "/dev/video0" and V.device_path("2") == "/dev/video2"
    assert V.device_path("/dev/v4l/by-id/x") == "/dev/v4l/by-id/x" and V.device_path(object()) is None
