"""eval.record --camera takes an index or a /dev/v4l/by-id/ path (the Brio's index moves on replug)."""
from eval.record import camera_arg


def test_camera_arg_index_or_path():
    assert camera_arg("0") == 0
    assert camera_arg(" 2 ") == 2
    p = "/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0"
    assert camera_arg(p) == p
