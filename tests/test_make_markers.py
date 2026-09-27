"""scripts/make_markers.py: the printed patterns decode as the right tags."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_markers as M  # noqa: E402


def decode(grid, family):
    n = len(grid)
    img = np.array([[0 if b else 255 for b in row] for row in grid], np.uint8)
    img = cv2.resize(img, (n * 20, n * 20), interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(family), cv2.aruco.DetectorParameters())
    _, ids, _ = det.detectMarkers(img)
    return None if ids is None else [int(i) for i in ids.flatten()]


def test_table_marker_cells_decode():
    assert decode(M.cells(2), cv2.aruco.DICT_4X4_50) == [2]


def test_the_one_big_apriltag_decodes_and_prints_at_the_asked_size():
    grid = M.cells(0, "apriltag_36h11")
    assert len(grid) == 8 and decode(grid, cv2.aruco.DICT_APRILTAG_36h11) == [0]
    pdf = M.pdf_tag(0, 160.0)
    assert pdf.startswith(b"%PDF") and b"AprilTag 36h11 ID 0" in pdf and b"160 mm" in pdf
