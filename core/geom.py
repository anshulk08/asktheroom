"""Box and point helpers. Boxes are (x1, y1, x2, y2) in any consistent unit."""
from __future__ import annotations

import math

Box = tuple[float, float, float, float]
Point = tuple[float, float]


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> Box | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def overlap_frac(a: Box, b: Box) -> float:
    """Share of box b covered by box a (0-1). 'Hand covers object' = overlap_frac(hand, obj)."""
    inter = intersection(a, b)
    if inter is None or area(b) == 0:
        return 0.0
    return area(inter) / area(b)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    if inter is None:
        return 0.0
    i = area(inter)
    return i / (area(a) + area(b) - i)


def center(b: Box) -> Point:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def dist(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def contains_point(b: Box, p: Point) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def shift(b: Box, d: Point) -> Box:
    return (b[0] + d[0], b[1] + d[1], b[2] + d[0], b[3] + d[1])
