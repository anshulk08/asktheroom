import pytest

from core.geom import area, center, contains_point, dist, intersection, iou, overlap_frac, shift


def test_area_of_box():
    assert area((0, 0, 4, 3)) == 12


def test_area_of_inverted_box_is_zero():
    assert area((4, 3, 0, 0)) == 0


def test_intersection_of_overlapping_boxes():
    assert intersection((0, 0, 4, 4), (2, 2, 6, 6)) == (2, 2, 4, 4)


def test_intersection_of_disjoint_boxes_is_none():
    assert intersection((0, 0, 1, 1), (2, 2, 3, 3)) is None


def test_overlap_frac_is_share_of_second_box_covered():
    # hand (first) covers the right half of the object (second)
    assert overlap_frac((2, 0, 10, 4), (0, 0, 4, 4)) == pytest.approx(0.5)


def test_overlap_frac_disjoint_is_zero():
    assert overlap_frac((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == pytest.approx(1.0)


def test_iou_half_shifted_boxes():
    assert iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)


def test_center_and_dist():
    assert center((0, 0, 4, 2)) == (2, 1)
    assert dist((0, 0), (3, 4)) == pytest.approx(5)


def test_contains_point_inclusive_edges():
    assert contains_point((0, 0, 4, 4), (4, 2))
    assert not contains_point((0, 0, 4, 4), (5, 2))


def test_shift_moves_box():
    assert shift((0, 0, 2, 2), (1.5, -1)) == (1.5, -1, 3.5, 1)
