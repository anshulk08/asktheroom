"""act/pointing.py: turret pan/tilt geometry, parallax, cameras, firmware step rounding."""

import math
import random
import unittest

from act.pointing import Camera, beam_miss, direction_angles, head_axes, solve_aim

# Firmware quantization (firmware/turret): 3200 microsteps/rev through 4:1 pan, 3.2:1 tilt.
STEPS_PER_DEG = (3200 * 4.0 / 360, 3200 * 3.2 / 360)


def quantize(pan, tilt):
    return (round(pan * STEPS_PER_DEG[0]) / STEPS_PER_DEG[0],
            round(tilt * STEPS_PER_DEG[1]) / STEPS_PER_DEG[1])


class KnownGeometry(unittest.TestCase):
    def assertAim(self, target, pan, tilt, **kw):
        aim = solve_aim(target, **kw)
        self.assertAlmostEqual(aim.pan, pan, places=6, msg=f"pan for {target}")
        self.assertAlmostEqual(aim.tilt, tilt, places=6, msg=f"tilt for {target}")
        return aim

    def test_straight_ahead(self):
        self.assertAim((0, 0, 2), 0, 0)

    def test_right_and_left(self):
        self.assertAim((1, 0, 1), 45, 0)
        self.assertAim((-1, 0, 1), -45, 0)
        self.assertAim((1, 0, 0), 90, 0)

    def test_up_and_down(self):
        self.assertAim((0, 1, 1), 0, 45)
        self.assertAim((0, -1, 1), 0, -45)

    def test_combined(self):
        # 45 deg right, then 45 deg up: horizontal distance sqrt(2), height sqrt(2).
        self.assertAim((1, math.sqrt(2), 1), 45, 45)

    def test_behind(self):
        self.assertAim((1, 0, -1), 135, 0)

    def test_distance_does_not_matter_without_offset(self):
        for scale in (0.3, 1, 10, 100):
            self.assertAim((0.5 * scale, 0.2 * scale, 1 * scale), *direction_angles((0.5, 0.2, 1)))

    def test_limits_flag(self):
        self.assertFalse(solve_aim((0, -1, 0.1)).reachable)   # needs about -84 deg tilt
        self.assertFalse(solve_aim((-0.1, 0, -1)).reachable)  # needs about -174 deg pan
        self.assertTrue(solve_aim((0.3, 0.1, 2)).reachable)


class Parallax(unittest.TestCase):
    def test_laser_above_axis_aims_down_slightly(self):
        # Laser 5 cm above the tilt axis, target 1 m ahead at axis height.
        aim = solve_aim((0, 0, 1), laser_offset=(0, 0.05, 0))
        self.assertAlmostEqual(aim.tilt, -math.degrees(math.asin(0.05 / 1)), places=6)
        self.assertLess(aim.miss_m, 1e-9)

    def test_parallax_shrinks_with_distance(self):
        off = (0.03, 0.05, 0.02)
        near = solve_aim((0, 0, 0.5), laser_offset=off)
        far = solve_aim((0, 0, 50), laser_offset=off)
        self.assertGreater(abs(near.tilt), abs(far.tilt) * 50)

    def test_ignoring_offset_would_miss(self):
        off = (0.0, 0.05, 0.0)
        naive = solve_aim((0, 0, 1))                   # pretend the laser is on the axis
        self.assertAlmostEqual(beam_miss(naive.pan, naive.tilt, (0, 0, 1), off), 0.05, places=6)

    def test_random_targets_with_offset_are_hit(self):
        rng = random.Random(1)
        off = (0.02, 0.04, 0.03)
        worst = 0
        for _ in range(2000):
            pan, tilt = rng.uniform(-160, 160), rng.uniform(-40, 85)
            dist = rng.uniform(0.4, 20)
            _, _, fwd = head_axes(pan, tilt)
            target = tuple(dist * c for c in fwd)
            aim = solve_aim(target, laser_offset=off)
            worst = max(worst, aim.miss_m)
        self.assertLess(worst, 1e-6)


class Cameras(unittest.TestCase):
    def test_center_pixel_is_straight_ahead(self):
        cam = Camera(1280, 720, hfov_deg=70)
        aim = solve_aim(cam.pixel_to_point(640, 360, 3.0))
        self.assertAlmostEqual(aim.pan, 0, places=9)
        self.assertAlmostEqual(aim.tilt, 0, places=9)

    def test_image_edge_is_half_fov(self):
        cam = Camera(1280, 720, hfov_deg=70)
        pan, tilt = direction_angles(cam.pixel_to_point(1280, 360, 2.0))
        self.assertAlmostEqual(pan, 35, places=9)
        self.assertAlmostEqual(tilt, 0, places=9)
        _, tilt = direction_angles(cam.pixel_to_point(640, 0, 2.0))
        self.assertGreater(tilt, 0)   # top of image = up

    def test_offset_camera_round_trip(self):
        # Camera mounted 12 cm left, 8 cm above the turret, turned 10 deg right, 5 deg down.
        cam = Camera(1920, 1080, hfov_deg=90, position=(-0.12, 0.08, 0.0), yaw_deg=10, pitch_deg=-5)
        rng = random.Random(2)
        for _ in range(500):
            truth = (rng.uniform(-2, 2), rng.uniform(-0.5, 1.5), rng.uniform(1, 6))
            u, v, depth = cam.point_to_pixel(truth)
            back = cam.pixel_to_point(u, v, depth)
            for a, b in zip(back, truth):
                self.assertAlmostEqual(a, b, places=9)
            self.assertLess(beam_miss(*_angles(solve_aim(back)), truth), 1e-9)

    def test_offset_camera_matters(self):
        # Treating an offset camera as if it sat on the turret points at the wrong spot.
        cam = Camera(1280, 720, hfov_deg=70, position=(0.3, 0, 0))
        truth = (0, 0, 1.5)
        u, v, depth = cam.point_to_pixel(truth)
        naive = Camera(1280, 720, hfov_deg=70).pixel_to_point(u, v, depth)
        self.assertGreater(beam_miss(*_angles(solve_aim(naive)), truth), 0.25)

    def test_head_camera_centers_target(self):
        # Camera on the tilt head at the rotation center: after turning to pixel_to_angles,
        # the target must project to the image center.
        cam = Camera(1280, 720, hfov_deg=70)
        rng = random.Random(3)
        for _ in range(300):
            pan, tilt = rng.uniform(-90, 90), rng.uniform(-30, 60)
            u, v = rng.uniform(0, 1280), rng.uniform(0, 720)
            new_pan, new_tilt = cam.pixel_to_angles(u, v, pan, tilt)
            # Point that pixel was looking at, 4 m out, seen from the current pose.
            head_cam = Camera(1280, 720, 70, yaw_deg=pan, pitch_deg=tilt)
            target = head_cam.pixel_to_point(u, v, 4.0)
            recentered = Camera(1280, 720, 70, yaw_deg=new_pan, pitch_deg=new_tilt)
            cu, cv, _ = recentered.point_to_pixel(target)
            self.assertAlmostEqual(cu, 640, places=6)
            self.assertAlmostEqual(cv, 360, places=6)


class FirmwareQuantization(unittest.TestCase):
    def test_dot_error_from_step_rounding(self):
        rng = random.Random(4)
        worst = {1: 0, 3: 0, 10: 0}
        for _ in range(3000):
            pan, tilt = rng.uniform(-160, 160), rng.uniform(-40, 85)
            for dist in worst:
                _, _, fwd = head_axes(pan, tilt)
                target = tuple(dist * c for c in fwd)
                aim = solve_aim(target)
                worst[dist] = max(worst[dist], beam_miss(*quantize(aim.pan, aim.tilt), target))
        # Half a microstep on each axis at 10 m is about 4 mm.
        self.assertLess(worst[10], 0.004)


def _angles(aim):
    return aim.pan, aim.tilt
