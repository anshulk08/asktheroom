"""Pointing geometry for the stepper turret (actuator: turret): pan/tilt degrees that put the laser
beam through a 3D point. Owner: A. Pure math, no hardware; tests in tests/test_pointing.py.

Turret frame (meters), origin where the pan and tilt axes meet, at pan=0 tilt=0:
    x = right, y = up, z = forward
Angles match firmware/turret: + pan turns right, + tilt looks up.

The laser can sit off the rotation center (laser_offset, meters in the tilting head's
own x/y/z). At short range that parallax matters, so solve_aim() iterates until the
beam itself, not the center of rotation, passes through the target.

Cameras:
  * Camera(...) with a fixed mount pose turns a pixel plus a depth into a turret-frame
    point: cam.pixel_to_point(u, v, depth) -> feed to solve_aim().
  * A camera riding on the tilt head uses cam.pixel_to_angles(u, v, pan, tilt) for
    "where to point so this pixel ends up centered".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

DegLimits = tuple[tuple[float, float], tuple[float, float]]
# The firmware's own soft limits (MIN_DEG/MAX_DEG in firmware/turret/turret.ino).
FIRMWARE_LIMITS_DEG: DegLimits = ((-90.0, 90.0), (-90.0, 90.0))


def head_axes(pan_deg, tilt_deg):
    """Unit vectors (right, up, forward) of the tilting head, in the turret frame."""
    p, t = math.radians(pan_deg), math.radians(tilt_deg)
    sp, cp, st, ct = math.sin(p), math.cos(p), math.sin(t), math.cos(t)
    right = (cp, 0.0, -sp)
    up = (-st * sp, ct, -st * cp)
    forward = (sp * ct, st, cp * ct)
    return right, up, forward


def head_to_turret(vec, pan_deg, tilt_deg):
    """Rotate a vector given in head coordinates (x right, y up, z forward) into the turret frame."""
    r, u, f = head_axes(pan_deg, tilt_deg)
    return tuple(vec[0] * r[i] + vec[1] * u[i] + vec[2] * f[i] for i in range(3))


def direction_angles(vec):
    """(pan, tilt) in degrees that point the head's forward axis along vec."""
    x, y, z = vec
    horizontal = math.hypot(x, z)
    if horizontal == 0 and y == 0:
        raise ValueError("zero-length direction")
    return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(y, horizontal))


@dataclass
class Aim:
    pan: float
    tilt: float
    reachable: bool      # within the limits
    miss_m: float        # distance from the beam to the target at this solution


def beam(pan_deg, tilt_deg, laser_offset=(0.0, 0.0, 0.0)):
    """Origin and unit direction of the laser beam in the turret frame."""
    origin = head_to_turret(laser_offset, pan_deg, tilt_deg)
    _, _, forward = head_axes(pan_deg, tilt_deg)
    return origin, forward


def beam_miss(pan_deg, tilt_deg, target, laser_offset=(0.0, 0.0, 0.0)):
    """Perpendicular distance (m) from the beam to target; inf if the target is behind the laser."""
    o, d = beam(pan_deg, tilt_deg, laser_offset)
    rel = tuple(target[i] - o[i] for i in range(3))
    along = sum(rel[i] * d[i] for i in range(3))
    if along <= 0:
        return math.inf
    perp = tuple(rel[i] - along * d[i] for i in range(3))
    return math.sqrt(sum(c * c for c in perp))


def solve_aim(target, laser_offset=(0.0, 0.0, 0.0), limits: Optional[DegLimits] = None,
              iterations=20, tol_m=1e-9) -> Aim:
    """Pan/tilt that put the laser beam through `target` (x, y, z in meters, turret frame).
    `limits` ((pan_lo, pan_hi), (tilt_lo, tilt_hi)) in degrees only sets Aim.reachable."""
    if math.sqrt(sum(c * c for c in target)) <= math.sqrt(sum(c * c for c in laser_offset)):
        raise ValueError("target is inside the laser's offset radius")
    pan, tilt = direction_angles(target)
    for _ in range(iterations):
        # The beam is origin + s*forward with origin = ox*right + oy*up + oz*forward, so it
        # hits the target when forward is parallel to target - (ox*right + oy*up).
        # right/up depend on the angles, hence the fixed-point iteration.
        lateral = head_to_turret((laser_offset[0], laser_offset[1], 0.0), pan, tilt)
        want = tuple(target[i] - lateral[i] for i in range(3))
        new_pan, new_tilt = direction_angles(want)
        done = abs(new_pan - pan) < 1e-9 and abs(new_tilt - tilt) < 1e-9
        pan, tilt = new_pan, new_tilt
        if done or beam_miss(pan, tilt, target, laser_offset) < tol_m:
            break
    (plo, phi), (tlo, thi) = limits or FIRMWARE_LIMITS_DEG
    reachable = plo <= pan <= phi and tlo <= tilt <= thi
    return Aim(pan, tilt, reachable, beam_miss(pan, tilt, target, laser_offset))


@dataclass
class Camera:
    """Pinhole camera. Pose is relative to the turret frame; yaw/pitch use the same sign
    conventions as pan/tilt (+yaw looks right, +pitch looks up)."""
    width: int
    height: int
    hfov_deg: float
    position: tuple = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0

    @property
    def fx(self):
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def cx(self):
        return self.width / 2

    @property
    def cy(self):
        return self.height / 2

    def pixel_ray(self, u, v):
        """Unnormalized ray through pixel (u, v) in camera coordinates (x right, y up, z forward).
        Assumes square pixels, so fy == fx."""
        return ((u - self.cx) / self.fx, -(v - self.cy) / self.fx, 1.0)

    def pixel_to_point(self, u, v, depth_m):
        """Turret-frame point for pixel (u, v) at `depth_m` along the camera's optical axis."""
        ray = self.pixel_ray(u, v)
        cam_pt = tuple(c * depth_m for c in ray)
        world = head_to_turret(cam_pt, self.yaw_deg, self.pitch_deg)
        return tuple(world[i] + self.position[i] for i in range(3))

    def point_to_pixel(self, point):
        """Inverse of pixel_to_point: turret-frame point -> (u, v, depth)."""
        rel = tuple(point[i] - self.position[i] for i in range(3))
        r, u_ax, f = head_axes(self.yaw_deg, self.pitch_deg)
        x = sum(rel[i] * r[i] for i in range(3))
        y = sum(rel[i] * u_ax[i] for i in range(3))
        z = sum(rel[i] * f[i] for i in range(3))
        return self.cx + self.fx * x / z, self.cy - self.fx * y / z, z

    def pixel_to_angles(self, u, v, pan_deg, tilt_deg):
        """For a camera riding on the tilt head (looking along the laser), currently at
        pan_deg/tilt_deg: the absolute (pan, tilt) that would center pixel (u, v).
        Exact when the camera sits at the rotation center; close for small offsets."""
        return direction_angles(head_to_turret(self.pixel_ray(u, v), pan_deg, tilt_deg))
