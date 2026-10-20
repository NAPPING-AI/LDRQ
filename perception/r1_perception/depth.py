"""Depth inside a box, and what is in the corridor ahead. numpy only.

Depth is the rectified depth image in meters along the optical axis (z), 0 or
non-finite where the sensor has no return. Raw sensor units are converted before this
module sees them (sources.py: 16UC1 millimeters, 32FC1 meters).
"""
from __future__ import annotations

import numpy as np


def box_depth(depth, box, shrink=0.5, dmin=0.15, dmax=10.0, min_valid=0.1):
    """Median depth of the central part of a box, or None.

    `shrink` keeps the middle fraction of the box in each direction: a box is a
    rectangle around a shape that is not one, and its corners are background. None when
    fewer than `min_valid` of those pixels have a return -- a median of three pixels is
    a guess, and walk_to must not stop on a guess."""
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * shrink / 2.0, (y1 - y0) * shrink / 2.0
    a0, a1 = int(max(0, np.floor(cx - hw))), int(min(w, np.ceil(cx + hw)))
    b0, b1 = int(max(0, np.floor(cy - hh))), int(min(h, np.ceil(cy + hh)))
    if a1 <= a0 or b1 <= b0:
        return None
    patch = np.asarray(depth[b0:b1, a0:a1], dtype=np.float64).ravel()
    ok = np.isfinite(patch) & (patch >= dmin) & (patch <= dmax)
    if ok.sum() < max(1, min_valid * patch.size):
        return None
    return float(np.median(patch[ok]))


def corridor(depth, rect, mount, stop_m, half_width_m=0.35, min_height_m=0.10,
             max_height_m=1.9, step=4, min_points=30, dmax=10.0):
    """The nearest return inside the box the robot is about to walk into.

    Every `step`-th pixel is turned into a body-frame point (forward, left, up). A point
    counts if it is ahead, within `half_width_m` of the walking line, and between
    `min_height_m` and `max_height_m` above the floor -- so the floor itself never does,
    however the camera is pitched, and neither does the ceiling. Returns (obstacle,
    nearest_forward_m or None, points counted). `obstacle` needs `min_points` inside
    `stop_m`, so one speckle of sensor noise is not a wall."""
    d = np.asarray(depth[::step, ::step], dtype=np.float64)
    hh, ww = d.shape
    v, u = np.mgrid[0:hh, 0:ww]
    xn, yn = rect.ray(u * step, v * step)
    ok = np.isfinite(d) & (d > 0.05) & (d < dmax)
    if not ok.any():
        return False, None, 0
    z = d[ok]
    f, l, up = mount.to_body(xn[ok] * z, yn[ok] * z, z)
    inside = (f > 0.0) & (np.abs(l) <= half_width_m) & (up >= min_height_m) & \
        (up <= max_height_m)
    if not inside.any():
        return False, None, 0
    fwd = f[inside]
    near = int((fwd < stop_m).sum())
    return near >= min_points, float(np.percentile(fwd, 5)), near
