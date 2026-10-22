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


def corridor_points(depth, rect, mount, half_width_m=0.35, min_height_m=0.10,
                    max_height_m=1.9, step=4, dmax=10.0):
    """Forward distances (m) of the returns inside the box the robot is about to walk
    into, and the fraction of sampled pixels with any return at all.

    Every `step`-th pixel is turned into a body-frame point (forward, left, up). A point
    counts if it is ahead, within `half_width_m` of the walking line, and between
    `min_height_m` and `max_height_m` above the floor -- so the floor itself never does,
    however the camera is pitched, and neither does the ceiling."""
    d = np.asarray(depth[::step, ::step], dtype=np.float64)
    hh, ww = d.shape
    v, u = np.mgrid[0:hh, 0:ww]
    xn, yn = rect.ray(u * step, v * step)
    ok = np.isfinite(d) & (d > 0.05) & (d < dmax)
    valid = float(ok.mean()) if ok.size else 0.0
    if not ok.any():
        return np.zeros(0), valid
    z = d[ok]
    f, l, up = mount.to_body(xn[ok] * z, yn[ok] * z, z)
    inside = (f > 0.0) & (np.abs(l) <= half_width_m) & (up >= min_height_m) & \
        (up <= max_height_m)
    return f[inside], valid


def corridor(depth, rect, mount, stop_m, half_width_m=0.35, min_height_m=0.10,
             max_height_m=1.9, step=4, min_points=30, dmax=10.0):
    """The nearest return inside the walking corridor (corridor_points). Returns
    (obstacle, nearest_forward_m or None, points counted). `obstacle` needs `min_points`
    inside `stop_m`, so one speckle of sensor noise is not a wall."""
    fwd, _ = corridor_points(depth, rect, mount, half_width_m, min_height_m, max_height_m,
                             step, dmax)
    if not fwd.size:
        return False, None, 0
    near = int((fwd < stop_m).sum())
    return near >= min_points, float(np.percentile(fwd, 5)), near


def kth_nearest(fwd, k):
    """The k-th smallest forward distance, or None with fewer than k points.

    V3's proximity distance. "Something is nearer than d" is then exactly "at least k
    corridor points are nearer than d", for every d -- so the bridge can compare it with
    a stop distance that changes with speed and still need k points, as `corridor` does
    at its one fixed distance. (A percentile of the whole corridor would not: a far wall
    full of points can hide a near object of 40.)"""
    k = max(1, int(k))
    if fwd.size < k:
        return None
    return float(np.partition(fwd, k - 1)[k - 1])


def valid_fraction(depth, region=None, step=4):
    """Fraction of pixels with a return, over the image or region (x0, y0, x1, y1) as
    fractions of its width and height. Stereo depth has a minimum range: an object
    pressed against the camera shows up as a hole, not as a near distance."""
    h, w = depth.shape[:2]
    if region is not None:
        x0, y0, x1, y1 = region
        depth = depth[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))]
    d = np.asarray(depth[::step, ::step], dtype=np.float64)
    if not d.size:
        return 0.0
    return float((np.isfinite(d) & (d > 0.05)).mean())
