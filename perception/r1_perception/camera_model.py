"""The head camera as geometry: a fisheye in, a pinhole out, angles to the body.

Lens model: OpenCV's fisheye (Kannala-Brandt, four coefficients), the model
cv2.fisheye.calibrate fits and the one tools/calibrate_fisheye.py writes. A 150 degree
lens cannot be rectified to one pinhole image without stretching its edges without
bound, so the rectified view covers a chosen horizontal field (`rect_hfov_deg`, ~100)
and the outer band of the lens is given up. YOLOX runs on the rectified image, so the
boxes are in a pinhole frame where an angle is one atan.

Frames. Camera: x right, y down, z along the optical axis (OpenCV). Body: x forward,
y left, z up, origin on the floor below the camera. The camera is mounted
`height_m` above the floor and pitched down by `pitch_deg` (positive = looking down),
with a yaw offset `yaw_deg` (positive = looking left) for a head that is not straight.
**Bearing is positive to the LEFT**, the same sign as the IMU yaw the closed-loop turn
uses, so "turn by the bearing" needs no sign flip anywhere.

numpy only; Python 3.8 clean.
"""
from __future__ import annotations

import math

import numpy as np


class Fisheye(object):
    """K (fx, fy, cx, cy) and D (k1..k4) for an image of size (w, h)."""

    def __init__(self, fx, fy, cx, cy, k=(0.0, 0.0, 0.0, 0.0), size=(1280, 720)):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.k = tuple(float(v) for v in k)
        self.size = (int(size[0]), int(size[1]))

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg["cam_fx"], cfg["cam_fy"], cfg["cam_cx"], cfg["cam_cy"],
                   (cfg.get("cam_k1", 0.0), cfg.get("cam_k2", 0.0), cfg.get("cam_k3", 0.0),
                    cfg.get("cam_k4", 0.0)),
                   (cfg["cam_width"], cfg["cam_height"]))

    def _distort_theta(self, th):
        k1, k2, k3, k4 = self.k
        t2 = th * th
        return th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def project(self, x, y):
        """Normalized pinhole coordinates (X/Z, Y/Z) -> fisheye pixels."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        r = np.hypot(x, y)
        th = np.arctan(r)
        thd = self._distort_theta(th)
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        return self.fx * x * s + self.cx, self.fy * y * s + self.cy

    def unproject(self, u, v, iters=20):
        """Fisheye pixels -> normalized pinhole coordinates (Newton on theta)."""
        xd = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        yd = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        thd = np.hypot(xd, yd)
        th = np.array(thd, copy=True)
        k1, k2, k3, k4 = self.k
        for _ in range(iters):
            t2 = th * th
            f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - thd
            df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            th = th - f / df
        th = np.clip(th, 0.0, math.pi / 2 - 1e-6)
        s = np.where(thd > 1e-12, np.tan(th) / np.maximum(thd, 1e-12), 1.0)
        return xd * s, yd * s


class Rectified(object):
    """The pinhole view YOLOX sees: size (w, h), horizontal field `hfov_deg`, square
    pixels, principal point at the center -- or, for a source that is already a
    pinhole image (the stereo module's rectified left camera), its own fx, fy, cx, cy."""

    def __init__(self, w, h, hfov_deg=None, fx=None, fy=None, cx=None, cy=None):
        self.w, self.h = int(w), int(h)
        if fx is None:
            self.hfov_deg = float(hfov_deg)
            self.f = (self.w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
            self.fy = self.f
            self.cx, self.cy = (self.w - 1) / 2.0, (self.h - 1) / 2.0
        else:
            self.f, self.fy = float(fx), float(fy if fy is not None else fx)
            self.cx = float(cx) if cx is not None else (self.w - 1) / 2.0
            self.cy = float(cy) if cy is not None else (self.h - 1) / 2.0
            self.hfov_deg = math.degrees(math.atan(self.cx / self.f)
                                         + math.atan((self.w - 1 - self.cx) / self.f))

    @property
    def vfov_deg(self):
        return math.degrees(2 * math.atan((self.h / 2.0) / self.fy))

    def maps(self, fisheye):
        """(map_x, map_y) float32 for cv2.remap: rectified pixel -> fisheye pixel."""
        u, v = np.meshgrid(np.arange(self.w, dtype=np.float64),
                           np.arange(self.h, dtype=np.float64))
        mx, my = fisheye.project((u - self.cx) / self.f, (v - self.cy) / self.fy)
        return mx.astype(np.float32), my.astype(np.float32)

    def ray(self, u, v):
        """Rectified pixel -> unnormalized ray in the camera frame (x, y, 1)."""
        return ((np.asarray(u, dtype=np.float64) - self.cx) / self.f,
                (np.asarray(v, dtype=np.float64) - self.cy) / self.fy)


def remap(img, map_x, map_y, nearest=False):
    """cv2.remap when available; a numpy nearest-neighbour fallback otherwise (depth is
    always nearest: interpolating across an object edge invents a surface)."""
    try:
        import cv2
        return cv2.remap(img, map_x, map_y,
                         cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    except ImportError:
        h, w = img.shape[:2]
        xi = np.rint(map_x).astype(np.int64)
        yi = np.rint(map_y).astype(np.int64)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out = np.zeros(map_x.shape + img.shape[2:], dtype=img.dtype)
        out[ok] = img[yi[ok], xi[ok]]
        return out


class Mount(object):
    """Where the camera sits on the body: height above the floor, pitch down, yaw left."""

    def __init__(self, height_m=1.2, pitch_deg=0.0, yaw_deg=0.0):
        self.height_m = float(height_m)
        self.pitch = math.radians(float(pitch_deg))
        self.yaw = math.radians(float(yaw_deg))

    def to_body(self, xc, yc, zc):
        """Camera-frame point(s) -> body frame (forward, left, up), origin on the floor
        below the camera."""
        xc, yc, zc = (np.asarray(a, dtype=np.float64) for a in (xc, yc, zc))
        # camera (right, down, forward) -> level frame (forward, left, up), then pitch
        # down about the left axis, then yaw about up.
        f0, l0, u0 = zc, -xc, -yc
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        f1 = f0 * cp + u0 * sp
        u1 = -f0 * sp + u0 * cp
        cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        f2 = f1 * cy_ - l0 * sy_
        l2 = f1 * sy_ + l0 * cy_
        return f2, l2, u1 + self.height_m

    def bearing_elev_deg(self, xn, yn):
        """Normalized ray (x/z, y/z) -> (bearing, elevation) in degrees; bearing left +."""
        f, l, u = self.to_body(xn, yn, 1.0)
        u = u - self.height_m
        return (np.degrees(np.arctan2(l, f)),
                np.degrees(np.arctan2(u, np.hypot(f, l))))

    def range_m(self, xn, yn, depth_z):
        """Horizontal distance from the camera to the point at optical depth `depth_z`
        along the normalized ray (x/z, y/z)."""
        f, l, _ = self.to_body(np.asarray(xn) * depth_z, np.asarray(yn) * depth_z, depth_z)
        return np.hypot(f, l)
