"""Depth from the head's own left and right images (V3 plan 01 §1.8, B1-B3).

The stereo module's own depth (/dev/video-dep) cannot be used close up: it fills what it
cannot match (the head shell hides part of the right camera) with near values, and it is
not aligned with the left image. This computes depth the plain way instead, with the
calibration the user supplied on 2026-10-23 (OpenCV, 544x448):

  StereoRig.rectify     undistort and rectify both images (maps built once)
  StereoRig.disparity   SGBM at half resolution, left-right consistency, NO filling:
                        a pixel that does not match is NaN ("unknown"), never "near".
                        SGBM also "matches" where there is nothing to match -- a blank
                        wall, the sky over a horizon, along a horizontal edge -- by
                        smoothing across from its neighbours, and there it can be wildly
                        wrong: a synthetic floor scene gave a 600-pixel blob at 0.15 m
                        next to its horizon (v3/perception/tests/test_stereo.py). So a
                        disparity is kept only where the left image has horizontal texture
                        (|Sobel x| >= texture_min), and small islands left after that are
                        dropped (speckle filter). On the same scene through H264 with
                        sensor noise, a smaller blob at 0.15 m survived that, so every
                        NEAR return (nearer than verify_m: what the reflex acts on) is
                        checked again at full resolution -- an 11 x 11 patch must have
                        contrast and correlate (ZNCC) with the right image at that
                        disparity -- and is dropped if not. A real near surface passes;
                        a match smoothed across a blank region does not
  StereoRig.depth       z = f * B / disparity, in the rectified left camera
  StereoRig.points      the 3-D point of every valid pixel (x right, y down, z forward)
  shell_mask            the pixels that never change while the hanging robot turns: its
                        own head shell, left out before matching
  fit_floor             the floor plane, fitted in disparity (where it is linear and the
                        noise is even): the camera's height above it (a tape-free check
                        against the measured 1.2 m) and its pitch

numpy + cv2 (Python 3.8): what the robot has.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except ImportError:          # the tests that need no matching still import this module
    cv2 = None


class StereoRig(object):
    def __init__(self, calib, scale=0.5, num_disparities=32, block=5, texture_min=8.0,
                 speckle_px=60, speckle_range=2.0, verify_m=1.0, verify_half=5,
                 verify_std=4.0, verify_zncc=0.7):
        """`calib`: the calibration npz path (or a dict with its arrays). `scale`: the
        matcher's resolution relative to 544x448; `num_disparities` is at that scale."""
        d = np.load(calib) if isinstance(calib, str) else calib
        self.w, self.h = int(d["width"]), int(d["height"])
        self.P_left, self.P_right = np.asarray(d["P_left"], float), np.asarray(d["P_right"], float)
        self.f = float(self.P_left[0, 0])
        self.cx, self.cy = float(self.P_left[0, 2]), float(self.P_left[1, 2])
        self.fB = -float(self.P_right[0, 3])           # focal length (px) x baseline (m)
        self.baseline_m = self.fB / self.f
        self.scale = float(scale)
        self.num_disparities = int(num_disparities)
        self.block = int(block)
        self.texture_min = float(texture_min)      # |Sobel x| on the matcher's image
        self.speckle_px = int(speckle_px)          # islands smaller than this are dropped
        self.speckle_range = float(speckle_range)  # (disparity steps within an island, px)
        self.verify_m = float(verify_m)            # returns nearer than this are re-checked
        self.verify_half = int(verify_half)        # patch half size (full res)
        self.verify_std = float(verify_std)        # the left patch's contrast, grey levels
        self.verify_zncc = float(verify_zncc)      # correlation with the right patch
        self._d = d
        self._maps = None
        self._sgbm = None

    # ------------------------------------------------------------ geometry
    @property
    def min_range_m(self):
        """Nearest distance the matcher can see: the largest disparity it searches."""
        return self.fB / (self.num_disparities / self.scale)

    def depth(self, disp):
        """Disparity in FULL-resolution pixels -> metres; NaN stays NaN."""
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.fB / disp
        z[~np.isfinite(z) | (z <= 0)] = np.nan
        return z

    def points(self, depth, step=1):
        """(N, 3) points, rectified left camera frame (x right, y down, z forward), and
        their (v, u) pixel positions in the depth image (at its own resolution)."""
        sh, sw = depth.shape
        k = sw / float(self.w)                        # depth image px per full-res px
        v, u = np.mgrid[0:sh:step, 0:sw:step]
        z = depth[v, u]
        ok = np.isfinite(z)
        z, u, v = z[ok], u[ok], v[ok]
        x = (u / k - self.cx) * z / self.f
        y = (v / k - self.cy) * z / self.f
        return np.stack([x, y, z], 1), np.stack([v, u], 1)

    # ------------------------------------------------------------ images
    def _build_maps(self):
        d = self._d
        size = (self.w, self.h)
        ml = cv2.initUndistortRectifyMap(d["K_left"], d["D_left"], d["R_left"], d["P_left"], size,
                                         cv2.CV_16SC2)
        mr = cv2.initUndistortRectifyMap(d["K_right"], d["D_right"], d["R_right"], d["P_right"], size,
                                         cv2.CV_16SC2)
        self._maps = (ml, mr)

    def rectify(self, left, right):
        """Raw grey (or BGR) images -> rectified grey images, full resolution."""
        if self._maps is None:
            self._build_maps()
        (l1, l2), (r1, r2) = self._maps
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return (cv2.remap(left, l1, l2, cv2.INTER_LINEAR),
                cv2.remap(right, r1, r2, cv2.INTER_LINEAR))

    def disparity(self, left_rect, right_rect, mask=None, mask_right=None):
        """Rectified grey images -> disparity in full-resolution pixels at the matcher's
        resolution (NaN where it does not match). `mask` / `mask_right` (full-res, True =
        ignore: the head shell in each rectified image) are blanked before matching -- a
        blank region has no texture, so nothing in the left image can match into the
        right image's shell -- and the left mask is NaN in the result."""
        if mask is not None or mask_right is not None:
            left_rect, right_rect = left_rect.copy(), right_rect.copy()
            if mask is not None:
                left_rect[mask] = 0
            if mask_right is not None:
                right_rect[mask_right] = 0
        if self._sgbm is None:
            b = self.block
            self._sgbm = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.num_disparities, blockSize=b,
                P1=8 * b * b, P2=32 * b * b, disp12MaxDiff=1, uniquenessRatio=10,
                speckleWindowSize=50, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        if self.scale != 1.0:
            size = (int(round(self.w * self.scale)), int(round(self.h * self.scale)))
            l = cv2.resize(left_rect, size, interpolation=cv2.INTER_AREA)
            r = cv2.resize(right_rect, size, interpolation=cv2.INTER_AREA)
        else:
            l, r = left_rect, right_rect
        raw16 = self._sgbm.compute(l, r)
        raw16[:, :self.num_disparities] = -16         # the strip the search cannot cover
        if self.texture_min > 0:
            gx = np.abs(cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3))
            raw16[gx < self.texture_min] = -16
        if self.speckle_px > 0:
            cv2.filterSpeckles(raw16, -16, self.speckle_px, int(round(16 * self.speckle_range)))
        raw = raw16.astype(np.float32) / 16.0
        disp = raw / self.scale                       # in full-resolution pixels
        disp[raw <= 0] = np.nan
        if mask is not None:
            m = mask if mask.shape == disp.shape else cv2.resize(
                mask.astype(np.uint8), (disp.shape[1], disp.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(bool)
            disp[m] = np.nan
        if self.verify_m > 0:
            self._verify_near(disp, left_rect, right_rect)
        return disp

    def _verify_near(self, disp, left_rect, right_rect):
        """Drop the near returns (in place) whose full-resolution patches do not match.
        Vectorized: every candidate's two patches are gathered at once."""
        k = disp.shape[1] / float(self.w)
        with np.errstate(invalid="ignore"):
            near = np.argwhere(disp > self.fB / self.verify_m)
        if len(near) == 0:
            return
        hs = self.verify_half
        v, u = near[:, 0], near[:, 1]
        d = disp[v, u]
        x = np.rint(u / k).astype(int)
        y = np.rint(v / k).astype(int)
        xr = np.rint(u / k - d).astype(int)
        inside = (y - hs >= 0) & (y + hs < self.h) & (x - hs >= 0) & (x + hs < self.w) \
            & (xr - hs >= 0) & (xr + hs < self.w)
        disp[v[~inside], u[~inside]] = np.nan
        v, u, x, y, xr = v[inside], u[inside], x[inside], y[inside], xr[inside]
        if len(v) == 0:
            return
        off = np.arange(-hs, hs + 1)
        rows = (y[:, None] + off[None, :])[:, :, None]          # (N, P, 1)
        L = left_rect.astype(np.float32)
        R = right_rect.astype(np.float32)
        a = L[rows, (x[:, None] + off[None, :])[:, None, :]]      # (N, P, P)
        b = R[rows, (xr[:, None] + off[None, :])[:, None, :]]
        a = a.reshape(len(v), -1)
        b = b.reshape(len(v), -1)
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        sa = np.sqrt((a * a).mean(1))
        sb = np.sqrt((b * b).mean(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            zncc = (a * b).mean(1) / (sa * sb)
        bad = (sa < self.verify_std) | (sb < 1e-3) | ~(zncc >= self.verify_zncc)
        disp[v[bad], u[bad]] = np.nan

def corridor_nearest(points, floor, half_width=0.35, min_h=0.10, max_h=1.5, max_range=3.0, k=30):
    """The k-th nearest forward distance among points in the walking corridor: within
    `half_width` of the camera's forward line, between `min_h` and `max_h` above the floor
    (so the floor itself is not an obstacle: 10 cm, because a degree of pitch error and the
    matching noise put floor points 5-8 cm "above" the fitted plane 1 m ahead), nearer
    than `max_range`. Points are camera
    frame (StereoRig.points); `floor` is fit_floor's dict. Returns (distance or None, the
    number of corridor points). The k-th, not the nearest: a few stray pixels are not an
    obstacle (perception's depth.kth_nearest does the same)."""
    p = np.asarray(points, dtype=np.float64)
    if floor is None or len(p) == 0:
        return None, 0
    th = math.radians(floor["pitch_down_deg"])
    ro = math.radians(floor["roll_deg"])
    down = np.array([math.sin(ro) * math.cos(th), math.cos(ro) * math.cos(th), math.sin(th)])
    down /= np.linalg.norm(down)
    height = floor["height_m"] - p @ down             # above the floor
    fwd = np.array([0.0, -math.sin(th), math.cos(th)])
    ahead = p @ fwd
    side = p[:, 0]
    sel = (np.abs(side) < half_width) & (height > min_h) & (height < max_h) & (ahead > 0) \
        & (ahead < max_range)
    n = int(sel.sum())
    if n < k:
        return None, n
    return float(np.sort(ahead[sel])[k - 1]), n


def shell_mask(frames, std_max=2.0, grow=6, lower_only=0.5):
    """Pixels that do not change over `frames` (N, H, W grey, taken while the robot turns):
    the robot's own body in view. Only the lower `lower_only` of the image is considered
    (the shell is at the bottom corners; a featureless ceiling must not be called shell).
    Grown by `grow` px. Returns (mask bool H x W, fraction of the image)."""
    f = np.asarray(frames, dtype=np.float32)
    still = f.std(axis=0) < std_max
    h = still.shape[0]
    still[:int(h * (1.0 - lower_only)), :] = False
    m = still.astype(np.uint8)
    if cv2 is not None and grow > 0:
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    m = m.astype(bool)
    return m, float(m.mean())


def fit_floor(disp, rig, tol_px=0.5, iters=400, seed=0, below_px=5):
    """The floor plane from a disparity image (full-resolution pixels, at any scale), by
    RANSAC then least squares IN DISPARITY: on a plane, disparity is exactly linear in the
    pixel, d = a (u - cx) + b (v - cy) + c, and stereo noise is about constant in
    disparity (not in metres, where it grows as z^2), so this is the fit the noise calls
    for. From (a, b, c / f) = (B / h) n: the camera's height h above the plane and the
    plane's normal n. Only pixels `below_px` under the principal point are used (the
    floor is below the horizon). Returns a dict: height_m, pitch_down_deg, roll_deg,
    inliers (fraction within tol_px), n -- or None."""
    sh, sw = disp.shape
    k = sw / float(rig.w)
    v, u = np.mgrid[0:sh, 0:sw]
    ok = np.isfinite(disp) & (v / k > rig.cy + below_px)
    if ok.sum() < 50:
        return None
    U, V, D = u[ok] / k - rig.cx, v[ok] / k - rig.cy, disp[ok].astype(np.float64)
    A = np.stack([U, V, np.ones_like(U)], 1)
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(D), 3, replace=False)
        try:
            x = np.linalg.solve(A[idx], D[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(A @ x - D) < tol_px
        if best is None or inl.sum() > best[1].sum():
            best = (x, inl)
    if best is None:
        return None
    x, inl = best
    a, b, c = np.linalg.lstsq(A[inl], D[inl], rcond=None)[0]
    vec = np.array([a, b, c / rig.f])
    if vec[1] <= 0:                                  # not below the camera: no floor
        return None
    h = rig.baseline_m / np.linalg.norm(vec)
    n = vec / np.linalg.norm(vec)                    # the downward normal, camera frame
    # camera frame: y down, z forward. Level, the downward normal is +y; pitched down by
    # theta, the forward axis points theta below the horizon and the normal gains +sin(theta) in z.
    return {"height_m": float(h), "pitch_down_deg": math.degrees(math.atan2(n[2], n[1])),
            "roll_deg": math.degrees(math.atan2(n[0], n[1])), "inliers": float(inl.mean()),
            "n": int(len(D))}
