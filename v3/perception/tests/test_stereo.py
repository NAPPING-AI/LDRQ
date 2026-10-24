#!/usr/bin/env python3
"""V3 own stereo depth (r1_perception/stereo.py) on a synthetic scene: a textured floor
1.2 m below a camera pitched 5 deg down, rendered for both rectified cameras with the
calibration's focal length and baseline. The depth must put the floor back where it is,
and fit_floor must find the height and the pitch -- the same tape-free check B2 runs on
the robot's images. Also: the head-shell mask, and the rig built from the calibration.

  python3 v3/perception/tests/test_stereo.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from r1_perception import stereo as ST  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok    " + name)
    else:
        FAIL += 1
        print("  FAIL  {}  <- {}".format(name, detail))


# The calibration's rectified numbers (2026-10-23): f 139.796 px, centre (272.859, 217.488),
# f x baseline 8.32717 px*m. Written here so the test needs no private file.
CAL = {"width": 544, "height": 448,
       "P_left": np.array([[139.79616, 0, 272.85889, 0], [0, 139.79616, 217.48784, 0], [0, 0, 1, 0]]),
       "P_right": np.array([[139.79616, 0, 272.85889, -8.32717], [0, 139.79616, 217.48784, 0],
                            [0, 0, 1, 0]])}


def render(rig, height, pitch_deg, cam_x, tex, cell=0.005, box=None):
    """Rectified grey image of a textured floor seen by a camera at x = cam_x (m).
    box: (distance ahead m, half width m, box height m) -- a textured box standing on the
    floor straight ahead, its front face a vertical plane."""
    th = math.radians(pitch_deg)
    n = np.array([0.0, math.cos(th), math.sin(th)])            # world down, camera frame
    e1, e2 = np.array([1.0, 0, 0]), np.array([0.0, -math.sin(th), math.cos(th)])
    v, u = np.mgrid[0:rig.h, 0:rig.w].astype(np.float64)
    r = np.stack([(u - rig.cx) / rig.f, (v - rig.cy) / rig.f, np.ones_like(u)], -1)
    nr = r @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = height / nr
    p = t[..., None] * r + np.array([cam_x, 0, 0])
    a, b = p @ e1, p @ e2
    gi = (a / cell + tex.shape[1] / 2)
    gj = (b / cell)
    ok = (nr > 0.05) & (gi >= 0) & (gi < tex.shape[1] - 1) & (gj >= 0) & (gj < tex.shape[0] - 1)
    img = np.full((rig.h, rig.w), 128.0)

    def sample(ok, gi, gj):
        i0, j0 = np.floor(gi[ok]).astype(int), np.floor(gj[ok]).astype(int)
        fi, fj = gi[ok] - i0, gj[ok] - j0
        return (tex[j0, i0] * (1 - fi) * (1 - fj) + tex[j0, i0 + 1] * fi * (1 - fj)
                + tex[j0 + 1, i0] * (1 - fi) * fj + tex[j0 + 1, i0 + 1] * fi * fj)

    img[ok] = sample(ok, gi, gj)
    if box is not None:
        dist, half_w, box_h = box
        fwd = e2                                     # horizontal forward, camera frame
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = (dist - np.array([cam_x, 0, 0]) @ fwd) / (r @ fwd)
        q = tb[..., None] * r + np.array([cam_x, 0, 0])
        up = height - q @ n                          # height above the floor
        side = q @ e1
        hit = (tb > 0) & (np.abs(side) < half_w) & (up > 0) & (up < box_h) & (~ok | (tb < t))
        bi = side / cell + tex.shape[1] / 2
        bj = (up + 1.0) / cell                       # another part of the texture
        good = hit & (bi >= 0) & (bi < tex.shape[1] - 1) & (bj >= 0) & (bj < tex.shape[0] - 1)
        img[good] = sample(good, bi, bj)
    return np.clip(img, 0, 255).astype(np.uint8)


rig = ST.StereoRig(CAL)
print("== the rig ==")
check("baseline from the calibration: 5.96 cm", abs(rig.baseline_m - 0.05957) < 2e-4, rig.baseline_m)
check("half resolution, 32 disparities: sees from 0.13 m", abs(rig.min_range_m - 0.130) < 0.002, rig.min_range_m)
z = rig.depth(np.array([[83.27, 8.327, np.nan, 0.0]]))
check("depth = f x B / disparity; no match stays unknown (NaN), never near",
      abs(z[0, 0] - 0.1) < 1e-4 and abs(z[0, 1] - 1.0) < 1e-4 and np.isnan(z[0, 2]) and np.isnan(z[0, 3]), z)

if ST.cv2 is None:
    print("  skip  matching: no cv2 here")
else:
    print("== a floor 1.2 m below, camera pitched 5 deg down ==")
    rng = np.random.RandomState(0)
    tex = rng.uniform(0, 255, (1200, 1200))
    tex = ST.cv2.GaussianBlur(tex, (0, 0), 1.0)
    L = render(rig, 1.2, 5.0, 0.0, tex)
    R = render(rig, 1.2, 5.0, rig.baseline_m, tex)
    disp = rig.disparity(L, R)
    zimg = rig.depth(disp)
    lower = zimg[int(zimg.shape[0] * 0.65):, :]
    check("the lower image (all floor) mostly matches (after the texture filter)",
          np.isfinite(lower).mean() > 0.7, np.isfinite(lower).mean())
    fl = ST.fit_floor(disp, rig)
    check("fit_floor: camera 1.2 m above the floor (+-3 %)", fl is not None and abs(fl["height_m"] - 1.2) < 0.036, fl)
    check("fit_floor: pitched 5 deg down (+-1 deg), no roll",
          fl is not None and abs(fl["pitch_down_deg"] - 5.0) < 1.0 and abs(fl["roll_deg"]) < 1.0, fl)
    check("...most floor pixels on the plane (within 0.5 px of disparity)",
          fl is not None and fl["inliers"] > 0.75, fl)
    v, u = np.mgrid[0:224, 0:272].astype(float) * 2.0
    th = math.radians(5.0)
    nr = ((v - rig.cy) / rig.f) * math.cos(th) + math.sin(th)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.where(nr > 0.05, rig.fB * nr / 1.2, np.nan)
    fe = ST.fit_floor(exact, rig)
    check("fit_floor on the exact disparity of that floor: 1.2000 m, 5.000 deg",
          abs(fe["height_m"] - 1.2) < 1e-6 and abs(fe["pitch_down_deg"] - 5.0) < 1e-4, fe)
    near = np.nanmin(zimg) if np.isfinite(zimg).any() else float("nan")
    check("no phantom: nothing nearer than where the floor enters the image (~0.6 m ahead) -- "
          "SGBM alone put a blob at 0.15 m next to the horizon", near > 0.5, near)
    raw = ST.StereoRig(CAL, texture_min=0, speckle_px=0, verify_m=0)
    zr = raw.depth(raw.disparity(L, R))
    check("...(and without the three filters the phantom is still there, so the "
          "test can see one)", int(np.nansum(zr < 0.5)) > 100, int(np.nansum(zr < 0.5)))
    print("== a box 0.5 m ahead on that floor ==")
    Lb = render(rig, 1.2, 5.0, 0.0, tex, box=(0.5, 0.2, 0.4))
    Rb = render(rig, 1.2, 5.0, rig.baseline_m, tex, box=(0.5, 0.2, 0.4))
    db = rig.disparity(Lb, Rb)
    flb = ST.fit_floor(db, rig)
    ptsb, _ = rig.points(rig.depth(db))
    nb, ncb = ST.corridor_nearest(ptsb, flb)
    check("the near-return check keeps a real box: the corridor's 30th-nearest point is the box "
          "(0.5 m +- 0.05)", nb is not None and abs(nb - 0.5) < 0.05, (nb, ncb))
    n0, nc0 = ST.corridor_nearest(rig.points(zimg)[0], fl)
    check("...and without the box the corridor is clear (nothing nearer than 1 m)",
          n0 is None or n0 > 1.0, (n0, nc0))
    print("== the head-shell mask ==")
    frames = []
    for k in range(20):
        f = rng.uniform(0, 255, (448, 544)).astype(np.uint8)
        f[380:, 430:] = 90                                     # the shell: never changes
        frames.append(f)
    m, frac = ST.shell_mask(frames, grow=4)
    check("the still corner is masked, the moving image is not",
          m[400:, 450:].all() and not m[:300, :].any() and frac < 0.06, frac)

print("--------------------------------------------------")
print("pass {}  fail {}".format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
