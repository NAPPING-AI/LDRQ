#!/usr/bin/env python3
"""Camera pitch (and focal length) from the floor, with the camera's height known -- no
tape measure and no cam_param.txt needed.

The walking corridor (depth.corridor_points) keeps points between corridor_min_height_m
(0.10 m) and corridor_max_height_m above the FLOOR, which it places with cam_height_m
and cam_pitch_deg (perception.yaml). A pitch a few degrees off puts the floor a few
centimetres "above the floor" far ahead, where it counts as an obstacle.

How: for a pinhole camera, the inverse depth of a plane is linear in the pixel:

    1/z = a*u + b*v + c

so the floor is found in depth frames WITHOUT knowing the intrinsics -- a robust fit
(RANSAC) of that relation over the lower part of the image, which ignores whatever stands
on the floor. With the up vector n in camera axes (x right, y down, z forward) and the
floor at n . P = -h:

    a = -n_x / (f h),   b = -n_y / (f h),   c = -n_z / h - a cx - b cy

so with h known (cam_height_m, or --height) and the principal point at the image centre,
n and f follow: the pitch, the roll, and the focal length (fx = fy assumed). Each frame
gives its own answer; their spread says how far to trust it.

    python3 perception/tools/fit_floor.py <capture dir> [--height 1.2] [--write]

--write puts cam_pitch_deg, cam_fx and cam_fy into the config (the rest untouched; the old
file kept as <config>.before_fit). It refuses when the frames disagree.
Numpy only.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1_perception import camera_model as C  # noqa: E402
from r1_perception import depth as D  # noqa: E402
from r1_perception import pipeline as PL  # noqa: E402

# what a plausible answer looks like: anything outside is a fit to something else
F_RANGE = (100.0, 1000.0)       # px, for a 544 px wide image: 140 deg .. 30 deg horizontal
PITCH_RANGE = (-20.0, 70.0)     # deg, positive = looking down
ROLL_MAX = 15.0                 # deg


def samples(depth, lower=0.6, step=4, zmin=0.3, zmax=6.0):
    """(u, v, z) of the image's lower part, every step-th pixel, valid depths only."""
    h, w = depth.shape
    v0 = int(h * (1.0 - lower))
    d = np.asarray(depth, dtype=np.float64)[v0::step, ::step]
    vv, uu = np.mgrid[v0:h:step, 0:w:step]
    vv, uu = vv[:d.shape[0], :d.shape[1]], uu[:d.shape[0], :d.shape[1]]
    ok = np.isfinite(d) & (d > zmin) & (d < zmax)
    return uu[ok].astype(np.float64), vv[ok].astype(np.float64), d[ok]


def solve(abc, h, cx, cy):
    """(a, b, c) of 1/z = a u + b v + c and the height -> (f, pitch_deg, roll_deg, n) or None."""
    a, b, c = abc
    nz = -h * (c + a * cx + b * cy)
    s = a * a + b * b
    if s <= 0 or abs(nz) >= 1.0:
        return None
    f = math.sqrt(1.0 - nz * nz) / (h * math.sqrt(s))
    nx, ny = -a * f * h, -b * f * h
    pitch = math.degrees(math.atan2(-nz, -ny))      # up = (0, -cos p, -sin p) for no roll
    roll = math.degrees(math.atan2(nx, -ny))        # its size only: |roll| > 5 deg is warned about
    return f, pitch, roll, np.array([nx, ny, nz])


def plausible(sol):
    if sol is None:
        return False
    f, pitch, roll, _ = sol
    return F_RANGE[0] <= f <= F_RANGE[1] and PITCH_RANGE[0] <= pitch <= PITCH_RANGE[1] \
        and abs(roll) <= ROLL_MAX


def fit_floor(u, v, z, h, cx, cy, iters=400, rel_tol=0.03, seed=0):
    """RANSAC on 1/z = a u + b v + c. Returns a dict, or None without a plausible floor."""
    if len(z) < 100:
        return None
    rng = np.random.RandomState(seed)
    A = np.stack([u, v, np.ones_like(u)], axis=1)
    w = 1.0 / z

    def inliers(abc):
        pred = A @ abc
        with np.errstate(divide="ignore", invalid="ignore"):
            zp = np.where(pred > 0, 1.0 / pred, np.inf)
        return np.abs(zp - z) < np.maximum(0.02, rel_tol * z)

    best, best_n = None, 0
    for _ in range(iters):
        idx = rng.choice(len(z), 3, replace=False)
        try:
            abc = np.linalg.solve(A[idx], w[idx])
        except np.linalg.LinAlgError:
            continue
        # the floor gets nearer lower down (b > 0), and must make sense as a camera
        if abc[1] <= 0 or not plausible(solve(abc, h, cx, cy)):
            continue
        m = inliers(abc)
        if m.sum() > best_n:
            best, best_n = abc, int(m.sum())
    if best is None:
        return None
    m = inliers(best)
    for _ in range(2):                              # refine on the consensus
        best = np.linalg.lstsq(A[m], w[m], rcond=None)[0]
        m = inliers(best)
    sol = solve(best, h, cx, cy)
    if not plausible(sol):
        return None
    f, pitch, roll, _ = sol
    pred = A[m] @ best
    return {"abc": best, "f": f, "pitch": pitch, "roll": roll, "inliers": float(m.mean()),
            "n_inliers": int(m.sum()),
            "resid_m": float(np.sqrt(np.mean((1.0 / pred - z[m]) ** 2))),
            "horizon_v": float(-(best[0] * cx + best[2]) / best[1])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frames", help="a capture directory with NNNNNN_depth.npy")
    ap.add_argument("--config", default=str(ROOT / "config" / "perception.yaml"))
    ap.add_argument("--height", type=float, default=None,
                    help="camera height above the floor, m (default: cam_height_m)")
    ap.add_argument("--lower", type=float, default=0.6, help="fraction of the image used (bottom)")
    ap.add_argument("--write", action="store_true",
                    help="write cam_pitch_deg, cam_fx, cam_fy into --config")
    args = ap.parse_args()
    cfg = PL.read_flat_yaml(args.config)
    h = float(args.height if args.height is not None else cfg.get("cam_height_m", 1.2))
    W, H = int(cfg["cam_width"]), int(cfg["cam_height"])
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.frames), "*_depth.npy")))
    if not files:
        print("no *_depth.npy in {}".format(args.frames))
        return 2
    print("camera height {:.3f} m (known); principal point at the image centre ({:.1f}, {:.1f})"
          .format(h, cx, cy))
    rows = []
    for fp in files:
        depth = np.load(fp).astype(np.float32)
        r = fit_floor(*samples(depth, args.lower), h=h, cx=cx, cy=cy)
        name = os.path.basename(fp)
        if r is None:
            print("{}: no floor found (too little floor in view, or nothing plausible)".format(name))
            continue
        rows.append((fp, r, depth))
        print("{}: pitch {:+.1f} deg down, roll {:+.1f}, f {:.0f} px ({:.0f} deg wide), floor "
              "{:.0%} of the lower image, fit error {:.3f} m, horizon at row {:.0f}".format(
                  name, r["pitch"], r["roll"], r["f"], 2 * math.degrees(math.atan(W / 2.0 / r["f"])),
                  r["inliers"], r["resid_m"], r["horizon_v"]))
    if not rows:
        print("\nno frame had a floor to fit: point the camera at a clear stretch of floor "
              "(1-3 m ahead) and capture again")
        return 1
    P = np.array([r["pitch"] for _, r, _ in rows])
    F = np.array([r["f"] for _, r, _ in rows])
    R = np.array([r["roll"] for _, r, _ in rows])
    p_med, f_med, r_med = float(np.median(P)), float(np.median(F)), float(np.median(R))
    p_spread, f_spread = float(P.max() - P.min()), float((F.max() - F.min()) / f_med)
    print("\nfitted (median of {} frames): pitch {:.1f} deg (spread {:.1f}), f {:.0f} px "
          "(spread {:.0%}), roll {:.1f} deg".format(len(rows), p_med, p_spread, f_med, f_spread, r_med))
    print("configured                   : pitch {:.1f} deg, fx {:.0f} px".format(
        float(cfg.get("cam_pitch_deg", 0.0)), float(cfg.get("cam_fx", 0.0))))
    good = len(rows) >= 3 and p_spread <= 3.0 and f_spread <= 0.10 and abs(r_med) <= 5.0
    if abs(r_med) > 5.0:
        print("roll {:.1f} deg: the corridor assumes none -- bring the frames back".format(r_med))

    def corridor(pitch, f, depth):
        rect = C.Rectified(W, H, fx=f, fy=f, cx=cx, cy=cy)
        fwd, _ = D.corridor_points(depth, rect, C.Mount(h, pitch, cfg.get("cam_yaw_deg", 0.0)),
                                   float(cfg.get("corridor_half_width_m", 0.35)),
                                   float(cfg.get("corridor_min_height_m", 0.10)),
                                   float(cfg.get("corridor_max_height_m", 1.9)),
                                   int(cfg.get("corridor_step", 4)))
        return fwd.size, D.kth_nearest(fwd, int(cfg.get("corridor_min_points", 30)))

    print("\nthe walking corridor in these frames (configured | fitted); on a clear floor the "
          "fitted one finds nothing, and whatever it does find is really there:")
    for fp, _, depth in rows:
        n0, k0 = corridor(float(cfg.get("cam_pitch_deg", 0.0)), float(cfg["cam_fx"]), depth)
        n1, k1 = corridor(p_med, f_med, depth)
        print("  {}: {} points, obstacle_m {} | {} points, obstacle_m {}".format(
            os.path.basename(fp), n0, "-" if k0 is None else "{:.2f}".format(k0),
            n1, "-" if k1 is None else "{:.2f}".format(k1)))
    if not args.write:
        print("\nto keep them: the same command with --write" if good else
              "\nnot good enough to write: need 3+ frames, pitch spread <= 3 deg, f spread <= 10%")
        return 0 if good else 1
    if not good:
        print("\nNOT written: the frames disagree (need 3+ frames, pitch spread <= 3 deg, "
              "f spread <= 10%, |roll| <= 5 deg)")
        return 1
    write(args.config, {"cam_pitch_deg": "{:.1f}".format(p_med), "cam_fx": "{:.1f}".format(f_med),
                        "cam_fy": "{:.1f}".format(f_med)}, h, args.frames)
    return 0


def write(path, values, h, frames):
    import re
    import shutil
    import time
    text = pathlib.Path(path).read_text()
    note = "  # fitted {} from the floor at height {:.3f} m ({})".format(time.strftime("%F"), h, frames)
    new = text
    for key, val in values.items():
        new, k = re.subn(r"(?m)^{}:[^\n]*$".format(key), "{}: {}{}".format(key, val, note), new)
        if k != 1:
            print("\nnot written: {} has no single {} line".format(path, key))
            return
    shutil.copy2(path, path + ".before_fit")
    pathlib.Path(path).write_text(new)
    print("\nwritten to {} (the old file: {}.before_fit):".format(path, path))
    for line in new.splitlines():
        if line.split(":")[0] in values:
            print("  " + line)


if __name__ == "__main__":
    sys.exit(main())
