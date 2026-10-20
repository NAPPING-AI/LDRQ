#!/usr/bin/env python3
"""Fisheye intrinsics from checkerboard pictures, written as perception.yaml lines.

Dev box (needs OpenCV). On the robot, capture 20-30 views of a printed checkerboard
with the camera's OWN images (capture writes them as NNNNNN_raw.jpg): near and far,
tilted, and with the board in every corner of the picture -- a 150 degree lens is
mostly corner, and corners the calibration never saw are corners it does not model.

  python3 perception/tools/calibrate_fisheye.py --images ~/r1_captures/calib/*_raw.jpg \\
      --cols 9 --rows 6 --square-m 0.025 [--check-out undistorted.jpg]

Prints the cam_* lines to paste into perception/config/perception.yaml, the RMS
reprojection error (under ~0.5 px is a good calibration), and how many views were used.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys

import numpy as np


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column")
    ap.add_argument("--square-m", type=float, required=True)
    ap.add_argument("--check-out", default=None, help="write one undistorted view here")
    ap.add_argument("--rect-hfov-deg", type=float, default=100.0)
    ap.add_argument("--hfov-guess-deg", type=float, default=150.0,
                    help="the lens's nominal horizontal field of view, for the starting guess")
    a = ap.parse_args(argv)

    paths = []
    for p in a.images:
        paths += sorted(glob.glob(p)) or [p]
    obj = np.zeros((1, a.cols * a.rows, 3), np.float64)
    obj[0, :, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square_m
    objpts, imgpts, size, used = [], [], None, []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("skip (unreadable): " + p)
            continue
        if size is None:
            size = img.shape[::-1]
        elif img.shape[::-1] != size:
            print("skip (size {} != {}): {}".format(img.shape[::-1], size, p))
            continue
        ok, corners = cv2.findChessboardCorners(
            img, (a.cols, a.rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if not ok:
            print("skip (no board found): " + p)
            continue
        corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), crit)
        objpts.append(obj.copy())
        imgpts.append(corners.reshape(1, -1, 2).astype(np.float64))
        used.append(p)
    if len(used) < 8:
        sys.exit("only {} usable views; take more (20-30), covering the corners".format(len(used)))
    # Start from the lens's nominal field of view (equidistant: f = half width / half
    # angle). Without a guess OpenCV starts from f = width / pi, which for a wide lens can
    # leave its homography initialization degenerate (InitExtrinsics: norm_u1 > 0).
    f0 = (size[0] / 2.0) / math.radians(a.hfov_guess_deg / 2.0)

    def guess():
        return (np.array([[f0, 0, (size[0] - 1) / 2.0], [0, f0, (size[1] - 1) / 2.0], [0, 0, 1]]),
                np.zeros((4, 1)))
    K, D = guess()

    def flag(name):   # cv2.fisheye.CALIB_* in OpenCV 4 (the robot), cv2.CALIB_* in 5
        return getattr(cv2.fisheye, name, None) or getattr(cv2, name)
    flags = (flag("CALIB_RECOMPUTE_EXTRINSIC") + flag("CALIB_CHECK_COND") +
             flag("CALIB_FIX_SKEW") + flag("CALIB_USE_INTRINSIC_GUESS"))
    # CALIB_CHECK_COND stops on a view whose extrinsics are ill-conditioned (a board seen
    # nearly edge-on) and names it; drop that view and try again, and say so.
    import re
    dropped = []
    while True:
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(objpts, imgpts, size, K, D, flags=flags,
                                                    criteria=crit)
            break
        except cv2.error as exc:
            m = re.search(r"input array (\d+)", str(exc))
            if not m or len(used) - 1 < 8:
                raise
            i = int(m.group(1))
            dropped.append(used.pop(i))
            objpts.pop(i)
            imgpts.pop(i)
            K, D = guess()
    for p in dropped:
        print("dropped (ill-conditioned, e.g. seen edge-on): " + p)
    print("\n{} of {} views used, image {}x{}, RMS reprojection error {:.3f} px{}".format(
        len(used), len(paths), size[0], size[1], rms,
        "" if rms < 0.5 else "  <- high: check the board, the focus, the coverage"))
    print("\n# perception.yaml, from tools/calibrate_fisheye.py ({} views, RMS {:.3f} px)".format(
        len(used), rms))
    for k, v in (("cam_width", size[0]), ("cam_height", size[1])):
        print("{}: {}".format(k, v))
    for k, v in (("cam_fx", K[0, 0]), ("cam_fy", K[1, 1]), ("cam_cx", K[0, 2]), ("cam_cy", K[1, 2])):
        print("{}: {:.4f}".format(k, v))
    for i in range(4):
        print("cam_k{}: {:.8f}".format(i + 1, float(D[i, 0])))
    if a.check_out:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
        from r1_perception import camera_model as C
        fish = C.Fisheye(K[0, 0], K[1, 1], K[0, 2], K[1, 2], D.ravel(), size)
        rect = C.Rectified(640, 480, a.rect_hfov_deg)
        mx, my = rect.maps(fish)
        out = C.remap(cv2.imread(used[0]), mx, my)
        cv2.imwrite(a.check_out, out)
        print("\nwrote {}: straight lines in the room must look straight".format(a.check_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
