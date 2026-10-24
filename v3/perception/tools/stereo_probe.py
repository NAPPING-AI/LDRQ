#!/usr/bin/env python3
"""Can this computer compute the stereo depth V3 needs, and how fast? (V3 B0)

    python3 perception/tools/stereo_probe.py [--seconds 10]

Reports the OpenCV version and whether it has StereoSGBM, the CPU count, and the time
per frame of SGBM on a synthetic 544x448 stereo pair (a random texture shifted by a known
disparity) for the configurations the plan considers: full and half resolution, 32 and 64
disparities, with and without the left-right check. SGBM's run time hardly depends on
what is in the picture, so a synthetic pair times it fairly; it also checks that the
known disparity comes back.

Run it with the stack in its usual state (policy running, LLM idle) so the numbers
include the contention the perception node will see. Python 3.8 + numpy + cv2.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np


def synthetic_pair(w, h, disparity, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, size=(h, w + disparity)).astype(np.uint8)
    # blur a little so it looks like texture, not per-pixel noise
    k = np.ones(3) / 3.0
    base = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, base).astype(np.uint8)
    # a point at column x in the left image is at x - disparity in the right image
    left = base[:, :w]
    right = base[:, disparity:disparity + w]
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=10.0, help="time budget for the whole bench")
    args = ap.parse_args()
    print("python {}".format(sys.version.split()[0]))
    print("cpus {}".format(os.cpu_count()))
    try:
        import cv2
    except ImportError:
        print("cv2: NOT importable -- our own stereo depth needs it")
        return 1
    print("cv2 {}; StereoSGBM {}; threads {}".format(
        cv2.__version__, "yes" if hasattr(cv2, "StereoSGBM_create") else "NO", cv2.getNumThreads()))
    if not hasattr(cv2, "StereoSGBM_create"):
        return 1
    configs = [  # (label, scale, numDisparities, left-right check)
        ("half 272x224, 32 disp, LR check", 2, 32, True),
        ("half 272x224, 32 disp", 2, 32, False),
        ("half 272x224, 64 disp, LR check", 2, 64, True),
        ("full 544x448, 64 disp, LR check", 1, 64, True),
    ]
    budget = args.seconds / len(configs)
    ok = True
    for label, scale, ndisp, lr in configs:
        w, h, true_d = 544 // scale, 448 // scale, 12
        left, right = synthetic_pair(w, h, true_d)
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=ndisp, blockSize=5,
                                     P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                     disp12MaxDiff=1 if lr else -1, speckleWindowSize=0,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        times = []
        t_end = time.monotonic() + budget
        while time.monotonic() < t_end or len(times) < 5:
            t0 = time.monotonic()
            disp = sgbm.compute(left, right)
            times.append(time.monotonic() - t0)
        d = disp[:, ndisp + 8:].astype(np.float32) / 16.0
        valid = d > 0
        med = float(np.median(d[valid])) if valid.any() else float("nan")
        times.sort()
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
        print("{:<34} median {:5.1f} ms  p95 {:5.1f} ms  ({} runs; disparity {:.2f} vs {} true, {:.0%} valid)".format(
            label, 1000 * statistics.median(times), 1000 * p95, len(times), med, true_d, valid.mean()))
        ok &= abs(med - true_d) < 0.5
    print("known disparity recovered: {}".format("yes" if ok else "NO"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
