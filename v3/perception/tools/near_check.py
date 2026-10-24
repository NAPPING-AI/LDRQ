#!/usr/bin/env python3
"""What is close to the camera, and is it always there? For V3's reflex: something within
a few tenths of a metre in every frame -- the robot's own arm or hand in view, a stand or
gantry in front of it, or depth the stereo module filled in -- makes the walking
corridor report an obstacle in an empty room.

    python3 perception/tools/near_check.py <capture dir> [--near 0.35] [--persist 0.8]

For each frame: the share of the image nearer than --near, and where it is (rows and
columns as fractions of the image: 0,0 is top left). Then the pixels near in at least
--persist of the frames, which do not move with the scene. With cv2 it also writes
<dir>/near_NNNNNN.jpg (the frame, near pixels red) and <dir>/near_persistent.jpg: look at
them to see what it is. Numpy (+ cv2 for the pictures).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def bbox(mask):
    """(row0, row1, col0, col1) as fractions of the image, or None for an empty mask."""
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    h, w = mask.shape
    return rows.min() / h, (rows.max() + 1) / h, cols.min() / w, (cols.max() + 1) / w


def where(b):
    if b is None:
        return "-"
    r0, r1, c0, c1 = b
    v = "top" if r1 <= 0.4 else "bottom" if r0 >= 0.6 else "middle" if r0 >= 0.3 and r1 <= 0.7 else "tall"
    hpos = "left" if c1 <= 0.4 else "right" if c0 >= 0.6 else "centre" if c0 >= 0.3 and c1 <= 0.7 else "wide"
    return "{} {} (rows {:.2f}-{:.2f}, cols {:.2f}-{:.2f})".format(v, hpos, r0, r1, c0, c1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frames")
    ap.add_argument("--near", type=float, default=0.35, help="m")
    ap.add_argument("--persist", type=float, default=0.8, help="share of frames")
    args = ap.parse_args()
    d = os.path.expanduser(args.frames)
    files = sorted(glob.glob(os.path.join(d, "*_depth.npy")))
    if not files:
        print("no *_depth.npy in {}".format(d))
        return 2
    try:
        import cv2
    except ImportError:
        cv2 = None
    count = None
    for fp in files:
        z = np.load(fp).astype(np.float32)
        near = np.isfinite(z) & (z > 0.0) & (z < args.near)
        valid = np.isfinite(z) & (z > 0.0)
        count = near.astype(np.int32) if count is None else count + near
        name = os.path.basename(fp)[:-len("_depth.npy")]
        print("{}: {:.1%} of the image nearer than {:.2f} m (nearest {:.2f} m), {:.0%} valid, at {}".format(
            name, near.mean(), args.near, float(z[valid].min()) if valid.any() else float("nan"),
            valid.mean(), where(bbox(near))))
        jpg = os.path.join(d, name + ".jpg")
        if cv2 is not None and os.path.exists(jpg):
            img = cv2.imread(jpg)
            if img is not None and img.shape[:2] == near.shape:
                img[near] = (0.4 * img[near] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
                cv2.imwrite(os.path.join(d, "near_" + name + ".jpg"), img)
    pers = count >= max(1, int(np.ceil(args.persist * len(files))))
    print("\nnear in at least {:.0%} of the {} frames: {:.1%} of the image, at {}".format(
        args.persist, len(files), pers.mean(), where(bbox(pers))))
    if pers.mean() > 0.002:
        print("  it does not move with the scene: the robot's own body, a stand in front of it, or "
              "depth the module fills in. Look at near_persistent.jpg.")
    else:
        print("  nothing stays near: what was near moved with the scene (a person, a box)")
    if cv2 is not None:
        base = os.path.join(d, os.path.basename(files[0])[:-len("_depth.npy")] + ".jpg")
        img = cv2.imread(base) if os.path.exists(base) else None
        if img is None or img.shape[:2] != pers.shape:
            img = np.zeros(pers.shape + (3,), np.uint8)
        img[pers] = (0.4 * img[pers] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(d, "near_persistent.jpg"), img)
        print("  pictures: {}/near_*.jpg".format(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
