#!/usr/bin/env python3
"""Bearing and depth error against AprilTag ground truth (stage E §3.6).

Put a tag36h11 (printed, black square measured; 0.25 m is ~22 px at 3 m in the 640 px
rectified view, and smaller tags are skipped, see --min-side-px) flat on the front of the
target object, facing the robot. For each capture (NNNNNN.jpg rectified + NNNNNN.json detections, from
`r1_perception_cli.py capture`), this finds the tag in the rectified image, solves its
pose with the rectified pinhole intrinsics from meta.json, and compares:

  bearing  of the tag's center (truth)  vs  bearing_deg of the detection whose box
           contains the tag (what turn_to turns by)
  range    horizontal distance to the tag's center (truth)  vs  range_m of that
           detection (median depth in the box: what walk_to stops on)

Both the tag and the box are on the same object, so this measures the perception chain
(intrinsics, rectification, mount pitch, box center vs object, depth median), not the
tag. Dev box: needs OpenCV with the aruco module (opencv-contrib or OpenCV >= 4.7).

  python3 perception/tools/eval_tag.py --captures ~/r1_captures/tag --tag-m 0.10 \\
      [--cls chair] [--out res.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from r1_perception import camera_model as C  # noqa: E402


def detect_tags(gray):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
    return [] if ids is None else [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]


def tag_pose(corners, size_m, f, cx, cy):
    import cv2
    h = size_m / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return tvec.ravel() if ok else None


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--captures", required=True)
    ap.add_argument("--tag-m", type=float, required=True, help="black square edge length")
    ap.add_argument("--cls", default=None, help="only detections of this class")
    ap.add_argument("--min-side-px", type=float, default=30.0,
                    help="smaller tags are skipped: their range is not truth (1 px ~ 1/side)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = os.path.expanduser(a.captures)
    meta = json.load(open(os.path.join(d, "meta.json")))
    r = meta["rect"]
    mt = meta["mount"]
    mount = C.Mount(mt["height_m"], mt["pitch_deg"], mt["yaw_deg"])
    rows = []
    for js in sorted(glob.glob(os.path.join(d, "[0-9]*.json"))):
        base = js[:-5]
        msg = json.load(open(js))
        img = cv2.imread(base + ".jpg", cv2.IMREAD_GRAYSCALE)
        tags = detect_tags(img) if img is not None else []
        if len(tags) != 1:
            print("{}: {} tags, skipped".format(os.path.basename(base), len(tags)))
            continue
        tid, corners = tags[0]
        side_px = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
        if side_px < a.min_side_px:
            # The tag's distance comes from its apparent size: one pixel of corner error
            # on a 12 px tag is 8 % of the range. Such a tag is not ground truth.
            print("{}: tag only {:.0f} px wide (< {:.0f}): skipped -- use a bigger tag or "
                  "come closer".format(os.path.basename(base), side_px, a.min_side_px))
            continue
        t = tag_pose(corners, a.tag_m, r["f"], r["cx"], r["cy"])
        if t is None:
            continue
        xn, yn = t[0] / t[2], t[1] / t[2]
        true_b, _ = mount.bearing_elev_deg(xn, yn)
        true_r = float(mount.range_m(xn, yn, t[2]))
        u, v = corners.mean(0)
        dets = [x for x in msg["dets"] if (a.cls is None or x["cls"] == a.cls)
                and x["bbox"][0] <= u <= x["bbox"][2] and x["bbox"][1] <= v <= x["bbox"][3]]
        if not dets:
            print("{}: tag at {:+.1f} deg {:.2f} m, no detection around it".format(
                os.path.basename(base), float(true_b), true_r))
            rows.append({"frame": os.path.basename(base), "true_bearing": float(true_b),
                         "true_range": true_r, "detected": False})
            continue
        det = max(dets, key=lambda x: x["conf"])
        row = {"frame": os.path.basename(base), "cls": det["cls"], "conf": det["conf"],
               "true_bearing": float(true_b), "bearing": det["bearing_deg"],
               "bearing_err": det["bearing_deg"] - float(true_b),
               "true_range": true_r, "range": det.get("range_m"),
               "range_err": None if det.get("range_m") is None else det["range_m"] - true_r,
               "detected": True}
        rows.append(row)
        print("{}: {} {:.2f} | bearing {:+.1f} vs tag {:+.1f} (err {:+.2f}) | range {} vs tag {:.2f}{}".format(
            row["frame"], row["cls"], row["conf"], row["bearing"], row["true_bearing"],
            row["bearing_err"], "-" if row["range"] is None else "{:.2f}".format(row["range"]),
            row["true_range"], "" if row["range_err"] is None else " (err {:+.3f})".format(row["range_err"])))
    got = [x for x in rows if x["detected"]]
    if not got:
        sys.exit("no frame with exactly one tag inside a detection")

    def stats(key):
        xs = [x[key] for x in got if x.get(key) is not None]
        if not xs:
            return None
        ab = sorted(abs(v) for v in xs)
        return {"n": len(xs), "mean": statistics.mean(xs), "mean_abs": statistics.mean(ab),
                "p95_abs": ab[min(len(ab) - 1, int(0.95 * (len(ab) - 1)))], "max_abs": ab[-1]}
    res = {"frames": len(rows), "detected": len(got), "bearing_err_deg": stats("bearing_err"),
           "range_err_m": stats("range_err"), "rows": rows}
    for k, unit in (("bearing_err_deg", "deg"), ("range_err_m", "m")):
        s = res[k]
        if s:
            print("{}: n {} | mean {:+.3f} {} (bias) | mean |err| {:.3f} | p95 {:.3f} | max {:.3f}".format(
                k, s["n"], s["mean"], unit, s["mean_abs"], s["p95_abs"], s["max_abs"]))
    print("detected around the tag in {} of {} frames".format(len(got), len(rows)))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
