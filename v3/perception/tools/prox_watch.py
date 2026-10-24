#!/usr/bin/env python3
"""Watch /r1_perception/proximity (V3) for a few seconds and summarize it: what the
bridge's reflex would act on, measured against a tape.

    python3 perception/tools/prox_watch.py --seconds 5 --label person_0.6m --csv ~/o/v3_tape.csv

Prints a line every 0.5 s while it runs, then the median, min and max of each reading
and of the frame age on arrival (now - capture time, CLOCK_MONOTONIC), and appends one
row to --csv. With --seconds 0 it runs until Ctrl-C. --trace FILE also writes every frame
(time, readings, age): pushing a box slowly toward the robot, the trace shows the reading
falling steadily until the stereo module stops measuring -- its minimum range, no tape needed.
ROS 2 (foxy or later), rclpy.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1_perception import pipeline as PL  # noqa: E402

FIELDS = ("person_m", "obstacle_m", "corridor_points", "depth_valid", "center_valid")


def stats(xs):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not xs:
        return None, None, None, 0
    return xs[len(xs) // 2], xs[0], xs[-1], len(xs)


def fmt(v):
    return "-" if v is None else "{:.3f}".format(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--topic", default="/r1_perception/proximity")
    ap.add_argument("--trace", default="", help="CSV of every frame")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray

    rows, bad = [], [0]

    def on_msg(m):
        try:
            p = PL.parse_proximity(list(m.data))
        except ValueError as exc:
            bad[0] += 1
            if bad[0] == 1:
                print("bad message: {}".format(exc))
            return
        p["age"] = time.monotonic() - p["t_cap_mono"]
        rows.append(p)

    rclpy.init()
    node = Node("prox_watch")
    node.create_subscription(Float64MultiArray, args.topic, on_msg,
                             QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    t0 = time.monotonic()
    t_print = t0
    trace = None
    if args.trace:
        trace = open(os.path.expanduser(args.trace), "w")
        trace.write("t_s,seq,person_m,person_near_box,obstacle_m,corridor_points,depth_valid,"
                    "center_valid,depth_ok,age_s\n")
    n_traced = 0
    try:
        while rclpy.ok() and (args.seconds <= 0 or time.monotonic() - t0 < args.seconds):
            rclpy.spin_once(node, timeout_sec=0.05)
            while trace is not None and n_traced < len(rows):
                r = rows[n_traced]
                n_traced += 1
                trace.write(",".join("" if v is None else str(v) for v in (
                    round(r["t_cap_mono"] - t0, 3), int(r["seq"]), r["person_m"],
                    int(r["person_near_box"]), r["obstacle_m"], r["corridor_points"],
                    r["depth_valid"], r["center_valid"], int(r["depth_ok"]), round(r["age"], 4))) + "\n")
            if time.monotonic() - t_print >= 0.5 and rows:
                t_print = time.monotonic()
                p = rows[-1]
                print("  person {:>6} {}| obstacle {:>6} ({} pts) | depth valid {:.2f} center {:.2f}{} | age {:.0f} ms"
                      .format(fmt(p["person_m"]), "(near box) " if p["person_near_box"] else "",
                              fmt(p["obstacle_m"]), int(p["corridor_points"] or 0),
                              p["depth_valid"] or 0.0, p["center_valid"] or 0.0,
                              "" if p["depth_ok"] else " NO DEPTH", p["age"] * 1e3), flush=True)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    if trace is not None:
        trace.close()
        print("{} frames traced to {}".format(n_traced, args.trace))

    dur = time.monotonic() - t0
    if not rows:
        print("no messages on {} in {:.1f} s (is r1_perception running?)".format(args.topic, dur))
        return 1
    print("\n{} frames in {:.1f} s ({:.1f} Hz){}".format(
        len(rows), dur, len(rows) / dur, "  label: " + args.label if args.label else ""))
    out = {"label": args.label, "frames": len(rows), "hz": round(len(rows) / dur, 2),
           "near_box_frames": sum(1 for r in rows if r["person_near_box"]),
           "no_depth_frames": sum(1 for r in rows if not r["depth_ok"])}
    for k in FIELDS + ("age",):
        med, lo, hi, n = stats([r[k] for r in rows])
        print("  {:<16} median {:>7}  min {:>7}  max {:>7}  ({} of {} frames)".format(
            k, fmt(med), fmt(lo), fmt(hi), n, len(rows)))
        out[k + "_med"], out[k + "_min"], out[k + "_max"] = med, lo, hi
    print("  person box near : {} frames, no depth: {} frames".format(
        out["near_box_frames"], out["no_depth_frames"]))
    if args.csv:
        path = os.path.expanduser(args.csv)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out))
            if new:
                w.writeheader()
            w.writerow(out)
        print("appended to {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
