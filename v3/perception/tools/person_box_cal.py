#!/usr/bin/env python3
"""person_box_k for proximity_mode person_box, without a tape measure.

In person_box mode a person's distance is person_box_k / (box height / image height).
This sets k from a person standing at a distance that needs no tape:

  arm's length  the helper faces the robot and stretches an arm out until the fingertips
                just touch its chest: about 0.65 m (--distance 0.65, the default)
  arm + a step  then one normal step back: about 1.3 m (--distance 1.3), the second
                reference, which checks the first

    python3 perception/tools/person_box_cal.py --seconds 5 --distance 0.65 --csv ~/o/v3_cal.csv
    python3 perception/tools/person_box_cal.py --seconds 5 --distance 1.3  --csv ~/o/v3_cal.csv
    python3 perception/tools/person_box_cal.py --from-csv ~/o/v3_cal.csv [--write]

Each run reads /r1_perception/detections for --seconds, takes the largest person box in
each frame, and prints the median box height (as a fraction of the image) and the k it
implies. A box that touches both the top and the bottom of the image cannot give k (its
height is cut off): the node already treats that as "right in front". --from-csv combines
the rows: k is the median of the usable ones, and the spread between them says how well
1 / distance fits this camera. --write puts k into perception.yaml (old file kept as
perception.yaml.before_cal). ROS 2 (foxy or later) for the reading runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import shutil
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1_perception import pipeline as PL  # noqa: E402

FIELDS = ("distance_m", "frames", "with_person", "h_frac_med", "h_frac_min", "h_frac_max",
          "touch_both", "k")


def largest_person(msg, safety_conf, edge):
    """(height fraction, touches top and bottom) of the largest person box, or None."""
    w_img, h_img = msg["img"]
    best = None
    for d in msg.get("dets", []):
        if d.get("cls") != "person" or d.get("conf", 0.0) < safety_conf:
            continue
        x0, y0, x1, y1 = d["bbox"]
        area = (x1 - x0) * (y1 - y0)
        if best is None or area > best[0]:
            touch = y0 <= edge * h_img and y1 >= (1.0 - edge) * h_img
            best = (area, (y1 - y0) / float(h_img), touch)
    return None if best is None else best[1:]


def summarize(samples, distance):
    """samples: [(h_frac, touch)] -> the CSV row for one reference."""
    hs = [h for h, t in samples if not t]
    touch = sum(1 for _, t in samples if t)
    row = {"distance_m": distance, "frames": None, "with_person": len(samples),
           "h_frac_med": None, "h_frac_min": None, "h_frac_max": None,
           "touch_both": touch, "k": None}
    if hs and touch <= len(samples) // 2:
        med = statistics.median(hs)
        row.update(h_frac_med=round(med, 4), h_frac_min=round(min(hs), 4),
                   h_frac_max=round(max(hs), 4), k=round(distance * med, 4))
    return row


def combine(rows):
    """k from the usable rows; (k, spread as a fraction, usable rows) or (None, None, 0)."""
    ks = [float(r["k"]) for r in rows if r.get("k") not in (None, "", "None")]
    if not ks:
        return None, None, 0
    k = statistics.median(ks)
    return k, (max(ks) - min(ks)) / k, len(ks)


def write_k(path, k, note):
    text = pathlib.Path(path).read_text()
    new, n = re.subn(r"(?m)^person_box_k:[^\n]*$",
                     "person_box_k: {:.3f}                       # {}".format(k, note), text)
    if n != 1:
        print("not written: {} has no single person_box_k line".format(path))
        return False
    shutil.copy2(path, path + ".before_cal")
    pathlib.Path(path).write_text(new)
    print("written to {} (the old file: {}.before_cal):".format(path, path))
    print("  " + next(line for line in new.splitlines() if line.startswith("person_box_k:")))
    return True


def read_live(seconds, safety_conf, edge, topic):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    got, frames = [], [0]

    def on_msg(m):
        try:
            msg = json.loads(m.data)
        except ValueError:
            return
        frames[0] += 1
        lp = largest_person(msg, safety_conf, edge)
        if lp is not None:
            got.append(lp)

    rclpy.init()
    node = Node("person_box_cal")
    node.create_subscription(String, topic, on_msg,
                             QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    t_end = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    return got, frames[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "config" / "perception.yaml"))
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--distance", type=float, default=0.65, help="m: 0.65 arm's length, 1.3 arm + a step")
    ap.add_argument("--csv", default="")
    ap.add_argument("--from-csv", default="", help="combine the rows of an earlier --csv")
    ap.add_argument("--write", action="store_true", help="with --from-csv: write k into --config")
    ap.add_argument("--topic", default="/r1_perception/detections")
    args = ap.parse_args()
    cfg = PL.read_flat_yaml(args.config)
    edge = float(cfg.get("person_box_edge", 0.02))
    safety = float(cfg.get("safety_conf", 0.30))

    if args.from_csv:
        rows = list(csv.DictReader(open(os.path.expanduser(args.from_csv))))
        for r in rows:
            print("  {} m: {} frames with a person, box height {} of the image, touching top and "
                  "bottom in {}, k {}".format(r["distance_m"], r["with_person"], r["h_frac_med"] or "-",
                                              r["touch_both"], r["k"] or "-"))
        k, spread, n = combine(rows)
        if k is None:
            print("no usable reference (every box touched the top and bottom, or no person seen)")
            return 1
        print("k = {:.3f} from {} reference(s){}; configured {}".format(
            k, n, ", spread {:.0%}".format(spread) if n > 1 else "", cfg.get("person_box_k")))
        for r in rows:
            if r.get("k") not in (None, "", "None"):
                d = float(r["distance_m"])
                est = k / float(r["h_frac_med"])
                print("  with k {:.3f}: the {:.2f} m reference reads {:.2f} m".format(k, d, est))
        if n > 1 and spread > 0.25:
            print("the references disagree by {:.0%}: 1 / distance does not fit this camera well -- "
                  "bring the CSV back before writing".format(spread))
            return 1
        if args.write:
            ok = write_k(args.config, k, "calibrated {} at {} m (no tape)".format(
                time.strftime("%F"), ", ".join(r["distance_m"] for r in rows if r.get("k"))))
            return 0 if ok else 1
        print("to keep it: the same command with --write")
        return 0

    samples, frames = read_live(args.seconds, safety, edge, args.topic)
    row = summarize(samples, args.distance)
    row["frames"] = frames
    print("{} detection messages, {} with a person; at {:.2f} m: box height {} of the image "
          "(min {}, max {}), touching top and bottom in {} frame(s) -> k {}".format(
              frames, row["with_person"], args.distance, row["h_frac_med"], row["h_frac_min"],
              row["h_frac_max"], row["touch_both"], row["k"]))
    if not samples:
        print("no person seen: is r1_perception running, and the helper facing the camera?")
        return 1
    if row["k"] is None:
        print("the box touches the top and bottom: at this distance the node already says "
              "'right in front' (person_near_box). Use the arm + a step reference for k.")
    if args.csv:
        path = os.path.expanduser(args.csv)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print("appended to {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
