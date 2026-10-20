#!/usr/bin/env python3
"""Detection precision and person recall on the robot's own pictures, and the two
confidence thresholds (stage E §3.5).

Input: a capture folder (NNNNNN.jpg = the rectified image YOLOX saw, NNNNNN.json = its
detections at the safety threshold) plus a label for every image, YOLO txt format, one
box per line "class_id cx cy w h" normalized to the image, class ids COCO's
(0 = person, 56 = chair, ...). Label from scratch -- not by correcting the model's own
boxes, which would make its misses invisible and its recall look perfect. An image with
nothing to label gets an empty .txt (that is how false positives on an empty room are
counted). Labels go in NNNNNN.txt next to the image, or in --labels DIR.

  python3 perception/tools/eval_detect.py --captures ~/r1_captures/eval \\
      [--classes person,chair,couch,dining\\ table,tv,bottle,cup,potted\\ plant] [--out res.json]

Reports, for the classes evaluated:
  * precision and recall per class at IoU >= 0.5, at a sweep of thresholds
  * person recall at every threshold -- what the SAFETY threshold is chosen for
  * the precision of CONFIRMED targets (same class + IoU across confirm_frames frames is
    the node's rule; on single pictures this is precision of one frame, a lower bound)
and suggests safety_conf (the highest threshold keeping person recall >= --person-recall)
and obj_target_conf (the lowest threshold reaching precision >= --target-precision).
Stdlib + the detections JSON: no model, no OpenCV needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from r1_perception.tracker import iou  # noqa: E402
from r1_perception.yolox import COCO_CLASSES  # noqa: E402

DEFAULT_CLASSES = ("person", "chair", "couch", "dining table", "tv", "laptop", "bottle", "cup",
                   "potted plant", "backpack")


def load(captures, labels_dir=None):
    items = []
    for js in sorted(glob.glob(os.path.join(os.path.expanduser(captures), "[0-9]*.json"))):
        base = js[:-5]
        msg = json.load(open(js))
        w, h = msg["img"]
        lab = os.path.join(labels_dir, os.path.basename(base) + ".txt") if labels_dir else base + ".txt"
        if not os.path.exists(lab):
            continue
        gt = []
        for line in open(lab).read().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            gt.append((COCO_CLASSES[c], ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                         (cx + bw / 2) * w, (cy + bh / 2) * h)))
        dets = [(d["cls"], d["bbox"], d["conf"]) for d in msg["dets"]]
        items.append({"name": os.path.basename(base), "gt": gt, "dets": dets})
    return items


def score(items, cls, thr, iou_thr=0.5):
    tp = fp = fn = 0
    for it in items:
        g = [b for c, b in it["gt"] if c == cls]
        d = sorted([(s, b) for c, b, s in it["dets"] if c == cls and s >= thr], reverse=True)
        used = set()
        for s, b in d:
            best, bj = 0.0, None
            for j, gb in enumerate(g):
                if j in used:
                    continue
                o = iou(b, gb)
                if o > best:
                    best, bj = o, j
            if bj is not None and best >= iou_thr:
                used.add(bj)
                tp += 1
            else:
                fp += 1
        fn += len(g) - len(used)
    p = tp / float(tp + fp) if tp + fp else None
    r = tp / float(tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--captures", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--thresholds", default="0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7")
    ap.add_argument("--person-recall", type=float, default=0.95)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    items = load(a.captures, a.labels)
    if not items:
        sys.exit("no labeled images in {} (NNNNNN.json + NNNNNN.txt)".format(a.captures))
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    thrs = [float(t) for t in a.thresholds.split(",")]
    lowest = min(min((s for _, _, s in it["dets"]), default=1.0) for it in items)
    n_gt = sum(len(it["gt"]) for it in items)
    print("{} labeled images, {} labeled objects; detections were saved at >= {:.2f}".format(
        len(items), n_gt, lowest))
    if lowest > min(thrs):
        print("  (thresholds below {:.2f} cannot be evaluated from these captures)".format(lowest))
    res = {"images": len(items), "objects": n_gt, "per_class": {}, "overall": {}}
    hdr = "{:<14}".format("class") + "".join("{:>11}".format("@" + str(t)) for t in thrs)
    print("\nprecision / recall (IoU >= 0.5)\n" + hdr)
    for c in classes:
        row = "{:<14}".format(c)
        res["per_class"][c] = {}
        for t in thrs:
            s = score(items, c, t)
            res["per_class"][c][str(t)] = s
            row += "{:>11}".format("-" if s["precision"] is None and s["recall"] is None else
                                   "{}/{}".format("-" if s["precision"] is None else "{:.2f}".format(s["precision"]),
                                                  "-" if s["recall"] is None else "{:.2f}".format(s["recall"])))
        print(row)
    row = "{:<14}".format("all of these")
    for t in thrs:
        tot = {"tp": 0, "fp": 0, "fn": 0}
        for c in classes:
            for k in tot:
                tot[k] += res["per_class"][c][str(t)][k]
        p = tot["tp"] / float(tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else None
        r = tot["tp"] / float(tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else None
        res["overall"][str(t)] = dict(tot, precision=p, recall=r)
        row += "{:>11}".format("{}/{}".format("-" if p is None else "{:.2f}".format(p),
                                              "-" if r is None else "{:.2f}".format(r)))
    print(row)
    pr = res["per_class"].get("person", {})
    safe = [t for t in thrs if pr.get(str(t), {}).get("recall") is not None
            and pr[str(t)]["recall"] >= a.person_recall]
    tgt = [t for t in thrs if res["overall"][str(t)]["precision"] is not None
           and res["overall"][str(t)]["precision"] >= a.target_precision]
    print("\nsafety_conf (person recall >= {:.2f}): {}".format(
        a.person_recall, "{:.2f}".format(max(safe)) if safe else "NONE of the thresholds -- the "
        "detector misses people here; do not rely on the camera guard"))
    print("obj_target_conf (precision >= {:.2f}): {}".format(
        a.target_precision, "{:.2f}".format(min(tgt)) if tgt else "NONE of the thresholds"))
    res["suggest"] = {"safety_conf": max(safe) if safe else None,
                      "obj_target_conf": min(tgt) if tgt else None}
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
        print("wrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
