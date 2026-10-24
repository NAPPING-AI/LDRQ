#!/usr/bin/env python3
"""B1 + B2 on a dev box: the B0 recordings, taken apart by stereo_unpack.py, through our
own stereo depth (r1_perception/stereo.py), against checks that need no tape measure.

    python3 perception/tools/stereo_eval.py pairs  <dir>                 # B1: how to pair
    python3 perception/tools/stereo_eval.py mask   <dir> --out shell.npz # B1: head shell
    python3 perception/tools/stereo_eval.py depth  <dir> --mask shell.npz [--height 1.2]

<dir> is one segment unpacked by stereo_unpack.py (5002.npy + 5002_frames.csv, 5003.*).

pairs   Are the two cameras' RTP timestamps one clock? For each left frame, the right frame
        that arrived nearest; their RTP difference should be a constant (any value: each
        stream starts at a random timestamp). If it is, frames pair by timestamp with that
        offset, whatever the network did; the report says how many pair, and how far
        apart they arrived. Writes <dir>/pairs.csv.
mask    The head shell in each rectified image: the pixels that never change while the
        hanging robot turns (the "turning" segment). Writes an npz with both masks.
depth   Every pair: rectified, matched (SGBM, no filling), the floor fitted (height and
        pitch: the floor must be a plane 1.2 m below the lens), the corridor's 30th-nearest
        point. Prints the medians and spreads, the frames with anything in the corridor
        nearer than --near, and the matching time. Writes <dir>/depth.csv.

numpy + cv2. The calibration (an OpenCV stereo npz): --calib, or $R1_STEREO_CALIB.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1_perception import stereo as ST  # noqa: E402

CALIB = os.environ.get("R1_STEREO_CALIB", "")
LEFT, RIGHT = "5002", "5003"


def read_frames(d, port):
    rows = list(csv.DictReader(open(str(pathlib.Path(d) / "{}_frames.csv".format(port)))))
    return [{"i": int(r["index"]), "ts": int(r["rtp_ts"]), "t": float(r["t_last"]),
             "ok": r["complete"] == "1"} for r in rows]


def signed32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def pair(d, max_skew_frac=0.5):
    """[(left index, right index, arrival gap s)] and a report dict."""
    L, R = read_frames(d, LEFT), read_frames(d, RIGHT)
    if not L or not R:
        raise SystemExit("{}: no frames for one of the ports".format(d))
    rt = np.array([r["t"] for r in R])
    step = statistics.median(np.diff([f["ts"] for f in L])) if len(L) > 1 else 3000
    period_s = step / 90000.0
    offs = []
    for f in L:
        j = int(np.argmin(np.abs(rt - f["t"])))
        # only frames that have a right frame arriving with them: a left frame whose right
        # partner was lost (or not decodable) would be "paired" with a neighbour
        if abs(rt[j] - f["t"]) < period_s / 2:
            offs.append(signed32(R[j]["ts"] - f["ts"]))
    if not offs:
        raise SystemExit("{}: no left frame has a right frame arriving with it".format(d))
    offs = np.array(offs)
    off = int(statistics.median(offs))
    spread = float(np.percentile(np.abs(offs - off), 90)) / 90.0       # ms at 90 kHz
    by_ts = {r["ts"]: r for r in R}
    pairs = []
    for f in L:
        want = (f["ts"] + off) & 0xFFFFFFFF
        cand = [r for ts, r in by_ts.items() if abs(signed32(ts - want)) <= max_skew_frac * step]
        if cand:
            r = min(cand, key=lambda r: abs(signed32(r["ts"] - want)))
            if f["ok"] and r["ok"]:
                pairs.append((f["i"], r["i"], r["t"] - f["t"]))
    gaps = [abs(g) for _, _, g in pairs]
    rep = {"left": len(L), "right": len(R), "rtp_step": step, "offset": off,
           "offset_spread_ms_p90": round(spread, 2), "one_clock": spread < 2.0,
           "pairs": len(pairs), "arrival_gap_ms_median": round(1000 * statistics.median(gaps), 1) if gaps else None,
           "arrival_gap_ms_max": round(1000 * max(gaps), 1) if gaps else None}
    return pairs, rep


def load_images(d):
    return np.load(str(pathlib.Path(d) / "{}.npy".format(LEFT))), \
        np.load(str(pathlib.Path(d) / "{}.npy".format(RIGHT)))


def cmd_pairs(args):
    pairs, rep = pair(args.dir)
    for k, v in rep.items():
        print("  {}: {}".format(k, v))
    if not rep["one_clock"]:
        print("  the RTP offset is not constant: the streams do not share a clock -- pair by arrival "
              "time only, and only while the head is still (the plan's fallback)")
    with open(str(pathlib.Path(args.dir) / "pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["left", "right", "arrival_gap_s"])
        w.writerows(pairs)
    print("  wrote {}/pairs.csv".format(args.dir))
    return 0 if rep["one_clock"] else 1


def cmd_mask(args):
    rig = ST.StereoRig(args.calib)
    Li, Ri = load_images(args.dir)
    idx = np.linspace(0, min(len(Li), len(Ri)) - 1, min(args.frames, len(Li), len(Ri))).astype(int)
    rl, rr = zip(*(rig.rectify(Li[i], Ri[i]) for i in idx))
    ml, fl = ST.shell_mask(rl, std_max=args.std, grow=args.grow)
    mr, fr = ST.shell_mask(rr, std_max=args.std, grow=args.grow)
    np.savez_compressed(args.out, left=ml, right=mr)
    print("  {} frames: shell {:.1%} of the left image, {:.1%} of the right; wrote {}".format(
        len(idx), fl, fr, args.out))
    return 0


def cmd_depth(args):
    rig = ST.StereoRig(args.calib, texture_min=args.texture_min)
    pairs, rep = pair(args.dir)
    Li, Ri = load_images(args.dir)
    m = np.load(args.mask) if args.mask else None
    rows, times = [], []
    for li, ri, gap in pairs[::args.every]:
        a, b = rig.rectify(Li[li], Ri[ri])
        t0 = time.monotonic()
        disp = rig.disparity(a, b, None if m is None else m["left"], None if m is None else m["right"])
        times.append(time.monotonic() - t0)
        fl = ST.fit_floor(disp, rig)
        z = rig.depth(disp)
        pts, _ = rig.points(z)
        near, ncorr = ST.corridor_nearest(pts, fl)
        rows.append({"left": li, "right": ri, "valid": float(np.isfinite(disp).mean()),
                     "floor_h": None if fl is None else fl["height_m"],
                     "pitch": None if fl is None else fl["pitch_down_deg"],
                     "roll": None if fl is None else fl["roll_deg"],
                     "floor_inliers": None if fl is None else fl["inliers"],
                     "corridor_points": ncorr, "corridor_m": near})
    if not rows:
        print("  no pairs")
        return 1
    with open(str(pathlib.Path(args.dir) / "depth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def med(key):
        xs = [r[key] for r in rows if r[key] is not None]
        if not xs:
            return "-"
        return "{:.3f} (p10 {:.3f}, p90 {:.3f})".format(np.median(xs), np.percentile(xs, 10), np.percentile(xs, 90))
    print("  {} pairs ({}); matched {}".format(len(rows), args.dir, med("valid")))
    print("  floor height m {}   expected {:.2f}".format(med("floor_h"), args.height))
    print("  pitch down deg {}   roll deg {}".format(med("pitch"), med("roll")))
    print("  floor inliers {}".format(med("floor_inliers")))
    near = [r for r in rows if r["corridor_m"] is not None and r["corridor_m"] < args.near]
    print("  corridor nearer than {:.2f} m in {} of {} frames{}".format(
        args.near, len(near), len(rows), ": " + ", ".join(
            "{:.2f}".format(r["corridor_m"]) for r in near[:10]) if near else ""))
    print("  matching ms per pair: median {:.1f}, p95 {:.1f} (this computer)".format(
        1000 * np.median(times), 1000 * np.percentile(times, 95)))
    hs = [r["floor_h"] for r in rows if r["floor_h"] is not None]
    ok = bool(hs) and abs(np.median(hs) - args.height) <= 0.05 * args.height
    print("  floor check (median within 5 % of {:.2f} m): {}".format(args.height, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("dir")
    p = sub.add_parser("mask")
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--std", type=float, default=2.0)
    p.add_argument("--grow", type=int, default=6)
    p = sub.add_parser("depth")
    p.add_argument("dir")
    p.add_argument("--mask", default="")
    p.add_argument("--height", type=float, default=1.2)
    p.add_argument("--near", type=float, default=0.6)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--texture-min", type=float, default=8.0)
    for q in sub.choices.values():
        q.add_argument("--calib", default=CALIB, required=not CALIB)
        q.add_argument("--ports", default="5002,5003", help="left,right as recorded")
    args = ap.parse_args()
    global LEFT, RIGHT
    LEFT, RIGHT = args.ports.split(",")
    return {"pairs": cmd_pairs, "mask": cmd_mask, "depth": cmd_depth}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
