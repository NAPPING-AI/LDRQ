#!/usr/bin/env python3
"""The stage E trial protocol -- 10 x turn_to, 10 x walk_to -- in a simulated room.

The robot here does exactly what it is told and the camera adds Gaussian noise to the
bearing and range, so this measures the CONTROL LOGIC (grounding, turn-by-bearing,
walk-in-pieces, re-measure, face again), not the robot: the real trials are
docs/stageE_perception.md's. It gives the number the robot's should be compared with,
and the same table format.

  python3 mission_ctl/eval/sim_object_trials.py [--bearing-noise 1.5] [--range-noise 0.04]

Truth here is the simulator's: the final heading error is the true bearing of the target
when the robot stops, the stop error is the true horizontal distance minus the asked one.
Stdlib only.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from r1_mission import executor as ex  # noqa: E402
from r1_mission import node as N  # noqa: E402
from r1_mission import percept as PC  # noqa: E402
from r1_mission import plan as P  # noqa: E402


def trial(script, obj, lim, env, bn, rn, seed, hz=50.0):
    prims = P.compile_plan(P.parse_script(script), env, lim)[0]
    w = PC.FakeWorld([obj], bearing_noise_deg=bn, range_noise_m=rn, seed=seed)
    why, _ = N.ground_plan(prims, lim, N.world_frames(w))
    if why:
        return {"ok": False, "why": "refused: " + why}
    e = ex.Executor(prims, lim)
    pc, nf, t, dt = PC.Percept(), 0.0, 0.0, 1.0 / hz
    while t < 300.0:
        while nf <= t:
            pc.add(w.frame(nf))
            nf += 1.0 / N.FRAME_HZ
        out = e.step(t, yaw=w.yaw, bridge_state="RUNNING", percept=pc)
        if out.state in (ex.DONE, ex.ABORTED):
            break
        w.move(out.vx, out.wz, dt)
        t += dt
    b, r, _, _ = w.relative(w.objects[0])
    rec = e.report()["executed"][0]
    return {"ok": e.state == ex.DONE, "why": e.abort_reason, "heading_err": b, "range": r,
            "t": t, "turns": len(rec.get("turns_deg", [])), "walks": len(rec.get("segments", []))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bearing-noise", type=float, default=1.5)
    ap.add_argument("--range-noise", type=float, default=0.04)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    lim = N.limits_from_cfg(N.read_flat_yaml(str(N.DEFAULT_CFG)))
    env = P.Envelope((0.0, 1.0), (0.0, 0.0), (-0.5, 0.5))
    rnd = random.Random(a.seed)
    print("camera noise: bearing sd {:.1f} deg, range sd {:.2f} m; {} trials each, seed {}"
          .format(a.bearing_noise, a.range_noise, a.n, a.seed))
    rows = {"turn_to": [], "walk_to": []}
    for i in range(a.n):
        b = rnd.choice([-1, 1]) * rnd.uniform(8.0, 45.0)        # inside the 100 deg view
        r = rnd.uniform(1.5, 4.0)
        obj = {"cls": "chair", "x": r * math.cos(math.radians(b)), "y": r * math.sin(math.radians(b))}
        rows["turn_to"].append(dict(trial("turn_to the chair", obj, lim, env, a.bearing_noise,
                                          a.range_noise, a.seed + i), start=(b, r)))
        stop = rnd.choice([0.8, 1.0, 1.5])
        r = rnd.uniform(stop + 1.0, 4.8)
        obj = {"cls": "chair", "x": r * math.cos(math.radians(b)), "y": r * math.sin(math.radians(b))}
        res = trial("walk_to the chair stop_at {}".format(stop), obj, lim, env, a.bearing_noise,
                    a.range_noise, a.seed + 100 + i)
        rows["walk_to"].append(dict(res, start=(b, r), stop=stop))
    for name, rs in rows.items():
        print("\n{}".format(name))
        print("  {:>3} {:>14} {:>9} {:>11} {:>11} {:>6} {:>6} {:>6}".format(
            "#", "start", "result", "heading", "stop err", "turns", "walks", "s"))
        for i, x in enumerate(rs):
            serr = x["range"] - x["stop"] if name == "walk_to" and x["ok"] else None
            print("  {:>3} {:>+6.1f} deg {:>4.2f} m {:>9} {:>+9.2f} deg {:>11} {:>6} {:>6} {:>6.1f}".format(
                i + 1, x["start"][0], x["start"][1], "DONE" if x["ok"] else "ABORTED",
                x.get("heading_err", float("nan")),
                "{:+.3f} m".format(serr) if serr is not None else "-", x.get("turns", "-"),
                x.get("walks", "-"), x.get("t", 0.0)))
        ok = [x for x in rs if x["ok"]]
        he = [abs(x["heading_err"]) for x in ok]
        print("  success {}/{} | |heading err| mean {:.2f} max {:.2f} deg".format(
            len(ok), len(rs), statistics.mean(he) if he else float("nan"), max(he) if he else float("nan")))
        if name == "walk_to" and ok:
            se = [x["range"] - x["stop"] for x in ok]
            print("  stop error mean {:+.3f} m, |err| mean {:.3f} max {:.3f} m".format(
                statistics.mean(se), statistics.mean(abs(v) for v in se), max(abs(v) for v in se)))
        for x in rs:
            if not x["ok"]:
                print("  aborted: {}".format(x["why"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
