#!/usr/bin/env python3
"""V3 reflex without a robot: the stop distance, the state machine against its hand-written
checkpoints (cases.py), golden.csv still reproduced, and perception's proximity message
read into the reflex's frame. Standard library + numpy (for the perception import).

  python3 v3/reflex/test_reflex.py
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "perception"))

import cases as CS  # noqa: E402
import reflex as RX  # noqa: E402
import stop_distance as SD  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok    " + name)
    else:
        FAIL += 1
        print("  FAIL  {}  <- {}".format(name, detail))


print("== stop distance ==")
check("braking table ends: 0.068 m at 0, 0.470 m at 1.0 m/s",
      abs(SD.brake(0.0) - 0.068) < 1e-12 and abs(SD.brake(1.0) - 0.470) < 1e-12)
check("braking interpolates: 0.4 m/s halfway between 0.3 and 0.5",
      abs(SD.brake(0.4) - (0.157 + 0.228) / 2) < 1e-12, SD.brake(0.4))
check("a backward command brakes like standing", SD.brake(-0.3) == SD.brake(0.0))
vs = [i / 20.0 for i in range(0, 21)]
for kind in ("person", "obstacle"):
    d = [SD.stop_distance(kind, v) for v in vs]
    check("{}: stop distance never shrinks with speed".format(kind),
          all(b >= a for a, b in zip(d, d[1:])), d)
check("floors hold: person 1.2 m, obstacle 0.6 m, at any speed",
      min(SD.stop_distance("person", v) for v in vs) >= 1.2
      and min(SD.stop_distance("obstacle", v) for v in vs) >= 0.6)
check("the plan's table: obstacle 0.87 m at 0.3 m/s, 1.54 m at 1.0; person 1.69 m at 1.0",
      round(SD.stop_distance("obstacle", 0.3), 2) == 0.87
      and round(SD.stop_distance("obstacle", 1.0), 2) == 1.54
      and round(SD.stop_distance("person", 1.0), 2) == 1.69)
for kind, trig in (("person", 0.6), ("obstacle", 0.45)):
    worst = max(SD.brake(v) + v * 0.25 - (SD.stop_distance(kind, v) - trig) for v in vs)
    check("{}: stopping from the stop distance (worst braking + 0.25 s) ends >= 0.10 m "
          "outside the trigger".format(kind), worst <= -0.10 + 1e-9, worst)
check("walk_to's minimum stop (V3 mission.yaml 0.9 m) is beyond the obstacle stop at "
      "obj_vx 0.3 m/s", SD.stop_distance("obstacle", 0.3) < 0.9)

print("== the spec's numbers ==")
d = RX.DEFAULTS
check("triggers: person 0.6 m, obstacle 0.45 m (user, 2026-10-22)",
      d["person_trigger_m"] == 0.6 and d["obstacle_trigger_m"] == 0.45)
check("retreat: -0.3 m/s for 1.5 s, no ramp (simulation, v3/README.md)", d["retreat_vx"] == -0.3 and d["retreat_s"] == 1.5)
check("two frames to confirm, two retreats per 10 s",
      d["confirm_frames"] == 2 and d["max_retreats"] == 2 and d["window_s"] == 10.0)
check("the reflex and stop_distance.py carry the same stop-distance numbers",
      RX.Reflex().sd_cfg == {k: SD.DEFAULTS[k] for k in RX.Reflex().sd_cfg})

print("== cases (hand-written checkpoints) ==")
for case in CS.CASES:
    bad = CS.check_case(case)
    check(case.name, not bad, "; ".join(bad))
names = [c.name for c in CS.CASES]
check("case names are unique", len(set(names)) == len(names))
check("golden.csv is what reflex.py produces now (cases.py --write to regenerate)",
      CS.GOLDEN.read_text() == CS.golden_text())

print("== perception's message -> the reflex's frame ==")
try:
    from r1_perception import pipeline as PL
except ImportError as exc:      # numpy missing
    print("  skip  perception import: {}".format(exc))
else:
    arr = PL.proximity_array({"person_m": 0.55, "person_near_box": False, "persons": 1,
                              "obstacle_m": None, "corridor_points": 0, "depth_valid": 0.9,
                              "center_valid": 0.8, "depth_ok": True}, 7, 1000.5, 1000.6)
    fr = RX.Frame.from_parsed(PL.parse_proximity(arr))
    check("seq, capture time and distances carried; no obstacle reading = none",
          fr.seq == 7 and fr.t_cap == 1000.5 and fr.person_m == 0.55 and fr.obstacle_m is None
          and fr.depth_ok and abs(fr.center_valid - 0.8) < 1e-12,
          [getattr(fr, k) for k in RX.Frame.__slots__])
    rx = RX.Reflex()
    outs = [rx.step(1000.52 + 0.08 * i, (0.0, 0.0, 0.0),
                    RX.Frame.from_parsed(PL.parse_proximity(
                        PL.proximity_array({"person_m": 0.5, "person_near_box": False, "persons": 1,
                                            "obstacle_m": None, "corridor_points": 0,
                                            "depth_valid": 0.9, "center_valid": 0.8,
                                            "depth_ok": True}, 8 + i, 1000.42 + 0.08 * i,
                                           1000.5 + 0.08 * i))))
            for i in range(2)]
    check("two of perception's frames with a person at 0.5 m start a retreat",
          [o.state for o in outs] == [RX.ARMED, RX.RETREAT], [o.state for o in outs])

print("--------------------------------------------------")
print("pass {}  fail {}".format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
