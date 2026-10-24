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

print("== the conductor: two policies, the dance, the reflex (dance_cases.py) ==")
import conductor as CD  # noqa: E402
import dance_cases as DC  # noqa: E402

for case in DC.CASES:
    bad = DC.check_case(case)
    check(case.name, not bad, "; ".join(bad))
check("golden_dance.csv is what conductor.py produces now (dance_cases.py --write)",
      DC.GOLDEN.read_text() == DC.golden_text())
cd = CD.Conductor()
check("the dance's bars are routine.py's: 3 + 3 + 2 + 2 s at 50 Hz, boundaries 0/150/300/400",
      cd.loop == 500 and cd.boundaries == (0, 150, 300, 400))
check("dance trigger 0.9 m, start clear 1.0 m (user, 2026-10-23)",
      CD.DANCE_DEFAULTS["dance_person_trigger_m"] == 0.9 and CD.DANCE_DEFAULTS["dance_clear_m"] == 1.0)
inv = []
for case in DC.CASES:
    rows = DC.run_case(case)
    prev = None
    for r in rows:
        out = r[-1]
        if (out.policy == CD.WALK) != (out.dance_tick == -1):
            inv.append("{} t={:.2f}: policy {} with tick {}".format(case.name, r[1], out.policy, out.dance_tick))
        if out.policy == CD.DANCE and out.reflex.cmd() != (0.0, 0.0, 0.0):
            inv.append("{} t={:.2f}: dancing, but the walking policy is commanded {}".format(
                case.name, r[1], out.reflex.cmd()))
        if prev is not None and prev.policy == CD.DANCE and out.policy == CD.DANCE \
                and out.dance_tick != prev.dance_tick + 1:
            inv.append("{} t={:.2f}: dance tick jumped {} -> {}".format(case.name, r[1], prev.dance_tick, out.dance_tick))
        if prev is not None and prev.policy == CD.DANCE and out.policy == CD.WALK \
                and out.dance_reason == CD.D_DONE and out.reflex.state == RX.ARMED \
                and not cd.at_boundary(prev.dance_tick + 1):
            inv.append("{} t={:.2f}: a normal end off a bar boundary".format(case.name, r[1]))
        prev = out
check("every step: walking <=> no dance tick; dancing => v2 gets (0,0,0); the tick counts "
      "by one; a normal end only on a bar boundary", not inv, inv[:3])
routine_dir = HERE.parent / "dance"
if (routine_dir / "routine.py").exists():
    sys.path.insert(0, str(routine_dir))
    try:
        import routine as RT  # noqa: E402
        check("conductor's bar ticks are routine.py's BARS",
              tuple(int(round(b[1] * 50)) for b in RT.BARS) == CD.DANCE_DEFAULTS["dance_bar_ticks"])
    except ImportError as exc:   # numpy missing
        print("  skip  routine import: {}".format(exc))

print("== the bridge (C++) and its config agree with the reference ==")
import re  # noqa: E402

# v3/ros2_ws/... in the repo, deploy/ros2_ws/... in an installed tree
BRIDGE = next(d for d in (HERE.parent / "ros2_ws" / "src" / "r1_hw_bridge",
                          HERE.parent / "deploy" / "ros2_ws" / "src" / "r1_hw_bridge") if d.is_dir())
hpp = (BRIDGE / "include" / "r1_hw_bridge" / "reflex_fsm.hpp").read_text()
idx = {m.group(1): int(m.group(2)) for m in re.finditer(r"\bkProx(\w+?)(?:Idx)? = (\d+)[,;]", hpp)}


def camel(name):
    return "".join(w.capitalize() for w in name.split("_"))


try:
    from r1_perception import pipeline as PL0
    layout = PL0.PROX_LAYOUT
except ImportError:
    layout = None
if layout is not None:
    want = {camel(k).replace("Mono", ""): i for i, k in enumerate(layout)}
    got = {k: v for k, v in idx.items() if k != "Len"}
    check("reflex_fsm.hpp's proximity indices are perception's PROX_LAYOUT",
          got == want and idx.get("Len") == len(layout), (got, want))
bv = [float(x) for x in re.search(r"kBrakeV\[\] = \{([^}]*)\}", hpp).group(1).split(",")]
bd = [float(x) for x in re.search(r"kBrakeD\[\] = \{([^}]*)\}", hpp).group(1).split(",")]
check("reflex_fsm.hpp's braking table is stop_distance.py's",
      list(zip(bv, bd)) == list(SD.BRAKE_WORST), list(zip(bv, bd)))
yaml = {}
for line in (BRIDGE / "config" / "bridge.yaml").read_text().splitlines():
    m = re.match(r"\s+reflex_(\w+):\s*([^#]+)", line)
    if m:
        yaml[m.group(1)] = m.group(2).strip().strip('"')
cfg_keys = set(RX.DEFAULTS) - {"enable"}
check("bridge.yaml has every reflex parameter, and the reflex off",
      cfg_keys <= set(yaml) and yaml.get("enable") == "false", sorted(cfg_keys - set(yaml)))
bad = {k: (yaml.get(k), RX.DEFAULTS[k]) for k in cfg_keys if k in yaml
       and float(yaml[k]) != float(RX.DEFAULTS[k])}
check("bridge.yaml's reflex values are reflex.py's defaults", not bad, bad)

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
