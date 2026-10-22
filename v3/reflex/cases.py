"""V3 reflex golden cases: scripted inputs, hand-written expectations, and the tick-by-tick
table the C++ state machine must reproduce.

Each case scripts, per 50 Hz control step, what the bridge would have: mission_ctl's
command, a proximity frame every 4th step (12.5 Hz, captured 0.1 s before it arrives),
reset requests, whether the bridge is RUNNING. Distances are scripted against the
capture time, not reacted to -- these test the state machine, not the robot.

Two layers of checking:
  * checkpoints, written by hand from the rules (v3/README.md, 2026-10-22): at time
    t the state, forward command and reason must be such-and-such; some states must
    never occur. test_reflex.py runs them against reflex.py.
  * golden.csv, every tick of every case as reflex.py computes it. The C++ test reads
    the inputs from it and must produce the same outputs; test_reflex.py also fails if
    reflex.py no longer reproduces it (regenerate deliberately: cases.py --write).

    python3 v3/reflex/cases.py --write
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reflex as RX  # noqa: E402

DT = 0.02
T0 = 1000.0          # CLOCK_MONOTONIC seconds at tick 0; any value works
FRAME_EVERY = 4
FRAME_LATENCY = 0.1
GOLDEN = HERE / "golden.csv"

CLEAR = {"person_m": None, "person_near_box": False, "obstacle_m": 3.0, "depth_ok": True,
         "center_valid": 0.95}


def scene(**kw):
    out = dict(CLEAR)
    out.update(kw)
    return out


class Case(object):
    def __init__(self, name, duration_s, cmd, world, checks, never=(), resets=(),
                 running=None, cfg=None, expect_retreats=None):
        """cmd(t) -> (vx, vy, wz); world(t_cap) -> scene dict, or None for no frame;
        times relative to tick 0. checks: (t, state, vx or None, reason or None).
        never: states that must not occur. resets: times of reset requests."""
        self.name, self.duration_s = name, duration_s
        self.cmd, self.world, self.checks, self.never = cmd, world, checks, tuple(never)
        self.resets = tuple(resets)
        self.running = running or (lambda t: True)
        self.cfg = dict(cfg or {})
        self.expect_retreats = expect_retreats

    def ticks(self):
        n = int(round(self.duration_s / DT))
        reset_k = {int(round(t / DT)) for t in self.resets}
        seq = 0
        for k in range(n):
            t_rel = k * DT
            frame = None
            if k % FRAME_EVERY == 0:
                w = self.world(t_rel - FRAME_LATENCY)
                if w is not None:
                    seq += 1
                    frame = RX.Frame(seq, T0 + t_rel - FRAME_LATENCY, w["person_m"],
                                     w["person_near_box"], w["obstacle_m"], w["depth_ok"],
                                     w["center_valid"])
            yield k, t_rel, T0 + t_rel, self.cmd(t_rel), frame, k in reset_k, \
                bool(self.running(t_rel))


def const(v):
    return lambda t: v


def step_cmd(*pieces):
    """step_cmd((t0, cmd0), (t1, cmd1), ...): cmd_i from t_i on."""
    def f(t):
        out = pieces[0][1]
        for t_i, c in pieces:
            if t >= t_i:
                out = c
        return out
    return f


def step_world(*pieces):
    """step_world((t0, scene0), (t1, scene1), ...); a scene of None = no frame."""
    def f(t):
        out = pieces[0][1]
        for t_i, w in pieces:
            if t >= t_i:
                out = w
        return out
    return f


S, A, R, ST, OFF = RX.STAND, RX.ARMED, RX.RETREAT, RX.STAND, RX.OFF
FWD3, FWD5, ZERO = (0.3, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.0)

CASES = [
    Case("clear_walk", 4.0, const(FWD3), const(scene()),
         [(1.0, A, 0.3, RX.R_NONE), (3.9, A, 0.3, RX.R_NONE)], never=(R, ST)),

    # A person walks up to a standing robot at 0.5 m/s and stops 0.5 m away. Inside
    # 1.2 m: forward blocked (it is standing anyway). Below 0.6 m on two frames: step
    # back 1.5 s. Then stand, and stay standing while the person stays close.
    Case("person_approach_standing", 8.0, const(ZERO),
         lambda t: scene(person_m=max(0.5, 2.0 - 0.5 * t)),
         [(1.0, A, 0.0, RX.R_NONE), (2.0, A, 0.0, RX.R_STOP_PERSON),
          (3.10, R, -0.3, RX.R_RETREAT_PERSON), (4.50, R, -0.3, RX.R_RETREAT_PERSON),
          (4.60, ST, 0.0, RX.R_STAND), (7.9, ST, 0.0, RX.R_STAND)], expect_retreats=1),

    # One frame says 0.4 m; the next says 2 m. Not two frames: no retreat.
    Case("single_false_positive", 4.0, const(ZERO),
         lambda t: scene(person_m=0.4) if abs(t - 1.9) < 0.03 else scene(),
         [(2.1, A, 0.0, None), (3.9, A, 0.0, RX.R_NONE)], never=(R, ST)),

    # Walking at 0.5 m/s toward an obstacle that closes to 0.9 m: blocked inside the
    # stop distance at 0.5 m/s (1.03 m). At 1.08 m still blocked (release needs
    # 1.03 + 0.10); at 1.2 m released. Never a retreat (0.9 > 0.45).
    Case("walk_to_obstacle_block", 8.0, const(FWD5),
         lambda t: scene(obstacle_m=1.2 if t >= 6.5 else 1.08 if t >= 5.0
                         else max(0.9, 3.0 - 0.5 * t)),
         [(1.0, A, 0.5, RX.R_NONE), (4.5, A, 0.0, RX.R_STOP_OBSTACLE),
          (5.8, A, 0.0, RX.R_STOP_OBSTACLE), (7.2, A, 0.5, RX.R_NONE)], never=(R, ST)),

    # Standing; an obstacle is pushed toward the robot at 0.3 m/s, down to 0.3 m.
    Case("obstacle_pushed_close", 5.0, const(ZERO),
         lambda t: scene(obstacle_m=max(0.3, 1.0 - 0.3 * max(0.0, t))),
         # 0.73 m at 1.0 s is outside the standstill stop distance (0.64 m); 0.61 m at 1.4 s is not
         [(1.0, A, 0.0, RX.R_NONE), (1.4, A, 0.0, RX.R_STOP_OBSTACLE),
          (2.2, R, -0.3, RX.R_RETREAT_OBSTACLE),
          (3.7, ST, 0.0, RX.R_STAND)], expect_retreats=1),

    # Frames stop at 1 s while walking: forward is blocked once the newest is 0.3 s old;
    # a backward command still passes.
    Case("stale_while_walking", 3.5, step_cmd((0.0, FWD3), (2.5, (-0.2, 0.0, 0.0))),
         lambda t: scene() if t < 0.9 else None,
         [(0.8, A, 0.3, RX.R_NONE), (1.5, A, 0.0, RX.R_STALE), (2.8, A, -0.2, RX.R_STALE)],
         never=(R, ST)),

    # Frames stop in the middle of a retreat: it finishes, then stands.
    Case("stale_during_retreat", 3.0, const(ZERO),
         lambda t: scene(person_m=0.4) if t < 0.4 else None,
         [(1.0, R, -0.3, RX.R_RETREAT_PERSON), (1.7, ST, 0.0, RX.R_STAND)], expect_retreats=1),

    # Three approaches in 10 s: two retreats, then the limit latches -- standing, needs
    # help -- and a fourth approach changes nothing.
    Case("retreat_limit", 8.0, const(ZERO),
         step_world((-1.0, scene(person_m=0.4)), (0.5, scene(person_m=2.0)),
                    (2.5, scene(person_m=0.4)), (3.0, scene(person_m=2.0)),
                    (5.0, scene(person_m=0.4)), (5.5, scene(person_m=2.0)),
                    (7.0, scene(person_m=0.4)), (7.5, scene(person_m=2.0))),
         [(0.5, R, -0.3, RX.R_RETREAT_PERSON), (1.7, ST, 0.0, RX.R_STAND),
          (3.0, R, -0.3, RX.R_RETREAT_PERSON), (4.5, ST, 0.0, RX.R_STAND),
          (5.4, ST, 0.0, RX.R_LIMIT), (7.4, ST, 0.0, RX.R_LIMIT), (7.9, ST, 0.0, RX.R_LIMIT)],
         expect_retreats=2),

    # After a retreat the person is at 0.7 m: a reset is refused (not clear of 0.6 +
    # 0.2). At 1.0 m it is accepted; walking forward is then blocked by the stop
    # distance (1.2 m) until the person is at 2 m.
    Case("reset_refused_then_accepted", 7.5, step_cmd((0.0, ZERO), (5.0, FWD3)),
         step_world((-1.0, scene(person_m=0.4)), (2.0, scene(person_m=0.7)),
                    (4.0, scene(person_m=1.0)), (6.0, scene(person_m=2.0))),
         [(1.7, ST, 0.0, RX.R_STAND), (3.0, ST, 0.0, RX.R_RESET_REFUSED),
          (3.02, ST, 0.0, RX.R_STAND), (5.0, A, 0.0, RX.R_STOP_PERSON),
          (6.5, A, 0.3, RX.R_NONE)],
         resets=(3.0, 5.0), expect_retreats=1),

    # Depth drops out while walking: forward blocked. A person then fills the picture
    # (box taller than person_box_frac, no depth): near, so a retreat.
    Case("depth_lost_box_person", 4.5, const(FWD3),
         step_world((-1.0, scene()), (1.0, scene(depth_ok=False, obstacle_m=None)),
                    (2.0, scene(depth_ok=False, obstacle_m=None, person_near_box=True))),
         [(0.8, A, 0.3, RX.R_NONE), (1.5, A, 0.0, RX.R_NO_DEPTH),
          (2.3, R, -0.3, RX.R_RETREAT_PERSON), (4.0, ST, 0.0, RX.R_STAND)], expect_retreats=1),

    # The middle of the depth image goes blank (something too near for stereo): forward
    # blocked, but no retreat on a guess.
    Case("depth_hole", 3.0, const(FWD3),
         step_world((-1.0, scene()), (1.0, scene(center_valid=0.05, obstacle_m=None))),
         [(0.8, A, 0.3, RX.R_NONE), (1.5, A, 0.0, RX.R_DEPTH_HOLE)], never=(R, ST)),

    Case("disabled", 2.0, const(FWD3), const(scene(person_m=0.3, obstacle_m=0.3)),
         [(1.0, OFF, 0.3, RX.R_DISABLED)], never=(A, R, ST), cfg={"enable": False}),

    # Not RUNNING: off, and frames seen meanwhile do not count toward a retreat. Back on,
    # until the first new frame the camera counts as stale: forward blocked.
    Case("not_running", 4.0, const((0.1, 0.0, 0.0)),
         step_world((-1.0, scene()), (1.9, scene(person_m=0.4))),
         [(0.5, OFF, 0.1, RX.R_NOT_RUNNING), (1.5, A, 0.1, RX.R_NONE),
          (2.5, OFF, 0.1, RX.R_NOT_RUNNING), (3.02, A, 0.0, RX.R_STALE),
          (3.06, A, 0.0, RX.R_STOP_PERSON),
          (3.2, R, -0.3, RX.R_RETREAT_PERSON)],
         running=lambda t: (1.0 <= t < 2.0) or t >= 3.0, expect_retreats=1),

    # The stop distance uses the fastest command of the last second: just after 0.5
    # m/s drops to 0.1 m/s, an obstacle at 0.85 m is inside the 0.5 m/s distance
    # (1.03 m) though outside the 0.1 m/s one (0.71 m + 0.10 release).
    Case("speed_window", 4.0, step_cmd((0.0, FWD5), (2.0, (0.1, 0.0, 0.0))),
         step_world((-1.0, scene()), (2.0, scene(obstacle_m=0.85))),
         [(1.5, A, 0.5, RX.R_NONE), (2.5, A, 0.0, RX.R_STOP_OBSTACLE),
          (3.2, A, 0.1, RX.R_NONE)], never=(R, ST)),

    # In STAND a second retreat needs a new approach: the person staying at 0.5 m changes
    # nothing; stepping away to 1.5 m and coming back to 0.4 m does.
    Case("rearm_new_approach", 7.0, const(ZERO),
         step_world((-1.0, scene(person_m=0.4)), (0.5, scene(person_m=0.5)),
                    (3.0, scene(person_m=1.5)), (4.0, scene(person_m=0.4))),
         [(1.0, R, -0.3, RX.R_RETREAT_PERSON), (2.9, ST, 0.0, RX.R_STAND),
          (3.8, ST, 0.0, RX.R_STAND), (4.3, R, -0.3, RX.R_RETREAT_PERSON),
          (6.0, ST, 0.0, RX.R_STAND)], expect_retreats=2),

    # A reset during a retreat is ignored; one in STAND with the way clear is not.
    Case("reset_during_retreat_ignored", 3.0, const(ZERO),
         step_world((-1.0, scene(person_m=0.4)), (0.2, scene(person_m=2.0))),
         [(0.5, R, -0.3, RX.R_RETREAT_PERSON), (1.0, R, -0.3, RX.R_RETREAT_PERSON),
          (1.7, ST, 0.0, RX.R_STAND), (2.0, A, 0.0, RX.R_NONE)],
         resets=(0.5, 2.0), expect_retreats=1),

    # Inside the stop distance only forward motion is cut: backward passes, and a
    # diagonal command keeps its sideways and turning parts.
    Case("blocked_forward_only", 3.0, step_cmd((0.0, (-0.3, 0.0, 0.0)), (2.0, (0.2, 0.2, 0.3))),
         const(scene(person_m=1.0)),
         [(1.0, A, -0.3, RX.R_STOP_PERSON), (2.5, A, 0.0, RX.R_STOP_PERSON)], never=(R, ST)),
]


def run_case(case):
    """[(k, t_rel, t, cmd, frame, reset, running, Output)] for one case."""
    rx = RX.Reflex(case.cfg)
    rows = []
    for k, t_rel, t, cmd, frame, reset, running in case.ticks():
        out = rx.step(t, cmd, frame, reset, running)
        rows.append((k, t_rel, t, cmd, frame, reset, running, out))
    return rows


def _f(x):
    if x is None:
        return "nan"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    return repr(float(x))


HEADER = ("case,k,t,running,reset,cmd_vx,cmd_vy,cmd_wz,frame,seq,t_cap,person_m,person_near_box,"
          "obstacle_m,depth_ok,center_valid,out_state,out_vx,out_vy,out_wz,out_reason,"
          "out_blocked,out_refused")


def golden_text():
    lines = ["# V3 reflex golden cases -- generated by v3/reflex/cases.py from reflex.py; do not",
             "# edit. Inputs, then the outputs the C++ state machine must reproduce. nan = none.",
             "# states: " + " ".join("{}={}".format(i, s) for i, s in enumerate(RX.STATES)),
             "# defaults: " + " ".join("{}={}".format(k, _f(v)) for k, v in RX.DEFAULTS.items()),
             HEADER]
    for case in CASES:
        if case.cfg:
            lines.append("#cfg {} ".format(case.name) + " ".join(
                "{}={}".format(k, _f(v)) for k, v in sorted(case.cfg.items())))
        for k, t_rel, t, cmd, fr, reset, running, out in run_case(case):
            fcols = [_f(None)] * 7 if fr is None else [
                _f(fr.seq), _f(fr.t_cap), _f(fr.person_m), _f(fr.person_near_box),
                _f(fr.obstacle_m), _f(fr.depth_ok), _f(fr.center_valid)]
            lines.append(",".join(
                [case.name, str(k), _f(t), _f(running), _f(reset)] + [_f(c) for c in cmd]
                + [_f(fr is not None)] + fcols
                + [str(RX.STATES.index(out.state)), _f(out.vx), _f(out.vy), _f(out.wz),
                   str(out.reason), _f(out.blocked), _f(out.reset_refused)]))
    return "\n".join(lines) + "\n"


def check_case(case):
    """Problems with one case against its hand-written expectations, as strings."""
    rows = run_case(case)
    bad = []
    for t_chk, state, vx, reason in case.checks:
        k = int(round(t_chk / DT))
        if k >= len(rows):
            bad.append("t={} beyond the case".format(t_chk))
            continue
        out = rows[k][-1]
        got = (out.state, round(out.vx, 9), out.reason)
        if out.state != state or (vx is not None and abs(out.vx - vx) > 1e-9) or \
                (reason is not None and out.reason != reason):
            bad.append("t={:.2f}: expected {} vx={} {}; got {} vx={} {} ({})".format(
                t_chk, state, vx, None if reason is None else RX.reason_text(reason),
                got[0], got[1], RX.reason_text(got[2]), got[2]))
    states = [r[-1].state for r in rows]
    for s in case.never:
        if s in states:
            bad.append("state {} occurred (first at t={:.2f})".format(
                s, rows[states.index(s)][1]))
    if case.expect_retreats is not None:
        n = sum(1 for a, b in zip([None] + states, states) if b == RX.RETREAT and a != RX.RETREAT)
        if n != case.expect_retreats:
            bad.append("{} retreats, expected {}".format(n, case.expect_retreats))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="regenerate golden.csv")
    args = ap.parse_args()
    failed = 0
    for case in CASES:
        bad = check_case(case)
        failed += bool(bad)
        print("{:5} {}".format("ok" if not bad else "FAIL", case.name))
        for b in bad:
            print("        " + b)
    if args.write:
        if failed:
            print("not writing golden.csv: {} case(s) fail their checkpoints".format(failed))
            return 1
        text = golden_text()
        rows = sum(1 for ln in text.splitlines() if ln and not ln.startswith("#")) - 1
        GOLDEN.write_text(text)
        print("wrote {} ({} ticks, {} cases)".format(GOLDEN, rows, len(CASES)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
