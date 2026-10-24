"""V3 conductor golden cases: the dance and the reflex together (conductor.py).

Like cases.py -- scripted inputs, hand-written checkpoints, and golden_dance.csv, every
tick as conductor.py computes it, for the bridge's C++ to reproduce -- with two more
inputs per tick: the dance message (none / True = dance, the heartbeat / False = end)
and whether the dance policy's output is fresh and finite.

    python3 v3/reflex/dance_cases.py --write
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cases as CS  # noqa: E402
import conductor as CD  # noqa: E402
import reflex as RX  # noqa: E402

DT = CS.DT
GOLDEN = HERE / "golden_dance.csv"
HB_EVERY = 10                 # mission_ctl's heartbeat: every 0.2 s


def heartbeat(*spans, stop_at=()):
    """dance(t): True every 0.2 s inside each (t0, t1), False at each stop time."""
    stops = {int(round(t / DT)) for t in stop_at}

    def f(t):
        k = int(round(t / DT))
        if k in stops:
            return False
        for t0, t1 in spans:
            k0 = int(round(t0 / DT))
            if k0 <= k < int(round(t1 / DT)) and (k - k0) % HB_EVERY == 0:
                return True
        return None
    return f


def approach(t_start, p0, speed, p_min):
    """person_m(t_cap): p0 until t_start, then nearer at `speed`, never below p_min."""
    return lambda tc: CS.scene(person_m=max(p_min, p0 - speed * max(0.0, tc - t_start)))


class DCase(CS.Case):
    def __init__(self, name, duration_s, cmd, world, dance, checks, dance_ok=None,
                 never_dance_after=None, **kw):
        """checks: (t, policy, dance_state, reflex state or None, vx or None, dance reason
        or None). never_dance_after: the policy must not be dance at or after this time."""
        super().__init__(name, duration_s, cmd, world, [], **kw)
        self.dance = dance
        self.dance_ok = dance_ok or (lambda t: True)
        self.dchecks = checks
        self.never_dance_after = never_dance_after


ZERO, FWD3 = CS.ZERO, CS.FWD3
W, D = CD.WALK, CD.DANCE
OFF, ON, END = CD.D_OFF, CD.D_DANCING, CD.D_ENDING
A, R, S = RX.ARMED, RX.RETREAT, RX.STAND

CASES = [
    # Stands, dances from 1.5 s; the heartbeat stops after 13.4 s -> ending at 13.92 s,
    # dance tick 621 -> handed back at the next bar boundary, tick 650 = 14.5 s.
    DCase("dance_heartbeat_end", 16.0, CS.const(ZERO), CS.const(CS.scene()),
          heartbeat((1.5, 13.5)),
          [(1.48, W, OFF, A, 0.0, None), (1.5, D, ON, A, 0.0, CD.D_NONE),
           (10.0, D, ON, A, 0.0, None), (14.0, D, END, A, 0.0, CD.D_END_HEARTBEAT),
           (14.48, D, END, None, None, None), (14.5, W, OFF, A, 0.0, CD.D_DONE)]),

    # Asked to stop at 5.1 s (tick 180, in bar 2) -> ends at tick 300 = 7.5 s.
    DCase("dance_stop_request", 9.0, CS.const(ZERO), CS.const(CS.scene()),
          heartbeat((1.5, 5.0), stop_at=(5.1,)),
          [(5.0, D, ON, A, 0.0, None), (5.2, D, END, A, 0.0, CD.D_END_REQUEST),
           (7.48, D, END, None, None, None), (7.5, W, OFF, A, 0.0, CD.D_DONE)]),

    # A person walks up to the dancing robot at 0.5 m/s from 3 m (from 4 s). Below
    # 0.9 m on two frames: the walking policy takes over on that step and steps back.
    # The mission keeps sending heartbeats: they are the old dance's, nothing restarts.
    DCase("dance_person_approach", 13.0, CS.const(ZERO), approach(4.0, 3.0, 0.5, 0.5),
          heartbeat((1.5, 12.9)),
          [(8.0, D, ON, A, 0.0, None), (8.6, W, OFF, R, -0.3, CD.D_STOP_RETREAT),
           (10.2, W, OFF, S, 0.0, CD.D_STOP_RETREAT), (12.8, W, OFF, S, 0.0, CD.D_STOP_RETREAT)],
          never_dance_after=8.6),

    # The same person at 0.7 m: a standing robot does not step back (trigger 0.6), and a
    # dance is refused (1.0 m must be clear). The person moves to 1.05 m at 4 s: the
    # dance starts on the next request; at 8 s the person steps in to 0.8 m: that is
    # inside the dance's 0.9 m trigger, so it steps back.
    DCase("dance_trigger_vs_walk_trigger", 11.0, CS.const(ZERO),
          CS.step_world((0.0, CS.scene(person_m=0.7)), (4.0, CS.scene(person_m=1.05)),
                        (8.0, CS.scene(person_m=0.8))),
          heartbeat((2.0, 3.0), (5.0, 10.9)),
          [(1.5, W, OFF, A, 0.0, None), (2.0, W, OFF, A, 0.0, CD.D_REFUSED_TOO_CLOSE),
           (5.0, D, ON, A, 0.0, CD.D_NONE), (7.9, D, ON, A, 0.0, None),
           (8.4, W, OFF, R, -0.3, CD.D_STOP_RETREAT)]),

    # Walking until 2 s; asked to dance at 2.5 s: refused, not standing still for 1 s yet.
    # The mission keeps asking; the first request after 3.0 s (3.1 s) starts it.
    DCase("dance_not_still", 5.0, CS.step_cmd((0.0, FWD3), (2.0, ZERO)), CS.const(CS.scene()),
          heartbeat((2.5, 4.9)),
          [(2.5, W, OFF, A, 0.0, CD.D_REFUSED_NOT_STILL), (2.9, W, OFF, A, 0.0, CD.D_REFUSED_NOT_STILL),
           (3.1, D, ON, A, 0.0, CD.D_NONE)]),

    # The dance policy's output goes stale at 4 s: the walking policy takes over at once
    # (standing, no retreat). Asked again while it is broken: refused, no dance policy --
    # once the old heartbeat has lapsed (it does at 5 s: none between 4.9 and 6.0).
    DCase("dance_policy_fault", 8.0, CS.const(ZERO), CS.const(CS.scene()),
          heartbeat((1.5, 4.9), (6.0, 7.9)),
          [(3.98, D, ON, A, 0.0, None), (4.0, W, OFF, A, 0.0, CD.D_STOP_FAULT),
           (6.0, W, OFF, A, 0.0, CD.D_REFUSED_NO_POLICY)],
          dance_ok=lambda t: t < 4.0, never_dance_after=4.0),

    # The camera stops at 5 s: ending (camera stale), handed back at tick 300 = 7.5 s.
    DCase("dance_camera_lost", 9.0, CS.const(ZERO),
          CS.step_world((0.0, CS.scene()), (5.0, None)), heartbeat((1.5, 8.9)),
          [(5.6, D, END, A, 0.0, CD.D_END_STALE), (7.5, W, OFF, A, 0.0, CD.D_DONE)],
          never_dance_after=7.5),

    # Danced 63 s (the bridge's cap) even though the heartbeat goes on: tick 3150 is also
    # a bar boundary, so it ends there, at 64 s. Nothing restarts while asked on.
    DCase("dance_max_length", 70.0, CS.const(ZERO), CS.const(CS.scene()), heartbeat((1.0, 69.9)),
          [(1.0, D, ON, A, 0.0, CD.D_NONE), (63.98, D, ON, A, 0.0, None),
           (64.0, W, OFF, A, 0.0, CD.D_DONE), (69.0, W, OFF, A, 0.0, CD.D_DONE)],
          never_dance_after=64.0),

    # The bridge leaves RUNNING at 5 s: the dance stops at once.
    DCase("dance_not_running", 7.0, CS.const(ZERO), CS.const(CS.scene()), heartbeat((1.5, 6.9)),
          [(4.98, D, ON, A, 0.0, None), (5.0, W, OFF, RX.OFF, 0.0, CD.D_STOP_OFF)],
          running=lambda t: t < 5.0, never_dance_after=5.0),

    # No dance without the reflex.
    DCase("dance_reflex_disabled", 3.0, CS.const(ZERO), CS.const(CS.scene()), heartbeat((1.5, 2.9)),
          [(1.5, W, OFF, RX.OFF, 0.0, CD.D_REFUSED_DISABLED)], cfg={"enable": False},
          never_dance_after=0.0),

    # A box pushed toward the dancing robot (obstacle 1.5 m -> 0.85 m at 3 s): steps back.
    DCase("dance_obstacle", 6.0, CS.const(ZERO),
          CS.step_world((0.0, CS.scene(obstacle_m=1.5)), (3.0, CS.scene(obstacle_m=0.85))),
          heartbeat((1.5, 5.9)),
          [(2.9, D, ON, A, 0.0, None), (3.3, W, OFF, R, -0.3, CD.D_STOP_RETREAT)],
          never_dance_after=3.3),

    # The whole loop: steps back from a person at 4 s; the person leaves; the mission
    # resets the reflex at 8 s (clear) and, after its heartbeat has lapsed, asks again.
    DCase("dance_after_reset", 12.0, CS.const(ZERO),
          CS.step_world((0.0, CS.scene()), (3.5, CS.scene(person_m=0.7)), (6.0, CS.scene(person_m=2.0))),
          heartbeat((1.5, 5.0), (9.0, 11.9)),
          [(3.4, D, ON, A, 0.0, None), (3.8, W, OFF, R, -0.3, CD.D_STOP_RETREAT),
           (7.0, W, OFF, S, 0.0, CD.D_STOP_RETREAT), (8.5, W, OFF, A, 0.0, None),
           (9.0, D, ON, A, 0.0, CD.D_NONE)],
          resets=(8.0,)),
]


def run_case(case):
    cd = CD.Conductor(case.cfg)
    rows = []
    for k, t_rel, t, cmd, frame, reset, running in case.ticks():
        dance, ok = case.dance(t_rel), bool(case.dance_ok(t_rel))
        out = cd.step(t, cmd, frame, reset, running, dance, ok)
        rows.append((k, t_rel, t, cmd, frame, reset, running, dance, ok, out))
    return rows


_f = CS._f
HEADER = CS.HEADER.replace("out_state,", "dance,dance_ok,out_state,") + \
    ",policy,dance_tick,dance_state,dance_reason"


def golden_text():
    lines = ["# V3 conductor golden cases -- generated by v3/reflex/dance_cases.py from conductor.py",
             "# and reflex.py; do not edit. nan = none. dance: nan = no message, 1 = dance, 0 = end.",
             "# states: " + " ".join("{}={}".format(i, s) for i, s in enumerate(RX.STATES)),
             "# policy: 0=walk 1=dance; dance_state: " + " ".join(
                 "{}={}".format(i, s) for i, s in enumerate(CD.DANCE_STATES)),
             "# defaults: " + " ".join("{}={}".format(k, _f(v)) for k, v in RX.DEFAULTS.items()),
             "# dance defaults: " + " ".join(
                 "{}={}".format(k, "/".join(str(x) for x in v) if isinstance(v, tuple) else _f(v))
                 for k, v in CD.DANCE_DEFAULTS.items()),
             HEADER]
    for case in CASES:
        if case.cfg:
            lines.append("#cfg {} ".format(case.name) + " ".join(
                "{}={}".format(k, _f(v)) for k, v in sorted(case.cfg.items())))
        for k, t_rel, t, cmd, fr, reset, running, dance, ok, out in run_case(case):
            fcols = [_f(None)] * 7 if fr is None else [
                _f(fr.seq), _f(fr.t_cap), _f(fr.person_m), _f(fr.person_near_box),
                _f(fr.obstacle_m), _f(fr.depth_ok), _f(fr.center_valid)]
            rx = out.reflex
            lines.append(",".join(
                [case.name, str(k), _f(t), _f(running), _f(reset)] + [_f(c) for c in cmd]
                + [_f(fr is not None)] + fcols + [_f(dance), _f(ok)]
                + [str(RX.STATES.index(rx.state)), _f(rx.vx), _f(rx.vy), _f(rx.wz),
                   str(rx.reason), _f(rx.blocked), _f(rx.reset_refused)]
                + [str(0 if out.policy == CD.WALK else 1), str(out.dance_tick),
                   str(CD.DANCE_STATES.index(out.dance_state)), str(out.dance_reason)]))
    return "\n".join(lines) + "\n"


def check_case(case):
    rows = run_case(case)
    bad = []
    for t_chk, policy, dstate, rstate, vx, dreason in case.dchecks:
        k = int(round(t_chk / DT))
        if k >= len(rows):
            bad.append("t={} beyond the case".format(t_chk))
            continue
        out = rows[k][-1]
        rx = out.reflex
        want = (policy, dstate, rstate, vx, dreason)
        got = (out.policy, out.dance_state, rx.state, round(rx.vx, 9), out.dance_reason)
        if (out.policy != policy or out.dance_state != dstate
                or (rstate is not None and rx.state != rstate)
                or (vx is not None and abs(rx.vx - vx) > 1e-9)
                or (dreason is not None and out.dance_reason != dreason)):
            bad.append("t={:.2f}: expected {}; got {} ({})".format(
                t_chk, want, got, CD.dance_reason_text(out.dance_reason)))
    if case.never_dance_after is not None:
        k0 = int(round(case.never_dance_after / DT))
        late = [r[1] for r in rows[k0:] if r[-1].policy == CD.DANCE]
        if late:
            bad.append("dancing at t={:.2f}, after {:.2f}".format(late[0], case.never_dance_after))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="regenerate golden_dance.csv")
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
            print("not writing golden_dance.csv: {} case(s) fail their checkpoints".format(failed))
            return 1
        text = golden_text()
        rows = sum(1 for ln in text.splitlines() if ln and not ln.startswith("#")) - 1
        GOLDEN.write_text(text)
        print("wrote {} ({} ticks, {} cases)".format(GOLDEN, rows, len(CASES)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
