#!/usr/bin/env python3
"""V3 in the mission layer, without ROS or a robot: reading the bridge's reflex messages
(recorded from the real C++ bridge), ending a plan when the reflex takes over or cuts a
forward walk, the speed-dependent camera guard, the reset before a new plan, and dry
runs in which the bridge's reflex is simulated (a person walking up, a box ahead).

  python3 v3/mission_ctl/tests/test_reflex.py
"""
from __future__ import annotations

import io
import pathlib
import sys
from contextlib import redirect_stdout

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from r1_mission import executor as ex  # noqa: E402
from r1_mission import nl  # noqa: E402
from r1_mission import node as N  # noqa: E402
from r1_mission import percept as PC  # noqa: E402
from r1_mission import plan as P  # noqa: E402
from r1_mission import reflex_link as RL  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok    " + name)
    else:
        FAIL += 1
        print("  FAIL  {}  <- {}".format(name, detail))


CFG = N.read_flat_yaml(str(N.DEFAULT_CFG))
LIM = N.limits_from_cfg(CFG)
V2 = P.Envelope((-0.6, 1.0), (-0.5, 0.5), (-1.57, 1.57), source="v2")


def plan(script, lim=LIM):
    prims, _, _ = P.compile_plan(P.parse_script(script), V2, lim)
    return prims


def status(state, reason=0, blocked=False, cmd_in=(0, 0, 0), cmd_out=None, **kw):
    return RL.ReflexStatus(state, reason, kw.pop("text", ""), blocked, kw.pop("refused", False),
                           cmd_in, cmd_in if cmd_out is None else cmd_out, **kw)


def run(prims, feed, lim=LIM, until=20.0, hz=50.0, percept=None):
    """Step an executor; feed(t) -> ReflexStatus or None. Returns (executor, last Output)."""
    e = ex.Executor(prims, lim)
    t, out = 0.0, None
    while t < until:
        out = e.step(t, yaw=0.0, bridge_state="RUNNING", percept=percept(t) if percept else None,
                     reflex=feed(t))
        if out.state in (ex.DONE, ex.ABORTED):
            break
        t += 1.0 / hz
    return e, out


print("== the config ==")
check("V3's mission.yaml follows the reflex, for people only (R1, 10-23)",
      LIM.reflex is True and LIM.reflex_obstacles is False, (LIM.reflex, LIM.reflex_obstacles))
OBST = N.limits_from_cfg(dict(CFG, reflex_obstacles="on"))
check("walk_to's minimum stop is above the obstacle stop distance at obj_vx",
      LIM.obj_stop_min_m == 0.9 and RL.reference()[1].stop_distance("obstacle", LIM.obj_vx) < 0.9)
off = N.limits_from_cfg(dict(CFG, reflex="off"))
check("reflex: off is read", off.reflex is False)

print("== the bridge's messages (recorded from the C++ bridge) ==")
msgs = [RL.ReflexStatus.from_json(line) for line in
        (HERE / "reflex_messages.jsonl").read_text().splitlines() if line.strip()]
kinds = {(m.state, m.reason, m.blocked) for m in msgs}
check("the recording has clear, blocked, retreat and stand",
      {("ARMED", 0, False), ("ARMED", 7, True), ("RETREAT", 9, False), ("STAND", 10, False)} <= kinds,
      sorted(kinds))
retreat = next(m for m in msgs if m.state == "RETREAT")
check("a retreat message: obstacle distance and the commands in and out",
      retreat.obstacle_m == 0.4 and retreat.cmd_out[0] == -0.3 and retreat.retreats == 1,
      retreat.to_dict())
blocked = next(m for m in msgs if m.reason == 7)
check("a block by an obstacle cuts forward: cmd_in 0.3 -> cmd_out 0",
      blocked.forward_clamped and blocked.cmd_in[0] == 0.3 and blocked.cmd_out[0] == 0.0,
      blocked.to_dict())
check("the cause says what and how far", retreat.cause() == "an obstacle 0.40 m ahead"
      and blocked.cause() == "an obstacle 0.80 m ahead", (retreat.cause(), blocked.cause()))
stale = next(m for m in msgs if m.reason == 3)
check("blocked for stale frames while standing is not a forward cut", not stale.forward_clamped)

print("== the executor ==")
walk = plan("walk 4s@0.3")
e, out = run(walk, lambda t: status("ARMED") if t < 1.0 else status(
    "RETREAT", 9, cmd_in=(0.3, 0, 0), cmd_out=(-0.3, 0, 0), obstacle_m=0.4))
rep = e.report()
check("a retreat ends the walk: ABORTED, the cause in the reason",
      out.state == ex.ABORTED and rep["abort_reason"].startswith(
          "stepped back because of an obstacle 0.40 m ahead"), rep["abort_reason"])
check("...and the report carries the bridge's status", rep.get("reflex", {}).get("state") == "RETREAT")
check("...the walk's record says how long it really walked",
      abs(rep["executed"][0]["actual_s"] - 1.0) < 0.03, rep["executed"][0])
text = nl.render_report(rep, walk)
check("the reply says it stepped back and stands until a new instruction",
      "stepped back because of an obstacle 0.40 m ahead -- standing still until a new instruction"
      in text, text)

e, out = run(walk, lambda t: status("STAND", 10, person_m=0.5))
check("a plan does not start while the reflex holds the robot standing",
      out.state == ex.ABORTED and e.report()["abort_reason"].startswith("not started:")
      and not e.report()["executed"], e.report())

e, out = run(walk, lambda t: status("ARMED") if t < 2.0 else status(
    "ARMED", 6, True, cmd_in=(0.3, 0, 0), cmd_out=(0.0, 0, 0), person_m=1.0))
check("the bridge cutting a forward walk ends it, with the cause",
      out.state == ex.ABORTED and e.report()["abort_reason"] ==
      "forward walking blocked by the bridge: a person 1.00 m away inside the stop distance",
      e.report()["abort_reason"])
check("...0.3 s after the cut, and the walk's time stops at the cut, not at the abort",
      abs(e.report()["executed"][0]["actual_s"] - 2.0) < 0.03 and 2.29 < e._t_last < 2.33,
      (e.report()["executed"][0]["actual_s"], e._t_last))
e, out = run(walk, lambda t: status("ARMED") if t < 2.0 else status(
    "ARMED", 7, True, cmd_in=(0.3, 0, 0), cmd_out=(0.0, 0, 0), obstacle_m=0.40) if t < 2.1
    else status("RETREAT", 9, cmd_in=(0.3, 0, 0), cmd_out=(-0.3, 0, 0), obstacle_m=0.40))
check("a block followed by a retreat within 0.3 s is reported as the retreat",
      e.report()["abort_reason"].startswith("stepped back because of an obstacle 0.40 m")
      and abs(e.report()["executed"][0]["actual_s"] - 2.0) < 0.03, e.report())
e, out = run(walk, lambda t: status("ARMED", 7, True, cmd_in=(0.3, 0, 0), cmd_out=(0.0, 0, 0),
                                    obstacle_m=0.8) if 1.0 <= t < 1.2 else status("ARMED"))
check("a cut shorter than 0.3 s does not end the walk", out.state == ex.DONE, e.report())

back = plan("walk backward 2s")
e, out = run(back, lambda t: status("ARMED", 6, True, cmd_in=(-0.3, 0, 0), person_m=1.0))
check("a backward walk the bridge lets through (blocked flag, nothing cut) runs to the end",
      out.state == ex.DONE, e.report())
turn = plan("turn left 2s")
e, out = run(turn, lambda t: status("ARMED", 3, True))
check("a turn while forward is blocked for stale frames runs to the end", out.state == ex.DONE)
e, out = run(walk, lambda t: None)
check("no reflex messages (reflex off in the bridge): the walk runs as before",
      out.state == ex.DONE)
e, out = run(walk, lambda t: status("ARMED") if t < 3.0 else status("STAND", 11, person_m=0.5))
check("the limit: the reason says it needs help",
      "stepped back too often" in e.report()["abort_reason"], e.report()["abort_reason"])

print("== the camera guard, by stop distance ==")
allg = N.limits_from_cfg(dict(CFG, perception_guard="all"))


def frames(obstacle_m=None, person_m=None):
    def f(t):
        pc = PC.Percept()
        pc.add(PC.Frame(t - 0.05, int(t * 100) + 1, [], {}, {
            "person_m": person_m, "person_near_box": False, "obstacle_m": obstacle_m,
            "depth_ok": True, "center_valid": 1.0}))
        return pc
    return f


w3 = plan("walk 3s@0.3", allg)
e, out = run(w3, lambda t: None, allg, percept=frames(obstacle_m=0.80))
check("walking at 0.3 m/s, an obstacle at 0.80 m is inside the 0.87 m stop distance",
      out.state == ex.ABORTED and "inside the 0.87 m stop distance at 0.30 m/s"
      in e.report()["abort_reason"], e.report()["abort_reason"])
e, out = run(w3, lambda t: None, allg, percept=frames(obstacle_m=0.95))
check("...at 0.95 m it is not", out.state == ex.DONE, e.report())
w8 = plan("walk 3s@0.8", allg)
e, out = run(w8, lambda t: None, allg, percept=frames(person_m=1.3))
check("walking at 0.8 m/s, a person at 1.3 m is inside the 1.49 m stop distance",
      out.state == ex.ABORTED and "1.49 m stop distance" in e.report()["abort_reason"],
      e.report()["abort_reason"])
e, out = run(plan("walk 3s@0.3", allg), lambda t: None, allg, percept=frames(person_m=1.3))
check("...at 0.3 m/s the 1.2 m floor applies and 1.3 m is clear", out.state == ex.DONE)

print("== the reset before a new plan ==")


class FakeBridge(object):
    """latest/send_reset/spin for reset_before_start: answers a reset after one spin."""

    def __init__(self, st, answer):
        self.st, self.answer, self.resets, self.t = st, answer, 0, 0.0

    def latest(self):
        return self.st

    def send(self):
        self.resets += 1

    def spin(self, dt):
        self.t += dt
        if self.resets and self.answer is not None:
            self.st = self.answer

    def clock(self):
        return self.t


def reset(st, answer):
    b = FakeBridge(st, answer)
    ok, msg = RL.reset_before_start(b.latest, b.send, b.spin, timeout_s=1.0, clock=b.clock)
    return ok, msg, b.resets


ok, msg, n = reset(None, None)
check("no reflex (off, or an older bridge): start, no reset sent", ok and n == 0)
ok, msg, n = reset(status("ARMED"), None)
check("ARMED: start, no reset sent", ok and n == 0 and msg is None)
ok, msg, n = reset(status("STAND", 10, person_m=1.5), status("ARMED"))
check("STAND, the way clear: one reset, the bridge lets go, start", ok and n == 1, msg)
ok, msg, n = reset(status("STAND", 10, person_m=0.7),
                   status("STAND", 12, refused=True, person_m=0.7))
check("STAND, still too close: refused, not started, the operator told why",
      not ok and n == 1 and "still too close: a person 0.70 m away" in msg, msg)
ok, msg, n = reset(status("STAND", 10), None)
check("no answer from the bridge: not started", not ok and "no answer" in msg, msg)
ok, msg, n = reset(status("RETREAT", 8, person_m=0.5), None)
check("in the middle of a retreat: not started, no reset sent", not ok and n == 0, msg)

print("== dry runs with the bridge's reflex simulated ==")


def dry(script, world_file, lim=LIM):
    world = N.load_world(str(HERE / "worlds" / world_file))
    res = {}
    with redirect_stdout(io.StringIO()) as buf:
        code = N.dry_run(plan(script, lim), lim, 50.0, result=res, world=world)
    return code, res, buf.getvalue()


code, res, trace = dry("stand 9s", "approach.json")
rep = res["report"]
check("a person walks up to the standing robot: it steps back, the plan ends",
      code == 1 and rep["final_state"] == ex.ABORTED
      and rep["abort_reason"].startswith("stepped back because of a person"), rep["abort_reason"])
check("...the robot moved back 1.5 s x 0.3 m/s (the dry run's robot does exactly what it is told)",
      abs(res["robot_xy"][0] + 0.45) < 0.03, res["robot_xy"])
check("...and the simulated bridge ends standing", res["reflex"]["state"] == "STAND", res["reflex"])
check("the trace shows the reflex's transitions", "reflex ARMED -> RETREAT" in trace
      and "reflex RETREAT -> STAND" in trace, trace[-600:])

code, res, trace = dry("walk 6s@0.3", "box_ahead.json", OBST)
rep = res["report"]
check("(obstacles on) walking at 0.3 m/s toward a box 2 m ahead: the bridge blocks it, the plan ends",
      rep["final_state"] == ex.ABORTED and rep["abort_reason"].startswith(
          "forward walking blocked by the bridge: an obstacle 0.8"), rep["abort_reason"])
check("...the robot stopped about 0.87 m from the box, without a retreat",
      1.1 < res["robot_xy"][0] < 1.16 and "RETREAT" not in trace, res["robot_xy"])
code, res, _ = dry("walk 6s@0.3", "box_ahead.json", off)
check("the same with reflex: off: no simulated bridge, the walk runs into the box",
      res["report"]["final_state"] == ex.DONE and res["robot_xy"][0] > 1.6, res["robot_xy"])
code, res, _ = dry("walk 6s@0.3", "box_ahead.json")
check("the same with obstacles off (V3 now): the box is not seen, the walk runs to the end",
      res["report"]["final_state"] == ex.DONE and res["robot_xy"][0] > 1.6, res["robot_xy"])
code, res, _ = dry("walk_to the chair stop_at 1.2; turn left 90", "room.json")
check("walk_to the chair still arrives with the reflex simulated",
      res["report"]["final_state"] == ex.DONE, res["report"].get("abort_reason"))

print("== the dance (V3): plan, LLM step, executor, the simulated bridge ==")
pd = plan("dance")
check("'dance' alone is the default {:.0f} s; its total adds up to a bar to end on".format(LIM.dance_default_s),
      len(pd) == 1 and pd[0].op == "dance" and pd[0].duration_s == LIM.dance_default_s
      and pd[0].expected_s == LIM.dance_default_s + LIM.dance_end_s, pd)
for script, why in (("dance 3s", "shorter than the shortest"), ("dance 90s", "longer than the longest")):
    try:
        plan(script)
        check("'{}' is refused".format(script), False, "compiled")
    except P.PlanError as exc:
        check("'{}' is refused ({})".format(script, why), why in str(exc), str(exc))
for key in ("dance", "reflex"):
    lim_x = N.limits_from_cfg(dict(CFG, **{key: "off"}))
    try:
        plan("dance 10s", lim_x)
        check("no dance with {}: off".format(key), False, "compiled")
    except P.PlanError as exc:
        check("no dance with {}: off".format(key), key in str(exc), str(exc))
check("a dance counts as moving: no object step after it",
      "dance" in P.MOVING_OPS and "dance" not in P.OBJECT_OPS)
check("the json front end needs a dance's duration (the LLM path fills in the default)",
      P.plan_from_json({"plan": [{"op": "dance", "duration_s": 12}]}, V2, LIM)[0][0].duration_s == 12)


def judged(text, steps, lim=LIM):
    u = nl.Understood(text)
    u.transcript = {"steps": steps}
    return nl.judge(u, V2, lim)


u = judged("dance", [{"op": "dance"}])
check("the LLM's bare dance -> the default length, said in the plan",
      u.outcome == "execute" and u.prims[0].duration_s == LIM.dance_default_s
      and "you did not say how long" in u.prims[0].text, (u.outcome, u.refusals))
u = judged("dance for 2 minutes", [{"op": "dance", "duration_s": 120}])
check("120 s is refused as too long, with the limit in the message",
      u.outcome == "refuse" and u.reasons == ["too_long"] and "60 s" in u.refusals[0]["message"],
      u.refusals)
u = judged("dance for 3 seconds", [{"op": "dance", "duration_s": 3}])
check("3 s is refused as too small", u.outcome == "refuse" and u.reasons == ["too_small"], u.refusals)
u = judged("dance", [{"op": "dance"}], N.limits_from_cfg(dict(CFG, dance="off")))
check("dance: off -> refused, saying it is switched off",
      u.reasons == ["no_dance"] and "switched off" in u.refusals[0]["message"], u.refusals)
u = judged("sing and dance", [{"op": "unsupported", "what": "sing"}, {"op": "dance"}])
check("all or nothing: an unsupported step refuses the dance with it", u.outcome == "refuse", u.outcome)
u = judged("dance while walking forward 2 meters",
           [{"op": "dance"}, {"op": "walk", "direction": "forward", "distance_m": 2}])
check("'dance while walking' written as a dance and a walk: refused, one thing at a time (G28)",
      u.outcome == "refuse" and u.reasons == ["simultaneous"], u.refusals)
u = judged("turn left 45 degrees and walk 1 meter at the same time",
           [{"op": "turn", "direction": "left", "angle_deg": 45},
            {"op": "walk", "direction": "forward", "distance_m": 1}])
check("...the same for any two moving steps 'at the same time'", u.reasons == ["simultaneous"], u.refusals)
u = judged("dance for 20 seconds while i watch", [{"op": "dance", "duration_s": 20}])
check("...but 'while' with one moving step is not two actions", u.outcome == "execute", u.refusals)
u = judged("dance for 10 seconds, then walk forward 1 meter",
           [{"op": "dance", "duration_s": 10}, {"op": "walk", "direction": "forward", "distance_m": 1}])
check("...and 'then' is a sequence, which runs", u.outcome == "execute", u.refusals)
check("'stop dancing' and 'quit dancing now' never reach the model (the fast stop)",
      nl.FAST_STOP.match("stop dancing") and nl.FAST_STOP.match("quit dancing now.")
      and not nl.FAST_STOP.match("stop dancing and walk"))

# The executor against scripted bridge statuses
DANCING = dict(policy="dance", dance_state="DANCING", dance_tick=10)


def dstatus(policy="walk", reason=0, text="", state="ARMED", dance_state="OFF"):
    return status(state, policy=policy, dance_reason=reason, dance_text=text, dance_state=dance_state)


e, out = run(plan("dance 6s"), lambda t: dstatus(reason=4, text="dance refused: someone or something is too close"),
             until=5.0)
check("the bridge refuses (someone within 1 m): the plan ends after the start window, saying why",
      e.state == ex.ABORTED and "too close" in e.abort_reason
      and e.records[0].actual_s <= LIM.dance_start_s + 0.6, (e.abort_reason, e.records[0].actual_s))
check("...and the last word to the bridge is dance=False", out.dance is False, out.dance)
txt = nl.render_report(e.report(), plan("dance 6s"))
check("...the reply (robot 10-24: an IndexError here) says it did not dance, and why",
      txt.startswith("1. Did not dance. Interrupted: the bridge did not dance: dance refused: "
                     "someone or something is too close.") and e.report()["executed"][0]["danced_s"] is None, txt)

e, out = run(plan("dance 6s"), lambda t: None, until=2.0)
check("no reflex status at all: no dance (it needs the V3 bridge with its reflex)",
      e.state == ex.ABORTED and "reflex" in e.abort_reason, e.abort_reason)


def bridge_script(t):
    if t < 0.3:
        return dstatus()
    if t < 6.6:
        return dstatus("dance", dance_state="DANCING")
    return dstatus(reason=RL.D_DONE, text="danced; back to standing")


outs = []
e = ex.Executor(plan("dance 6s; stand 2s"), LIM)
t = 0.0
while t < 12.0 and e.state not in (ex.DONE, ex.ABORTED):
    outs.append((t, e.step(t, yaw=0.0, bridge_state="RUNNING", reflex=bridge_script(t))))
    t += 0.02
asks = [(tt, o.dance) for tt, o in outs if o.dance is not None]
first_false = next(tt for tt, d in asks if d is False)
check("dancing: dance=True every step (the heartbeat) until the 6 s are up, then dance=False",
      all(d is True for tt, d in asks if tt < first_false) and 6.2 < first_false < 6.4, first_false)
check("...the step ends when the bridge is back on the walking policy, and the plan goes on",
      e.state == ex.DONE and [r.prim.op for r in e.records] == ["dance", "stand"], e.report())
txt = nl.render_report(e.report(), plan("dance 6s; stand 2s"))
check("...the reply says how long it danced (robot 10-24: no line for the dance at all)",
      txt.startswith("1. Danced on the spot for 6.") and txt.endswith("Done."), txt)
check("...the walking command is zero throughout the dance",
      all(o.vx == 0 and o.vy == 0 and o.wz == 0 for tt, o in outs if tt < 6.6))
e2 = ex.Executor(plan("dance 6s"), LIM)
e2.step(0.0, yaw=0.0, bridge_state="RUNNING", reflex=dstatus())
e2.step(0.5, yaw=0.0, bridge_state="RUNNING", reflex=dstatus("dance", dance_state="DANCING"))
e2.interrupt("operator interrupt (Ctrl-C)", 1.0)
check("Ctrl-C while dancing: the last word to the bridge is dance=False",
      e2.interrupt_output().dance is False)

code, res, trace = dry("dance 8s", "empty.json")
check("dry run, nobody around: the simulated bridge dances (after 1 s standing) and hands back",
      code == 0 and res["report"]["final_state"] == ex.DONE and res["reflex"]["policy"] == "walk"
      and res["robot_xy"] == (0.0, 0.0), (res["report"], res["robot_xy"]))
world = N.load_world(str(HERE / "worlds" / "empty.json"))
res = {}
with redirect_stdout(io.StringIO()):
    N.dry_run(plan("dance 8s"), LIM, 10.0, result=res, world=world)
check("...and at the CLI's 10 Hz too (10-24: the bars ran 5x long, 'did not end within 4 s')",
      res["report"]["final_state"] == ex.DONE, res["report"])
code, res, trace = dry("dance 12s", "approach.json")
rep = res["report"]
check("dry run, a person walks up to the dancing robot: it steps back at once, the plan ends",
      rep["final_state"] == ex.ABORTED and rep["abort_reason"].startswith("stepped back because of a person")
      and abs(res["robot_xy"][0] + 0.45) < 0.03, (rep["abort_reason"], res["robot_xy"]))
check("...the simulated bridge stopped the dance because of the retreat",
      res["reflex"]["dance_reason"] == 11 and res["reflex"]["policy"] == "walk", res["reflex"])
txt = nl.render_report(rep, plan("dance 12s"))
check("...the reply (robot 10-24: an IndexError here) says it danced, then stepped back",
      txt.startswith("1. Danced on the spot for ") and "Interrupted: stepped back because of a person"
      in txt and 0 < rep["executed"][0]["danced_s"] < rep["executed"][0]["actual_s"], (txt, rep["executed"]))
code, res, _ = dry("dance 8s", "person_close.json")
check("dry run, a person 0.95 m away: no dance (1.0 m must be clear), the robot does not move",
      res["report"]["final_state"] == ex.ABORTED and "too close" in res["report"]["abort_reason"]
      and res["robot_xy"] == (0.0, 0.0), (res["report"]["abort_reason"], res["robot_xy"]))
code, res, _ = dry("walk 2s@0.3; dance 6s", "empty.json")
check("dry run, walk then dance: the dance waits until the robot has stood still for 1 s",
      res["report"]["final_state"] == ex.DONE, res["report"].get("abort_reason"))

print("== the capability statement ==")
cap = N.capability_text(V2, LIM)
check("it says what the reflex does -- people only, not objects -- and that the camera looks "
      "forward only", "I step back on my own" in cap and "arm's length" in cap
      and "not do this for objects" in cap and "cannot see behind me" in cap, cap)
check("...with obstacles on it says objects too", "0.45 m" in N.capability_text(V2, OBST))
check("...and nothing with reflex: off", "step back" not in N.capability_text(V2, off))
check("...and the dance, with its limits and the 0.9 m / 1 m distances",
      "I can dance on the spot" in cap and "0.9 m" in cap and "1 m" in cap)
check("...no dance with dance: off", "dance on the spot" not in
      N.capability_text(V2, N.limits_from_cfg(dict(CFG, dance="off"))))
check("the LLM's prompt is the one the eval was run with (V3 adds the dance step to it)",
      nl.prompt_id(LIM) == nl.EVALUATED_PROMPT_ID, nl.prompt_id(LIM))

print("--------------------------------------------------")
print("pass {}  fail {}".format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
