#!/usr/bin/env python3
"""Stage E in the mission layer, without ROS, a camera or a robot: the object table,
the plan checks, grounding, and turn_to / walk_to executed against a simulated room
(percept.FakeWorld) -- including a person stepping in and a target that is not there.

  python3 mission_ctl/tests/test_objects.py
"""
from __future__ import annotations

import io
import json
import math
import os
import pathlib
import sys
from contextlib import redirect_stdout

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from r1_mission import executor as ex  # noqa: E402
from r1_mission import nl  # noqa: E402
from r1_mission import node as N  # noqa: E402
from r1_mission import objects as O  # noqa: E402
from r1_mission import percept as PC  # noqa: E402
from r1_mission import plan as P  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok    " + name)
    else:
        FAIL += 1
        print("  FAIL  {}  <- {}".format(name, detail))


def refuses(name, fn, needle):
    try:
        fn()
    except P.PlanError as exc:
        check(name, needle in str(exc), str(exc))
        return
    check(name, False, "no PlanError")


ENV = P.Envelope((0.0, 1.0), (0.0, 0.0), (-0.5, 0.5))
LIM = N.limits_from_cfg(N.read_flat_yaml(str(N.DEFAULT_CFG)))


def compile_(script, lim=LIM):
    return P.compile_plan(P.parse_script(script), ENV, lim)


def run(script, objects, lim=LIM, hz=50.0, ground=True, **wk):
    """compile -> ground on the world standing still -> execute in it."""
    prims, _, _ = compile_(script, lim)
    world = PC.FakeWorld(objects, **wk)
    why = None
    if ground:
        why, _ = N.ground_plan(prims, lim, N.world_frames(world))
    if why:
        return None, world, why
    e = ex.Executor(prims, lim)
    pc, next_f, t, dt = PC.Percept(), 0.0, 0.0, 1.0 / hz
    events = []
    while t < 400.0:
        while next_f <= t:
            pc.add(world.frame(next_f))
            next_f += 1.0 / N.FRAME_HZ
        out = e.step(t, yaw=world.yaw, bridge_state="RUNNING", percept=pc)
        if out.event:
            events.append(out.event)
        if out.state in (ex.DONE, ex.ABORTED):
            break
        world.move(out.vx, out.wz, dt)
        t += dt
    rep = e.report()
    rep["events"] = events
    rep["t_end"] = t
    return rep, world, None


def target_of(world, cls):
    o = next(o for o in world.objects if o["cls"] == cls)
    b, r, _, _ = world.relative(o)
    return b, r


print("== the object table ==")
for words, cls in (("the chair", "chair"), ("that sofa", "couch"), ("the dining table", "dining table"),
                   ("the table", "dining table"), ("my phone", "cell phone"),
                   ("the cup on the table", "cup"), ("the man in the blue shirt", "person"),
                   ("the two chairs", "chair"), ("a computer mouse", "mouse"),
                   ("the woman", "person"), ("the TV", "tv")):
    check("'{}' -> {}".format(words, cls), O.resolve(words)[0] == cls, O.resolve(words))
for words, why in (("the door", "not something I can recognize"),
                   ("the kitchen", "not something I can recognize"),
                   ("the chair or the sofa", "more than one kind"),
                   ("me", "not something"), ("the desk", "not something"),
                   ("the bag", "not something"), ("", "no object")):
    c, w = O.resolve(words)
    check("'{}' refused ({})".format(words, why), c is None and why in w, (c, w))
check("provenance: the words were said", O.said("the red chair", "walk to the red chair."))
check("provenance: a paraphrase was not", not O.said("the chair", "walk to the seat."))
check("provenance: plural and singular are the same word", O.said("the chairs", "face the chair"))

print("== plans ==")
pr, tot_s, tot_m = compile_("walk_to the chair stop_at 1.2; turn left 90")
check("walk_to + turn compiles; the class is the table's", pr[0].op == "walk_to"
      and pr[0].target_cls == "chair" and abs(pr[0].stop_at_m - 1.2) < 1e-9, pr)
check("an unsaid stop distance is the default and says so",
      compile_("walk_to chair")[0][0].stop_default
      and abs(compile_("walk_to chair")[0][0].stop_at_m - LIM.obj_stop_default_m) < 1e-9)
check("the typical estimate goes in the total, the budget is the timeout",
      pr[0].budget_s > pr[0].expected_s > 10.0 and tot_s < LIM.max_total_s, (pr[0].expected_s, pr[0].budget_s))
check("stand before an object step is fine", compile_("stand 2s; turn_to the bottle")[0][1].op == "turn_to")
refuses("a walk before an object step is refused", lambda: compile_("walk 2s; walk_to chair"),
        "must come before any walking or turning")
refuses("a turn before an object step is refused", lambda: compile_("turn left 90; turn_to chair"),
        "must come before")
refuses("two object steps are refused", lambda: compile_("turn_to chair; walk_to sofa"),
        "only one object step")
refuses("an unknown object is refused", lambda: compile_("walk_to the door"), "unknown object")
refuses("stopping inside the obstacle stop is refused", lambda: compile_("walk_to chair stop_at 0.5"),
        "too close")
refuses("a person: no nearer than the person stop + the arrival tolerance",
        lambda: compile_("walk_to person stop_at 1.3"), "too close")
check("a person at 1.5 m is fine", compile_("walk_to person stop_at 1.5")[0][0].stop_at_m == 1.5)
off = N.limits_from_cfg(dict(N.read_flat_yaml(str(N.DEFAULT_CFG)), perception="off"))
refuses("perception off: object steps refused", lambda: compile_("turn_to chair", off),
        "perception is off")
pj = P.plan_from_json({"plan": [{"op": "walk_to", "object": "the sofa", "stop_at_m": 1.5}]}, ENV, LIM)
check("the json front end takes object steps", pj[0][0].target_cls == "couch")
refuses("json: an object step needs an object", lambda: P.plan_from_json(
    {"plan": [{"op": "turn_to"}]}, ENV, LIM), "needs an 'object'")

print("== grounding (standing still, before [y/N]) ==")
room = [{"cls": "chair", "x": 3.0, "y": 1.0, "conf": 0.88},
        {"cls": "tv", "x": 4.0, "y": -0.5, "conf": 0.35}]
prims = compile_("walk_to the chair")[0]
why, lines = N.ground_plan(prims, LIM, N.world_frames(PC.FakeWorld(room)))
check("one confident chair: grounded, the plan says where and how far",
      why is None and prims[0].ground["cls"] == "chair" and "to my left" in lines[0]
      and "walk about" in lines[1], (why, lines))
why, _ = N.ground_plan(compile_("turn_to the tv")[0], LIM, N.world_frames(PC.FakeWorld(room)))
check("a tv at 0.35 is below the target threshold: refused, with the number",
      why and "not sure" in why and "0.35" in why, why)
two = room + [{"cls": "chair", "x": 2.0, "y": -1.5}]
why, _ = N.ground_plan(compile_("walk_to chair")[0], LIM, N.world_frames(PC.FakeWorld(two)))
check("two chairs: refused as ambiguous, with both bearings", why and "more than one" in why, why)
why, _ = N.ground_plan(compile_("walk_to chair")[0], LIM, N.world_frames(PC.FakeWorld(room, depth=False)))
check("no depth: walk_to refused", why and "no depth" in why, why)
why, _ = N.ground_plan(compile_("turn_to chair")[0], LIM, N.world_frames(PC.FakeWorld(room, depth=False)))
check("no depth: turn_to still works (it needs only the bearing)", why is None, why)
far = [{"cls": "chair", "x": 7.0, "y": 0.5}]
why, _ = N.ground_plan(compile_("walk_to chair")[0], LIM, N.world_frames(PC.FakeWorld(far)))
check("beyond the trusted depth range: refused", why and "only to" in why, why)
near = [{"cls": "chair", "x": 1.1, "y": 0.0}]
why, _ = N.ground_plan(compile_("walk_to chair")[0], LIM, N.world_frames(PC.FakeWorld(near)))
check("already at the stop distance: refused, nothing to walk", why and "already" in why, why)
why, _ = N.ground_plan(compile_("turn_to chair")[0], LIM, [])
check("no frames at all: refused, asks whether the node runs", why and "r1_perception" in why, why)

print("== turn_to, executed ==")
rep, w, _ = run("turn_to the chair", [{"cls": "chair", "x": 2.0, "y": 1.5}])
b, _ = target_of(w, "chair")
check("turn_to: DONE, facing within the tolerance by the camera and in truth",
      rep["final_state"] == "DONE" and abs(rep["executed"][0]["final_bearing_deg"]) <= LIM.obj_turn_tol_deg
      and abs(b) <= LIM.obj_turn_tol_deg, (rep["executed"][0], b))
check("turn_to: the turn was IMU-closed and about the bearing (36.9 deg)",
      abs(rep["executed"][0]["turned_deg"] - 36.87) < 2.0, rep["executed"][0])
rep, w, why = run("turn_to the chair", [{"cls": "chair", "x": 2.5, "y": -2.0}])
b, _ = target_of(w, "chair")
check("turn_to right, 39 deg: DONE and facing", rep is not None and rep["final_state"] == "DONE"
      and abs(b) <= 5.0, (why, rep and rep["abort_reason"], b))
_, _, why = run("turn_to the chair", [{"cls": "chair", "x": 1.0, "y": -2.5}])
check("a chair 68 deg right is outside the 100 deg view: refused before moving",
      why and "do not see" in why, why)
rep, w, _ = run("turn_to the chair", [{"cls": "chair", "x": 3.0, "y": 0.08}])
check("already facing (1.5 deg): no turn at all", rep["final_state"] == "DONE"
      and rep["executed"][0]["turns_deg"] == [], rep["executed"][0])
rep, w, _ = run("turn_to the chair", [{"cls": "chair", "x": 3.0, "y": 0.6}],
                bearing_noise_deg=1.0, seed=3)
b, _ = target_of(w, "chair")
check("a small correction (11 deg, under the 2 s floor's 17 deg) still turns, and ends facing",
      rep["final_state"] == "DONE" and abs(b) <= 5.0 and len(rep["executed"][0]["turns_deg"]) >= 1,
      (rep["executed"][0], b))

print("== walk_to, executed ==")
rep, w, _ = run("walk_to the chair stop_at 1.0", [{"cls": "chair", "x": 3.0, "y": 1.0}])
r0 = rep["executed"][0]
b, rng = target_of(w, "chair")
check("walk_to: DONE, stopped within the arrival tolerance of 1.0 m (truth)",
      rep["final_state"] == "DONE" and abs(rng - 1.0) <= LIM.obj_arrive_tol_m, (r0, rng))
check("walk_to: equal pieces, none over obj_segment_m",
      all(s["est_m"] <= LIM.obj_segment_m + 1e-6 for s in r0["segments"])
      and max(s["est_m"] for s in r0["segments"]) - min(s["est_m"] for s in r0["segments"]) < 0.05, r0)
check("walk_to: ends facing it", abs(b) <= LIM.obj_turn_tol_deg + 0.5, b)
check("walk_to: the report keeps measured and estimated apart",
      "open loop" in r0["walk_source"] and "depth camera" in r0["final_source"]
      and r0["turn_source"] == "imu_yaw", r0)
rep, w, _ = run("walk_to the chair stop_at 1.0", [{"cls": "chair", "x": 3.5, "y": -1.2}],
                bearing_noise_deg=1.5, range_noise_m=0.04, seed=7)
b, rng = target_of(w, "chair")
check("walk_to with camera noise (1.5 deg, 4 cm): DONE within 0.25 m of 1.0",
      rep["final_state"] == "DONE" and abs(rng - 1.0) <= 0.25, (rep["abort_reason"], rng))
rep, w, _ = run("walk_to the person stop_at 1.5", [{"cls": "person", "x": 4.0, "y": -1.0}])
_, rng = target_of(w, "person")
check("walk_to a person stops at 1.5 m and the guard never fires on that person",
      rep["final_state"] == "DONE" and abs(rng - 1.5) <= LIM.obj_arrive_tol_m, (rep["abort_reason"], rng))
rep, w, _ = run("walk_to the chair; turn left 90", [{"cls": "chair", "x": 3.0, "y": 0.0}])
check("steps after the object step run as usual",
      rep["final_state"] == "DONE" and rep["executed"][1]["op"] == "turn"
      and abs(rep["executed"][1]["achieved_deg"] - 90) < 3.5, rep["executed"])

print("== the safety layer ==")
rep, w, _ = run("walk_to the chair", [{"cls": "chair", "x": 4.0, "y": 0.0},
                                      {"cls": "person", "x": 1.6, "y": 0.1, "appear_t": 4.0}])
check("a person stepping in mid-walk: ABORTED at once, zero velocity, 'person' in the reason",
      rep["final_state"] == "ABORTED" and "person" in rep["abort_reason"], rep["abort_reason"])
check("... before the robot reached her (the person was 1.6 m out at the start)",
      math.hypot(1.6 - w.x, 0.1 - w.y) > 0.5, (w.x, w.y))
rep, w, _ = run("walk_to the chair", [{"cls": "chair", "x": 4.0, "y": 0.0},
                                      {"cls": "suitcase", "x": 1.4, "y": 0.0, "conf": 0.2}])
check("something low-confidence but in the corridor: stopped as an obstacle",
      rep["final_state"] == "ABORTED" and "obstacle" in rep["abort_reason"], rep["abort_reason"])


class Blind(PC.FakeWorld):
    """A camera that stops publishing after t_stop."""

    def __init__(self, objects, t_stop):
        PC.FakeWorld.__init__(self, objects)
        self.t_stop = t_stop
        self.last = None

    def frame(self, t):
        if t > self.t_stop and self.last is not None:
            return self.last            # the same frame again: seq does not advance
        self.last = PC.FakeWorld.frame(self, t)
        return self.last


prims = compile_("walk_to the chair")[0]
w = Blind([{"cls": "chair", "x": 4.0, "y": 0.0}], t_stop=3.0)
N.ground_plan(prims, LIM, N.world_frames(w))
e, pc, t, nf = ex.Executor(prims, LIM), PC.Percept(), 0.0, 0.0
while t < 60:
    while nf <= t:
        pc.add(w.frame(nf))
        nf += 1.0 / N.FRAME_HZ
    out = e.step(t, yaw=w.yaw, bridge_state="RUNNING", percept=pc)
    if out.state in (ex.DONE, ex.ABORTED):
        break
    w.move(out.vx, out.wz, 0.02)
    t += 0.02
check("the camera going quiet mid-walk stops the walk within obj_stale_s",
      e.state == ex.ABORTED and "stale" in e.abort_reason and t < 3.0 + LIM.obj_stale_s + 0.2,
      (e.abort_reason, t))
rep, _, _ = run("walk_to the chair", [{"cls": "chair", "x": 3.0, "y": 0.0, "id": 1}], ground=True)
check("guard mode 'object' does not touch plain walks without a camera",
      ex.Executor(compile_("walk 3s@0.3")[0], LIM).step(0.0, bridge_state="RUNNING").state == ex.RUNNING)
allg = N.limits_from_cfg(dict(N.read_flat_yaml(str(N.DEFAULT_CFG)), perception_guard="all"))
e = ex.Executor(compile_("walk 3s@0.3", allg)[0], allg)
e.step(0.0, bridge_state="RUNNING")
out = e.step(0.1, bridge_state="RUNNING", percept=None)
check("guard mode 'all': a plain walk without camera frames stops", out.state == ex.ABORTED
      and "camera" in (e.abort_reason or ""), e.abort_reason)

print("== English ==")


def understand(text, steps):
    u = nl.Understood(text)
    u.transcript = {"steps": steps}
    return nl.judge(u, ENV, LIM)


u = understand("Walk to the red chair and stop 1.5 meters from it.",
               [{"op": "walk_to", "object": "the red chair", "stop_at_m": 1.5}])
check("walk_to with a said stop distance executes",
      u.outcome == "execute" and u.prims[0].target_cls == "chair" and u.prims[0].stop_at_m == 1.5, u.to_dict())
u = understand("Walk to the red chair.", [{"op": "walk_to", "object": "the red chair", "stop_at_m": 1.5}])
check("a stop distance nobody said is dropped -> the default",
      u.outcome == "execute" and u.prims[0].stop_default and u.invented, u.to_dict())
u = understand("Walk to the seat.", [{"op": "walk_to", "object": "the chair"}])
check("a paraphrased object (seat -> chair) is refused as unknown_object",
      u.outcome == "refuse" and u.reasons == ["unknown_object"], u.to_dict())
u = understand("Walk to the door.", [{"op": "walk_to", "object": "the door"}])
check("the door: unknown_object, and the message lists what I can find",
      u.reasons == ["unknown_object"] and "chair" in u.refusals[0]["message"], u.refusals)
u = understand("Walk forward 2 meters and go to the chair.",
               [{"op": "walk", "direction": "forward", "distance_m": 2},
                {"op": "walk_to", "object": "the chair"}])
check("an object after a walk: object_order", u.reasons == ["object_order"], u.to_dict())
u = understand("Face the chair and walk to the sofa.",
               [{"op": "turn_to", "object": "the chair"}, {"op": "walk_to", "object": "the sofa"}])
check("two objects: object_order", u.reasons == ["object_order"], u.to_dict())
u = understand("Walk to the person and stop 1 meter from them.",
               [{"op": "walk_to", "object": "the person", "stop_at_m": 1}])
check("a person 1 m away: refused by the limits, nothing runs",
      u.outcome == "refuse" and u.reasons == ["limits"], u.to_dict())
txt = nl.render_report({"executed": [
    {"op": "walk_to", "target": "chair", "segments": [{"vx": 0.3, "s": 2.4, "est_m": 0.72}] * 3,
     "walked_est_m": 2.16, "final_range_m": 1.01, "stop_at_m": 1.0, "stop_default": True}],
    "final_state": "DONE"}, compile_("walk_to chair")[0])
check("the reply: pieces, estimated walk, measured stop, default named",
      "3 pieces" in txt and "about 2.2 m" in txt and "1.01 m" in txt and "the default" in txt, txt)
txt = nl.render_report({"executed": [{"op": "turn_to", "target": "chair", "turned_deg": -40.2,
                                      "final_bearing_deg": 1.5}], "final_state": "DONE"},
                       compile_("turn_to chair")[0])
check("the turn_to reply: IMU angle and camera bearing", "Turned right 40.2" in txt
      and "1.5° to my left" in txt, txt)

print("== CLI dry run with a world ==")
world = HERE / "worlds" / "room.json"
buf = io.StringIO()
with redirect_stdout(buf):
    code = N.main(["run", "walk_to the chair stop_at 1.2", "--dry-run", "--world", str(world)])
out = buf.getvalue()
check("run --dry-run --world: grounded, executed, DONE", code == 0 and "target:" in out
      and '"final_state": "DONE"' in out, out[-600:])
buf = io.StringIO()
with redirect_stdout(buf):
    code = N.main(["run", "turn_to the chair", "--dry-run"])
check("an object step in a dry run without --world is refused, not faked",
      code == 2 and "--world" in buf.getvalue(), buf.getvalue()[-300:])
buf = io.StringIO()
with redirect_stdout(buf):
    code = N.main(["run", "turn_to the tv", "--dry-run", "--world", str(world)])
check("the low-confidence tv in the room: refused before anything moves",
      code == 2 and "not sure" in buf.getvalue(), buf.getvalue()[-300:])
with open(str(world)) as f:
    check("the example room is valid JSON with objects", json.load(f)["objects"])

print("-" * 50)
print("pass {}  fail {}".format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
