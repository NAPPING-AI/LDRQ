"""What the mission layer knows from the camera, and how it picks a target.

Pure Python, stdlib only: the executor gets a `Percept` as an argument, exactly like
the yaw and the bridge state, so turn_to / walk_to are tested without ROS, a camera
or a robot. node.py fills a Percept from /r1_perception/detections; the dry run and
the tests fill it from `FakeWorld`.

Times here are on the EXECUTOR's clock (node.py converts the camera's time.time()
stamps), so "frames captured after the robot stood still" is one comparison.

Message format: perception/r1_perception/pipeline.py. Bearing is LEFT-positive, like
the IMU yaw: turning by a bearing needs no sign flip.
"""
from __future__ import annotations

import math

CONFIRMED, NOT_SEEN, AMBIGUOUS, LOW_CONF, NO_DEPTH, STALE = (
    "confirmed", "not_seen", "ambiguous", "low_confidence", "no_depth", "stale")


class Frame(object):
    __slots__ = ("t_cap", "seq", "dets", "guard")

    def __init__(self, t_cap, seq, dets, guard):
        self.t_cap = float(t_cap)
        self.seq = int(seq)
        self.dets = list(dets or [])
        self.guard = dict(guard or {})


class Percept(object):
    """The last few frames, newest last."""

    def __init__(self, keep=30):
        self.frames = []
        self.keep = int(keep)
        self.t_recv = None

    def add(self, frame, t_recv=None):
        if self.frames and frame.seq <= self.frames[-1].seq:
            return                              # a duplicate or a restart: keep the newer
        self.frames.append(frame)
        if len(self.frames) > self.keep:
            del self.frames[0]
        self.t_recv = frame.t_cap if t_recv is None else float(t_recv)

    @property
    def latest(self):
        return self.frames[-1] if self.frames else None

    def age(self, t):
        """Seconds since the newest frame was captured; inf with none."""
        return float("inf") if not self.frames else max(0.0, t - self.frames[-1].t_cap)

    def since(self, t):
        return [f for f in self.frames if f.t_cap >= t]


def from_message(msg, offset=0.0):
    """A /r1_perception/detections dict -> Frame. `offset` maps the camera's time.time()
    to the executor clock: t_exec = t_cap - offset."""
    return Frame(float(msg["t_cap"]) - offset, msg.get("seq", 0), msg.get("dets"),
                 msg.get("guard"))


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def choose_target(frames, cls, lim, expect_bearing=None, gate_deg=None, need_depth=False):
    """Pick the one detection of `cls` the mission acts on, from frames captured after
    the robot stood still. Returns (status, det, detail).

    The rules, all from `lim`:
      * at least obj_confirm_frames such frames must exist (else STALE: wait longer);
      * on the newest, a candidate has conf >= obj_target_conf and hits >= the same
        count (the node's tracker: that many consecutive frames agreed);
      * with an expected bearing (a target already chosen), only candidates within
        gate_deg of it count -- this is how the SAME chair is followed between steps;
      * exactly one candidate, else NOT_SEEN / AMBIGUOUS; LOW_CONF when something of
        that class is there but below the target threshold;
      * need_depth: the candidate must have a range (walk_to stops on depth)."""
    n = int(lim.obj_confirm_frames)
    if len(frames) < n:
        return STALE, None, "{} of {} frames since standing still".format(len(frames), n)
    last = frames[-1]
    same = [d for d in last.dets if d.get("cls") == cls]
    if expect_bearing is not None:
        g = float(gate_deg if gate_deg is not None else lim.obj_gate_deg)
        same = [d for d in same if abs(wrap_deg(d["bearing_deg"] - expect_bearing)) <= g]
    good = [d for d in same if d.get("conf", 0.0) >= lim.obj_target_conf
            and int(d.get("hits", 0)) >= n]
    if not good:
        if same:
            best = max(same, key=lambda d: d.get("conf", 0.0))
            return LOW_CONF, best, "best {} at {:.2f} (needs {:.2f} in {} frames in a row)".format(
                cls, best.get("conf", 0.0), lim.obj_target_conf, n)
        return NOT_SEEN, None, "no {} in view{}".format(
            cls, "" if expect_bearing is None else " near {:+.0f} deg".format(expect_bearing))
    if len(good) > 1:
        return AMBIGUOUS, None, "{} {}s, at {}".format(
            len(good), cls, ", ".join("{:+.0f} deg".format(d["bearing_deg"])
                                      for d in sorted(good, key=lambda d: -d["bearing_deg"])))
    d = good[0]
    if need_depth and d.get("range_m") is None:
        return NO_DEPTH, d, "the {} has no depth reading".format(cls)
    return CONFIRMED, d, ""


def bearing_after_turn(bearing_deg, turned_deg):
    return wrap_deg(bearing_deg - turned_deg)


def after_walk(bearing_deg, range_m, walked_m):
    """Where a static object should appear after walking straight `walked_m`."""
    b = math.radians(bearing_deg)
    x, y = range_m * math.cos(b) - walked_m, range_m * math.sin(b)
    return math.degrees(math.atan2(y, x)), math.hypot(x, y)


# --------------------------------------------------------------- simulation
class FakeWorld(object):
    """A flat world of point objects and a robot that does exactly what it is told:
    the dry run and the tests use it in place of the camera.

    objects: [{"cls": "chair", "x": 3.0, "y": 1.0, "conf": 0.9, "id": 1}], meters, in the
    robot's starting frame (x forward, y left). A detection is produced for every object
    inside the half field of view and `max_range`, with that object's confidence;
    `hits` counts consecutive frames in view, as the node's tracker does. The guard
    reports a person inside person_stop_m and anything inside the corridor ahead."""

    def __init__(self, objects, hfov_deg=100.0, max_range=8.0, person_stop_m=1.2,
                 obstacle_stop_m=0.6, corridor_half_width_m=0.35, depth=True,
                 bearing_noise_deg=0.0, range_noise_m=0.0, seed=0):
        import random
        self.objects = [dict(o) for o in objects]
        for i, o in enumerate(self.objects):
            o.setdefault("id", i + 1)
            o.setdefault("conf", 0.9)
        self.half = hfov_deg / 2.0
        self.max_range = max_range
        self.person_stop_m = person_stop_m
        self.obstacle_stop_m = obstacle_stop_m
        self.half_width = corridor_half_width_m
        self.depth = depth
        self.bn, self.rn = bearing_noise_deg, range_noise_m
        self.rng = random.Random(seed)
        self.x = self.y = self.yaw = 0.0
        self.seq = 0
        self.hits = {}

    def move(self, vx, wz, dt, vy=0.0):
        self.yaw += wz * dt
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (vx * c - vy * s) * dt
        self.y += (vx * s + vy * c) * dt

    def relative(self, o):
        dx, dy = o["x"] - self.x, o["y"] - self.y
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        fx, fy = c * dx - s * dy, s * dx + c * dy
        return math.degrees(math.atan2(fy, fx)), math.hypot(fx, fy), fx, fy

    def frame(self, t):
        self.seq += 1
        dets, person_m, near, obstacle, obst_m = [], None, False, False, None
        for o in self.objects:
            b, r, fx, fy = self.relative(o)
            if o.get("appear_t") is not None and t < o["appear_t"]:
                continue
            if fx > 0 and abs(fy) <= self.half_width:
                obst_m = fx if obst_m is None else min(obst_m, fx)
                if fx < self.obstacle_stop_m:
                    obstacle = True
            if abs(b) > self.half or r > self.max_range or fx <= 0:
                self.hits[o["id"]] = 0
                continue
            self.hits[o["id"]] = self.hits.get(o["id"], 0) + 1
            b_meas = b + (self.rng.gauss(0, self.bn) if self.bn else 0.0)
            r_meas = r + (self.rng.gauss(0, self.rn) if self.rn else 0.0)
            d = {"id": o["id"], "cls": o["cls"], "conf": o["conf"], "bearing_deg": b_meas,
                 "range_m": r_meas if self.depth else None,
                 "depth_m": r_meas if self.depth else None,
                 "hits": self.hits[o["id"]], "confirmed": self.hits[o["id"]] >= 3,
                 "bbox": [0, 0, 1, 1]}
            dets.append(d)
            if o["cls"] == "person":
                person_m = r if person_m is None else min(person_m, r)
                near = near or r < self.person_stop_m
        guard = {"person_near": near, "person_m": person_m, "obstacle": obstacle,
                 "obstacle_m": obst_m, "depth_ok": self.depth}
        return Frame(t, self.seq, dets, guard)
