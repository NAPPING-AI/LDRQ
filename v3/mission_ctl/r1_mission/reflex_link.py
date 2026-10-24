"""V3: the mission layer's side of the bridge's proximity reflex.

The reflex runs in the bridge (r1_hw_bridge, reflex_fsm.hpp), not here: it steps back
when a person or an obstacle comes too close, blocks forward walking inside a
speed-dependent stop distance, and then holds the robot standing until a reset. The
mission layer
  * reads what it did from /r1_hw_bridge/reflex (ReflexStatus) and ends the plan
    when the reflex took over, so the report says what happened instead of counting
    seconds of walking that never happened;
  * sends /r1_hw_bridge/reflex_reset when the operator confirms a new plan, and does
    not start while the bridge refuses it (reset_before_start);
  * in a dry run, plays the bridge: SimReflex runs the reference state machine
    (v3/reflex/conductor.py around reflex.py) on what the simulated camera sees.

V3 dance: the same status carries which policy drives the robot (walk = v2, dance = the
dance policy) and the dance's state; the mission asks for a dance on DANCE_TOPIC
(std_msgs/Bool: True = dance, sent as a heartbeat; False = end it).

Stdlib only.
"""
from __future__ import annotations

import json
import pathlib
import sys

REFLEX_TOPIC = "/r1_hw_bridge/reflex"
RESET_TOPIC = "/r1_hw_bridge/reflex_reset"
DANCE_TOPIC = "/r1_hw_bridge/dance"
WALK_POLICY, DANCE_POLICY = "walk", "dance"
# conductor.py's dance reasons the mission tells apart
D_DONE = 14
D_REFUSALS = (1, 2, 3, 4, 5, 6)

OFF, ARMED, RETREAT, STAND = "OFF", "ARMED", "RETREAT", "STAND"
TAKEN_OVER = (RETREAT, STAND)


def _v3_reflex_dir():
    # v3/mission_ctl/r1_mission/reflex_link.py -> v3/reflex
    return pathlib.Path(__file__).resolve().parents[2] / "reflex"


def reference():
    """The reference state machine and stop distance (v3/reflex), imported by path."""
    d = str(_v3_reflex_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    import reflex as RX              # noqa: E402
    import stop_distance as SD       # noqa: E402
    return RX, SD


def conductor():
    """The reference conductor (v3/reflex/conductor.py), imported by path."""
    reference()
    import conductor as CD           # noqa: E402
    return CD


class ReflexStatus(object):
    """One /r1_hw_bridge/reflex message (the bridge's PublishReflex JSON)."""
    __slots__ = ("t", "state", "reason", "text", "blocked", "reset_refused", "cmd_in",
                 "cmd_out", "person_m", "person_near_box", "obstacle_m", "frame_age_s",
                 "retreats", "policy", "dance_state", "dance_reason", "dance_tick", "dance_text")

    def __init__(self, state, reason=0, text="", blocked=False, reset_refused=False,
                 cmd_in=(0.0, 0.0, 0.0), cmd_out=(0.0, 0.0, 0.0), person_m=None,
                 person_near_box=False, obstacle_m=None, frame_age_s=None, retreats=0, t=None,
                 policy=WALK_POLICY, dance_state="OFF", dance_reason=0, dance_tick=-1,
                 dance_text=""):
        self.t = t
        self.state = str(state)
        self.reason = int(reason)
        self.text = str(text)
        self.blocked = bool(blocked)
        self.reset_refused = bool(reset_refused)
        self.cmd_in = tuple(float(x) for x in cmd_in)
        self.cmd_out = tuple(float(x) for x in cmd_out)
        self.person_m = person_m
        self.person_near_box = bool(person_near_box)
        self.obstacle_m = obstacle_m
        self.frame_age_s = frame_age_s
        self.retreats = int(retreats)
        self.policy = str(policy)
        self.dance_state = str(dance_state)
        self.dance_reason = int(dance_reason)
        self.dance_tick = int(dance_tick)
        self.dance_text = str(dance_text)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        return cls(d["state"], d.get("reason", 0), d.get("text", ""), d.get("blocked", False),
                   d.get("reset_refused", False), d.get("cmd_in", (0, 0, 0)),
                   d.get("cmd_out", (0, 0, 0)), d.get("person_m"), d.get("person_near_box", False),
                   d.get("obstacle_m"), d.get("frame_age_s"), d.get("retreats_in_window", 0),
                   d.get("t"), d.get("policy", WALK_POLICY), d.get("dance_state", "OFF"),
                   d.get("dance_reason", 0), d.get("dance_tick", -1), d.get("dance_text", ""))

    @property
    def forward_clamped(self):
        """The bridge is cutting a forward command right now."""
        return self.blocked and self.cmd_in[0] > 1e-6 and self.cmd_out[0] < self.cmd_in[0] - 1e-6

    def cause(self):
        """'a person 0.55 m ahead' / 'an obstacle 0.40 m ahead' / ... from the distances."""
        if self.person_near_box:
            return "a person right in front of the camera"
        if self.reason in (8, 6) and self.person_m is not None:
            return "a person {:.2f} m away".format(self.person_m)
        if self.reason in (9, 7) and self.obstacle_m is not None:
            return "an obstacle {:.2f} m ahead".format(self.obstacle_m)
        # otherwise (standing, a refused reset) whichever is nearer its trigger
        p, o = self.person_m, self.obstacle_m
        if p is not None and (o is None or p - 0.6 <= o - 0.45):
            return "a person {:.2f} m away".format(p)
        if o is not None:
            return "an obstacle {:.2f} m ahead".format(o)
        return self.text or self.state

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


def taken_over_reason(st):
    """The abort reason when the reflex has taken over (st.state RETREAT or STAND). No
    distance: the retreat is a timed command and nothing measures how far it went."""
    if st.reason == 11:      # the limit
        return ("stepped back too often -- I am standing still and need help "
                "(too close: {})".format(st.cause()))
    if st.state == RETREAT or st.reason in (8, 9):
        return ("stepped back because of {} -- standing still until a new "
                "instruction".format(st.cause()))
    return "the bridge is holding me standing ({})".format(st.text or "after stepping back")


def blocked_reason(st):
    """The abort reason when the bridge cuts a forward walk."""
    if st.reason == 3:
        return "forward walking blocked by the bridge: camera frames stale (is r1_perception running?)"
    if st.reason in (4, 5):
        return "forward walking blocked by the bridge: {} -- it cannot see the way".format(st.text)
    return "forward walking blocked by the bridge: {} inside the stop distance".format(st.cause())


def reset_before_start(latest, send_reset, spin, timeout_s=1.0, clock=None):
    """Before a confirmed plan starts: if the reflex holds the robot standing, ask the
    bridge to let go. `latest()` -> the newest ReflexStatus or None; `send_reset()`
    publishes the reset; `spin(dt)` lets messages in. Returns (ok, message):
      ok True   the reflex is not holding the robot (or let go);
      ok False  the bridge refused (still too close) or never answered."""
    import time
    clock = clock or time.monotonic
    st = latest()
    if st is None or st.state not in TAKEN_OVER:
        return True, None
    if st.state == RETREAT:
        return False, "the robot is stepping back right now ({}); ask again in a moment".format(
            st.cause())
    t_sent = clock()
    send_reset()
    seen_before = st
    while clock() - t_sent < timeout_s:
        spin(0.05)
        st = latest()
        if st is not seen_before and st is not None:
            if st.reset_refused:
                return False, ("the robot is standing still after stepping back and it is "
                               "still too close: {}. Move it away (a person beyond 0.8 m, an "
                               "obstacle beyond 0.65 m) and ask again".format(st.cause()))
            if st.state == ARMED:
                return True, "the reflex let go (it had stepped back earlier)"
    return False, "no answer from the bridge to the reflex reset within {:.1f} s".format(timeout_s)


# ------------------------------------------------------------------ dry run
class SimReflex(object):
    """The bridge's reflex for the dry run: the reference state machine on what a
    FakeWorld frame says, stepped at the executor's rate. `step()` takes the mission's
    command (and its dance message) and returns (the command the robot actually gets,
    ReflexStatus). The dance policy is taken to work: dancing, the base stays put."""

    def __init__(self, cfg=None):
        RX, _ = reference()
        self.RX = RX
        self.CD = conductor()
        self.cd = self.CD.Conductor(cfg)
        self.fsm = self.cd.reflex
        self.frame = None
        self.last = None
        self._t = None

    def see(self, frame):
        """A FakeWorld frame (percept.Frame with .prox) -> the reflex's next input."""
        p = frame.prox or {}
        self.frame = self.RX.Frame(frame.seq, frame.t_cap, p.get("person_m"),
                                   p.get("person_near_box", False), p.get("obstacle_m"),
                                   p.get("depth_ok", True), p.get("center_valid", 1.0))

    def step(self, t, cmd, reset=False, running=True, dance=None):
        f, self.frame = self.frame, None
        # the bridge steps the conductor every DT (50 Hz) and counts a dance in those
        # ticks: a dry run at a lower rate (the CLI's 10 Hz) steps it as often in between,
        # or its bars last 5x too long and a dance never ends in time
        dt = self.CD.DT
        n = 1 if self._t is None else max(1, int(round((t - self._t) / dt)))
        for k in range(n):
            co = self.cd.step(t - (n - 1 - k) * dt, cmd, f if k == 0 else None, reset and k == 0,
                              running, dance, True)
        self._t = t
        o = co.reflex
        cur = self.fsm.frame
        st = ReflexStatus(o.state, o.reason, self.RX.reason_text(o.reason), o.blocked,
                          o.reset_refused, cmd, o.cmd(),
                          None if cur is None else cur.person_m,
                          False if cur is None else cur.person_near_box,
                          None if cur is None or not cur.depth_ok else cur.obstacle_m,
                          None if cur is None else t - cur.t_cap,
                          len(self.fsm.retreats), t, co.policy, co.dance_state, co.dance_reason,
                          co.dance_tick, self.CD.dance_reason_text(co.dance_reason))
        self.last = st
        return o.cmd(), st
