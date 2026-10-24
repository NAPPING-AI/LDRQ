"""V3 proximity reflex: the reference implementation of the bridge's state machine.

The bridge (C++, v3/ros2_ws/src/r1_hw_bridge, reflex_fsm.hpp) runs the same machine
once per control step (50 Hz) and must agree with this file tick for tick on the
golden cases (cases.py -> golden.csv). mission_ctl's dry run uses this file too.

Per tick it takes what the bridge has at that moment -- the velocity command from
mission_ctl, the newest proximity frame (/r1_perception/proximity), a reset request,
whether the bridge is RUNNING -- and returns the command written into the policy's
observation, with the state and why.

States:
  OFF      reflex disabled or the bridge not RUNNING: the command passes untouched.
  ARMED    the command passes, except that forward speed is clamped to 0 while
           anything is inside the stop distance (stop_distance.py, grows with speed)
           or the camera cannot vouch for the path (stale frames, no depth, a hole in
           the middle of the depth image).
  RETREAT  (retreat_vx, 0, 0) for retreat_s, whatever the command says.
  STAND    (0, 0, 0) -- the LLM's `stand`: v2 steps in place. Held until a reset
           (mission_ctl, when the operator confirms a new plan) finds the way clear.

A retreat needs `confirm_frames` consecutive frames with a person nearer than
person_trigger_m (or a person box so tall it is near) or an obstacle nearer than
obstacle_trigger_m. In STAND a second retreat needs a NEW approach: the hazard must
first have moved beyond trigger + clear_margin. Someone who stays close gets a robot
that stays standing. At most max_retreats per window_s; past that the robot stands
and says it needs help.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stop_distance as SD  # noqa: E402

OFF, ARMED, RETREAT, STAND = "OFF", "ARMED", "RETREAT", "STAND"
STATES = (OFF, ARMED, RETREAT, STAND)

# Why the output differs from the input (or why a reset was refused). Integers in the
# golden CSV so that C++ compares codes, not strings.
R_NONE = 0              # command passed as given
R_DISABLED = 1          # OFF: reflex_enable false
R_NOT_RUNNING = 2       # OFF: bridge not RUNNING
R_STALE = 3             # forward blocked: newest frame older than stale_s (or none yet)
R_NO_DEPTH = 4          # forward blocked: the frame has no depth
R_DEPTH_HOLE = 5        # forward blocked: center of the depth image mostly without returns
R_STOP_PERSON = 6       # forward blocked: a person inside the stop distance
R_STOP_OBSTACLE = 7     # forward blocked: an obstacle inside the stop distance
R_RETREAT_PERSON = 8    # RETREAT, started by a person
R_RETREAT_OBSTACLE = 9  # RETREAT, started by an obstacle
R_STAND = 10            # STAND after a retreat
R_LIMIT = 11            # STAND: max_retreats reached -- needs help
R_RESET_REFUSED = 12    # STAND: a reset came but the way is not clear
REASONS = {
    R_NONE: "", R_DISABLED: "reflex disabled", R_NOT_RUNNING: "bridge not RUNNING",
    R_STALE: "camera frames stale", R_NO_DEPTH: "no depth", R_DEPTH_HOLE: "depth hole ahead",
    R_STOP_PERSON: "person inside the stop distance",
    R_STOP_OBSTACLE: "obstacle inside the stop distance",
    R_RETREAT_PERSON: "stepping back: person too close",
    R_RETREAT_OBSTACLE: "stepping back: obstacle too close",
    R_STAND: "standing after stepping back",
    R_LIMIT: "stepped back too often -- standing, needs help",
    R_RESET_REFUSED: "reset refused: still too close",
}

DEFAULTS = {
    "enable": True,
    "person_trigger_m": 0.6,
    "obstacle_trigger_m": 0.45,
    "clear_margin_m": 0.2,
    "confirm_frames": 2,
    "retreat_vx": -0.3,
    "retreat_s": 1.5,
    "max_retreats": 2,
    "window_s": 10.0,
    "stale_s": 0.3,
    "center_valid_min": 0.2,
    "release_margin_m": 0.10,
    # stop distance (stop_distance.py)
    "person_floor_m": 1.2,
    "obstacle_floor_m": 0.6,
    "stop_margin_m": 0.10,
    "latency_s": 0.25,
    "k_lat": 1.5,
    "k_brake": 1.3,
    "speed_window_s": 1.0,
}

INF = float("inf")


class Frame(object):
    """The fields of a proximity frame the reflex uses (pipeline.PROX_LAYOUT)."""
    __slots__ = ("seq", "t_cap", "person_m", "person_near_box", "obstacle_m", "depth_ok",
                 "center_valid")

    def __init__(self, seq, t_cap, person_m=None, person_near_box=False, obstacle_m=None,
                 depth_ok=True, center_valid=1.0):
        self.seq = int(seq)
        self.t_cap = float(t_cap)
        self.person_m = person_m
        self.person_near_box = bool(person_near_box)
        self.obstacle_m = obstacle_m
        self.depth_ok = bool(depth_ok)
        self.center_valid = float(center_valid)

    @classmethod
    def from_parsed(cls, p):
        """From perception's pipeline.parse_proximity()."""
        return cls(p["seq"], p["t_cap_mono"], p["person_m"], p["person_near_box"],
                   p["obstacle_m"], p["depth_ok"], p["center_valid"] or 0.0)


class Output(object):
    __slots__ = ("vx", "vy", "wz", "state", "reason", "blocked", "reset_refused")

    def __init__(self, cmd, state, reason, blocked=False, reset_refused=False):
        self.vx, self.vy, self.wz = (float(c) for c in cmd)
        self.state = state
        self.reason = reason
        self.blocked = bool(blocked)
        self.reset_refused = bool(reset_refused)

    def cmd(self):
        return (self.vx, self.vy, self.wz)


class Reflex(object):
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        c = self.cfg
        self.sd_cfg = {"person": {"trigger_m": c["person_trigger_m"], "floor_m": c["person_floor_m"]},
                       "obstacle": {"trigger_m": c["obstacle_trigger_m"],
                                    "floor_m": c["obstacle_floor_m"]},
                       "margin_m": c["stop_margin_m"], "latency_s": c["latency_s"],
                       "k_lat": c["k_lat"], "k_brake": c["k_brake"]}
        self.state = OFF
        self.frame = None
        self.near_count = 0
        self.near_kind = None
        self.last_seq = None
        self.rearmed = False
        self.t_retreat = None
        self.retreats = []          # start times, for the max_retreats window
        self.stand_reason = R_STAND
        self.blocked = None         # None, or the reason code of the current forward block
        self.vx_hist = []           # (t, applied vx), for the speed window

    # ---------------------------------------------------------------- helpers
    def _distances(self, f):
        """(person distance, obstacle distance) of a frame; INF = nothing seen. A person
        box tall enough to be near counts as distance 0."""
        p = 0.0 if f.person_near_box else (INF if f.person_m is None else float(f.person_m))
        o = INF if (not f.depth_ok or f.obstacle_m is None) else float(f.obstacle_m)
        return p, o

    def _near(self, f, triggers=None):
        """'person' / 'obstacle' if this frame is inside a retreat trigger, else None.
        `triggers` (person m, obstacle m) replaces the configured ones for this frame:
        the dance (conductor.py) triggers farther out, because the arms reach."""
        p, o = self._distances(f)
        tp, to = triggers or (self.cfg["person_trigger_m"], self.cfg["obstacle_trigger_m"])
        if p < tp:
            return "person"
        if o < to:
            return "obstacle"
        return None

    def _clear_of_trigger(self, f):
        """Beyond trigger + clear_margin on both counts, with depth to say so."""
        p, o = self._distances(f)
        m = self.cfg["clear_margin_m"]
        return (f.depth_ok and p > self.cfg["person_trigger_m"] + m
                and o > self.cfg["obstacle_trigger_m"] + m)

    def _fresh(self, t):
        return self.frame is not None and t - self.frame.t_cap <= self.cfg["stale_s"]

    def _speed(self, t, vx_req):
        """Forward speed the stop distance is computed for: the largest of the request
        and the commands applied over the last speed_window_s (step() keeps the window)."""
        w = self.cfg["speed_window_s"]
        v = max([vx_req] + [v for (ti, v) in self.vx_hist if t - ti < w])
        return max(0.0, v)

    def _block_reason(self, t, vx_req):
        """Why forward motion must be blocked now, or R_NONE. Hysteresis: a block by
        distance is released only beyond stop + release_margin."""
        if not self._fresh(t):
            return R_STALE
        f = self.frame
        if not f.depth_ok:
            return R_NO_DEPTH
        if f.center_valid < self.cfg["center_valid_min"]:
            return R_DEPTH_HOLE
        v = self._speed(t, vx_req)
        p, o = self._distances(f)
        sp = SD.stop_distance("person", v, self.sd_cfg)
        so = SD.stop_distance("obstacle", v, self.sd_cfg)
        rel = self.cfg["release_margin_m"]
        hold_p = rel if self.blocked == R_STOP_PERSON else 0.0
        hold_o = rel if self.blocked == R_STOP_OBSTACLE else 0.0
        if p < sp + hold_p:
            return R_STOP_PERSON
        if o < so + hold_o:
            return R_STOP_OBSTACLE
        return R_NONE

    def _start_retreat(self, t, kind):
        w = self.cfg["window_s"]
        self.retreats = [ti for ti in self.retreats if t - ti < w]
        self.near_count = 0
        self.rearmed = False
        if len(self.retreats) >= int(self.cfg["max_retreats"]):
            self.state, self.stand_reason = STAND, R_LIMIT
            return
        self.retreats.append(t)
        self.state, self.t_retreat = RETREAT, t
        self.retreat_reason = R_RETREAT_PERSON if kind == "person" else R_RETREAT_OBSTACLE

    # ------------------------------------------------------------------- step
    def step(self, t, cmd, frame=None, reset=False, running=True, triggers=None):
        """One control step at time t (s, CLOCK_MONOTONIC). `cmd` is (vx, vy, wz) as the
        bridge would otherwise write it; `frame` a new proximity Frame or None;
        `triggers` (person m, obstacle m) overrides the retreat triggers for a frame that
        arrives on this tick (None: the configured ones)."""
        c = self.cfg
        cmd = tuple(float(x) for x in cmd)
        if not c["enable"] or not running:
            self.state, self.near_count, self.blocked, self.frame = OFF, 0, None, None
            self.last_seq = None
            return Output(cmd, OFF, R_DISABLED if not c["enable"] else R_NOT_RUNNING)
        if self.state == OFF:
            self.state, self.near_count, self.rearmed = ARMED, 0, False

        if frame is not None and frame.seq != self.last_seq:
            self.frame, self.last_seq = frame, frame.seq
            fresh_frame = t - frame.t_cap <= c["stale_s"]
            kind = self._near(frame, triggers) if fresh_frame else None
            # consecutive near frames, person or obstacle alike; the latest names the cause
            if kind is not None:
                self.near_count, self.near_kind = self.near_count + 1, kind
            else:
                self.near_count, self.near_kind = 0, None
            if self.state == STAND and fresh_frame and self._clear_of_trigger(frame):
                self.rearmed = True

        refused = False
        if self.state == RETREAT and t - self.t_retreat >= c["retreat_s"] - 1e-9:
            self.state, self.stand_reason, self.near_count = STAND, R_STAND, 0
        if self.state == STAND and reset:
            if self._fresh(t) and self._clear_of_trigger(self.frame):
                self.state, self.near_count, self.blocked = ARMED, 0, None
            else:
                refused = True
        # in STAND only a new approach, and never once the limit has latched
        if self.near_count >= int(c["confirm_frames"]) and (
                self.state == ARMED or (self.state == STAND and self.rearmed
                                        and self.stand_reason != R_LIMIT)):
            self._start_retreat(t, self.near_kind)

        if self.state == RETREAT:
            out = Output((c["retreat_vx"], 0.0, 0.0), RETREAT, self.retreat_reason)
            self.blocked = None
        elif self.state == STAND:
            out = Output((0.0, 0.0, 0.0), STAND,
                         R_RESET_REFUSED if refused else self.stand_reason,
                         reset_refused=refused)
            self.blocked = None
        else:
            why = self._block_reason(t, cmd[0])
            self.blocked = why if why != R_NONE else None
            if why != R_NONE and cmd[0] > 0.0:
                out = Output((0.0, cmd[1], cmd[2]), ARMED, why, blocked=True)
            else:
                out = Output(cmd, ARMED, why, blocked=why != R_NONE)
        w = c["speed_window_s"]
        self.vx_hist = [(ti, v) for (ti, v) in self.vx_hist if t - ti < w]
        self.vx_hist.append((t, out.vx))
        return out


def reason_text(code):
    return REASONS.get(code, "?")


if __name__ == "__main__":
    r = Reflex()
    print("defaults:", r.cfg)
    print("stop distance at 0 / 0.3 / 1.0 m/s: person {:.2f} {:.2f} {:.2f}, obstacle {:.2f} {:.2f} {:.2f}"
          .format(*(SD.stop_distance(k, v, r.sd_cfg) for k in ("person", "obstacle")
                    for v in (0.0, 0.3, 1.0))))
