"""V3 conductor: which policy drives the robot, the dance's clock, and the reflex around it.

The reference implementation of what the bridge does each control step (50 Hz) once the
robot has two policies (V3 plan 01 §1.1-1.4):

  walk   v2: walks, turns, stands -- and is the only one that can step back
  dance  the dance policy (v3/dance): dances in place, cannot step back

Both run every step; the bridge sends one of them to the motors. This file decides
which, and when to switch. The proximity reflex (reflex.py) runs every step as before
and keeps the last word: a person or an obstacle too close while dancing makes the
walking policy take over on that same step and step back, and the dance is over.

Inputs per step, besides the reflex's own (command, proximity frame, reset, RUNNING):
  dance     None: no dance message this step. True: "dance" -- starts a dance if none is
            running, and is the heartbeat while one is. False: end it.
  dance_ok  the dance policy's latest output is fresh and finite.

The dance's clock counts control steps from the step it started on (tick 0 = the first
pose of the dance), so the bridge, the reference and the policy runner agree on it
exactly. A dance ends in one of two ways:
  normally   end requested, heartbeat lost, the length cap, or the camera gone stale:
             it keeps dancing until the next bar boundary -- where the dance is back in
             the default standing pose at rest -- and hands over there
  at once    the reflex steps back (or the bridge stops RUNNING, or the reflex is off),
             or the dance policy's output is stale or not finite

After a dance ends, for whatever reason, the dance messages that may still be coming are
the old dance's: a new dance needs a fresh request, i.e. the heartbeat must lapse first
(no True for dance_heartbeat_s). A mission that keeps sending never restarts a dance the
bridge has stopped.

Output: the policy, the dance tick (or -1), and the walking policy's command (the
reflex's output: while dancing (0, 0, 0), which is what keeps v2's history warm and
standing, or the retreat command on the step the reflex takes over).
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reflex as RX  # noqa: E402

WALK, DANCE = "walk", "dance"
D_OFF, D_DANCING, D_ENDING = "OFF", "DANCING", "ENDING"
DANCE_STATES = (D_OFF, D_DANCING, D_ENDING)

# Why the dance did not start, is ending, or stopped. Integers for the golden CSV.
D_NONE = 0
D_REFUSED_DISABLED = 1      # dance_enable false, or the reflex is off: no dance without it
D_REFUSED_NOT_ARMED = 2     # the reflex is stepping back or holding the robot standing
D_REFUSED_STALE = 3         # no fresh camera frame to check the space with
D_REFUSED_TOO_CLOSE = 4     # a person or an obstacle within dance_clear_m
D_REFUSED_NOT_STILL = 5     # the walking policy has not been standing still for dance_still_s
D_REFUSED_NO_POLICY = 6     # the dance policy has no fresh output
D_END_REQUEST = 7           # ending: asked to stop
D_END_HEARTBEAT = 8         # ending: no dance message within dance_heartbeat_s
D_END_MAX = 9               # ending: danced dance_max_s
D_END_STALE = 10            # ending: camera frames stale
D_STOP_RETREAT = 11         # stopped at once: the reflex stepped back (or stands: limit)
D_STOP_FAULT = 12           # stopped at once: the dance policy's output is stale or not finite
D_STOP_OFF = 13             # stopped at once: bridge not RUNNING, or reflex disabled
D_DONE = 14                 # handed over at a bar boundary after a normal end
DANCE_REASONS = {
    D_NONE: "", D_REFUSED_DISABLED: "dance refused: dancing (or the reflex) is disabled",
    D_REFUSED_NOT_ARMED: "dance refused: the reflex is stepping back or holding the robot",
    D_REFUSED_STALE: "dance refused: no fresh camera frame",
    D_REFUSED_TOO_CLOSE: "dance refused: someone or something is too close",
    D_REFUSED_NOT_STILL: "dance refused: not standing still yet",
    D_REFUSED_NO_POLICY: "dance refused: the dance policy is not running",
    D_END_REQUEST: "dance ending: asked to stop", D_END_HEARTBEAT: "dance ending: no heartbeat",
    D_END_MAX: "dance ending: maximum length", D_END_STALE: "dance ending: camera frames stale",
    D_STOP_RETREAT: "dance stopped: stepped back", D_STOP_FAULT: "dance stopped: dance policy fault",
    D_STOP_OFF: "dance stopped: bridge not RUNNING or reflex off",
    D_DONE: "danced; back to standing",
}
REFUSALS = tuple(range(D_REFUSED_DISABLED, D_REFUSED_NO_POLICY + 1))

DANCE_DEFAULTS = {
    "dance_enable": True,
    "dance_person_trigger_m": 0.9,      # plan 01 §1.3, user 10-23: the arms reach ~0.5 m
    "dance_obstacle_trigger_m": 0.9,
    "dance_clear_m": 1.0,               # user 10-23: nobody within 1.0 m to start
    "dance_still_s": 1.0,               # the walking policy stood still this long first
    "dance_heartbeat_s": 0.5,
    "dance_max_s": 63.0,                # mission's 60 s + up to a bar to end on
    "dance_bar_ticks": (150, 150, 100, 100),    # v3/dance/routine.py BARS at 50 Hz
}
DT = 0.02


class Output(object):
    __slots__ = ("policy", "dance_tick", "reflex", "dance_state", "dance_reason")

    def __init__(self, policy, dance_tick, reflex, dance_state, dance_reason):
        self.policy = policy
        self.dance_tick = dance_tick
        self.reflex = reflex                  # reflex.Output: the walking policy's command
        self.dance_state = dance_state
        self.dance_reason = dance_reason

    def cmd(self):
        return self.reflex.cmd()


class Conductor(object):
    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.cfg = dict(DANCE_DEFAULTS)
        self.cfg.update({k: cfg.pop(k) for k in list(cfg) if k in DANCE_DEFAULTS})
        self.reflex = RX.Reflex(cfg)
        c = self.cfg
        self.bars = tuple(int(b) for b in c["dance_bar_ticks"])
        self.loop = sum(self.bars)
        # bar boundaries within a loop, in ticks: 0, 150, 300, 400
        self.boundaries = tuple(sum(self.bars[:i]) for i in range(len(self.bars)))
        self.policy = WALK
        self.dance_state = D_OFF
        self.dance_reason = D_NONE
        self.tick = -1
        self.t_heartbeat = None
        self.need_fresh = False               # the last dance's messages may still arrive
        self.cmd_hist = []                    # (t, walking command applied) for "standing still"

    # ---------------------------------------------------------------- helpers
    def at_boundary(self, tick):
        return tick % self.loop in self.boundaries

    def _still(self, t):
        """The walking policy has been given (0, 0, 0) for the last dance_still_s."""
        w = self.cfg["dance_still_s"]
        if not self.cmd_hist or t - self.cmd_hist[0][0] < w - 1e-9:
            return False
        return all(c == (0.0, 0.0, 0.0) for ti, c in self.cmd_hist if t - ti < w + 1e-9)

    def _start_check(self, t):
        """D_NONE if a dance may start now, else the refusal."""
        r = self.reflex
        if not self.cfg["dance_enable"] or not r.cfg["enable"]:
            return D_REFUSED_DISABLED
        if r.state != RX.ARMED:
            return D_REFUSED_NOT_ARMED
        if not r._fresh(t):
            return D_REFUSED_STALE
        p, o = r._distances(r.frame)
        if not r.frame.depth_ok or min(p, o) <= self.cfg["dance_clear_m"]:
            return D_REFUSED_TOO_CLOSE
        if not self._still(t):
            return D_REFUSED_NOT_STILL
        return D_NONE

    def _stop(self, reason):
        self.policy, self.dance_state, self.dance_reason, self.tick = WALK, D_OFF, reason, -1
        self.need_fresh = True

    # ------------------------------------------------------------------- step
    def step(self, t, cmd, frame=None, reset=False, running=True, dance=None, dance_ok=True):
        c = self.cfg
        cmd = tuple(float(x) for x in cmd)
        if self.need_fresh and (self.t_heartbeat is None
                                or t - self.t_heartbeat > c["dance_heartbeat_s"] + 1e-9):
            self.need_fresh = False
        if dance is True:
            self.t_heartbeat = t
        dancing = self.policy == DANCE
        triggers = (c["dance_person_trigger_m"], c["dance_obstacle_trigger_m"]) if dancing else None
        out = self.reflex.step(t, (0.0, 0.0, 0.0) if dancing else cmd, frame, reset, running, triggers)

        if dancing:
            self.tick += 1
            if out.state in (RX.RETREAT, RX.STAND):
                self._stop(D_STOP_RETREAT)
            elif out.state == RX.OFF:
                self._stop(D_STOP_OFF)
            elif not dance_ok:
                self._stop(D_STOP_FAULT)
            else:
                if self.dance_state == D_DANCING:
                    if dance is False:
                        self.dance_state, self.dance_reason = D_ENDING, D_END_REQUEST
                    elif t - self.t_heartbeat > c["dance_heartbeat_s"] + 1e-9:
                        self.dance_state, self.dance_reason = D_ENDING, D_END_HEARTBEAT
                    elif self.tick * DT >= c["dance_max_s"] - 1e-9:
                        self.dance_state, self.dance_reason = D_ENDING, D_END_MAX
                    elif not self.reflex._fresh(t):
                        self.dance_state, self.dance_reason = D_ENDING, D_END_STALE
                if self.dance_state == D_ENDING and self.at_boundary(self.tick) and self.tick > 0:
                    self._stop(D_DONE)
        elif dance is True and self.dance_state == D_OFF and not self.need_fresh:
            why = D_REFUSED_NO_POLICY if not dance_ok else self._start_check(t)
            if why == D_NONE:
                self.policy, self.dance_state, self.dance_reason, self.tick = DANCE, D_DANCING, D_NONE, 0
            else:
                self.dance_reason = why

        # dance steps count as "not standing still": a new dance needs the walking policy
        # to have stood for dance_still_s after the last one
        applied = out.cmd() if self.policy == WALK else None
        w = c["dance_still_s"]
        self.cmd_hist = [(ti, ci) for ti, ci in self.cmd_hist if t - ti < w + 1e-9]
        self.cmd_hist.append((t, applied))
        return Output(self.policy, self.tick if self.policy == DANCE else -1, out,
                      self.dance_state, self.dance_reason)


def dance_reason_text(code):
    return DANCE_REASONS.get(code, "?")
