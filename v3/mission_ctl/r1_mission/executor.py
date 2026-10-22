"""The behavior state machine. Pure Python: the clock and the measurements are
arguments, so the whole thing is testable without ROS or a robot.

Deliberately NOT in the bridge. The bridge is the only thing that writes
rt/lowcmd, it has been stable since W06, and every change there is a safety
regression risk. Keeping mission logic in a separate process also means it can be
killed without disturbing the bridge's PD hold -- and because the bridge's
cmd_vel deadman decays a stale command to zero in 500 ms, a mission process that
dies stops the robot by construction rather than by remembering to.
"""
from __future__ import annotations

import math

from . import percept as PC
from .plan import OBJECT_OPS, TAPER_DEG, WALK_MIN_VX, expected_turn_s, taper_floor

IDLE, RUNNING, DONE, ABORTED = "IDLE", "RUNNING", "DONE", "ABORTED"


class Output(object):
    __slots__ = ("vx", "vy", "wz", "state", "event")

    def __init__(self, vx, vy, wz, state, event=None):
        self.vx, self.vy, self.wz = vx, vy, wz
        self.state = state
        self.event = event

    def as_tuple(self):
        return (self.vx, self.vy, self.wz)


class Record(object):
    """What actually happened for one primitive. This is the report the LLM layer
    sends back, so every field is either measured or explicitly an estimate."""

    def __init__(self, prim):
        self.prim = prim
        self.commanded_s = prim.duration_s
        self.expected_s = prim.expected_s
        self.actual_s = None
        self.target_deg = prim.yaw_deg
        self.achieved_deg = None
        self.est_distance_m = prim.distance_m
        self.est_err_m = prim.distance_err_m
        self.aborted = None
        # turn_to / walk_to
        self.looks = []          # each confirmed sighting: t, bearing, range, conf
        self.turns = []          # each turn's IMU-measured angle, degrees
        self.segments = []       # each walk: vx, seconds, open-loop meters
        self.final = None        # the last confirmed sighting
        self.note = None

    def _object_dict(self, d):
        p = self.prim
        d["target"] = p.target_cls
        d["object"] = p.target_text
        if p.stop_at_m is not None:
            d["stop_at_m"] = round(p.stop_at_m, 3)
            d["stop_default"] = p.stop_default
        d["turns_deg"] = [round(a, 2) for a in self.turns]
        d["turned_deg"] = round(sum(self.turns), 2)
        d["turn_source"] = "imu_yaw"
        if self.segments:
            d["segments"] = [{"vx": round(v, 3), "s": round(t, 3), "est_m": round(m, 3)}
                             for v, t, m in self.segments]
            d["walked_est_m"] = round(sum(m for _, _, m in self.segments), 3)
            d["walk_source"] = ("open loop per walk: time x commanded speed (not measured); "
                                "re-measured by the depth camera after each walk")
        d["looks"] = len(self.looks)
        if self.final is not None:
            d["final_bearing_deg"] = round(self.final["bearing_deg"], 2)
            if self.final.get("range_m") is not None:
                d["final_range_m"] = round(self.final["range_m"], 3)
                if p.stop_at_m is not None:
                    d["stop_error_m"] = round(self.final["range_m"] - p.stop_at_m, 3)
            d["final_source"] = "camera (bearing) and depth camera (range), standing still"
        if self.note:
            d["note"] = self.note
        return d

    def to_dict(self):
        d = {"op": self.prim.op, "text": self.prim.text}
        if self.prim.op == "walk" and self.prim.direction != "forward":
            d["direction"] = self.prim.direction
        if self.prim.op in OBJECT_OPS:
            if self.actual_s is not None:
                d["actual_s"] = round(self.actual_s, 3)
            self._object_dict(d)
            if self.aborted:
                d["aborted"] = self.aborted
            return d
        if self.commanded_s is not None:
            d["commanded_s"] = round(self.commanded_s, 3)
        if self.expected_s is not None:
            # a closed-loop turn as executed on a robot that tracks exactly; the
            # ratio actual_s / expected_s is what stage B reads (~1.25 on the real robot)
            d["expected_s"] = round(self.expected_s, 3)
        if self.actual_s is not None:
            d["actual_s"] = round(self.actual_s, 3)
        if self.target_deg is not None:
            d["target_deg"] = round(self.target_deg, 2)
        if self.achieved_deg is not None:
            d["achieved_deg"] = round(self.achieved_deg, 2)
            d["source"] = "imu_yaw"
        if self.est_distance_m is not None:
            est = self.est_distance_m
            if self.aborted and self.commanded_s and self.actual_s is not None:
                # an interrupted walk covered only its share of the plan's distance;
                # the robot log (2026-10-14) had 0.4 m for a walk stopped at 1.8 of 4 s
                d["planned_distance_m"] = round(est, 3)
                est = est * min(1.0, self.actual_s / self.commanded_s)
            d["est_distance_m"] = round(est, 3)
            if self.est_err_m is None:
                d["distance_source"] = ("open loop: commanded time x commanded speed "
                                        "(assumed equal to ground speed, not measured)")
            else:
                d["est_err_m"] = round(self.est_err_m, 3)
                d["distance_source"] = "open loop: commanded time x calibrated speed"
        if self.aborted:
            d["aborted"] = self.aborted
        return d


class Executor(object):
    def __init__(self, prims, limits):
        self.prims = list(prims)
        self.lim = limits
        self.state = IDLE
        self.i = -1
        self.records = []
        self._t_prim = None          # when the current primitive started
        self._from = (0.0, 0.0, 0.0)  # (vx, vy, wz) we are ramping away from
        self._yaw0 = None
        self._t_settle = None
        self._t_first = None
        self._t_last = None          # the clock at the last step()
        self.abort_reason = None
        self._ob = None              # turn_to / walk_to: the current phase and its state

    # ---------------------------------------------------------------- helpers
    @property
    def current(self):
        return self.prims[self.i] if 0 <= self.i < len(self.prims) else None

    def _target(self, prim):
        if prim is None or prim.op in ("stand", "stop"):
            return (0.0, 0.0, 0.0)
        return (prim.vx, prim.vy, prim.wz)

    def _ramped(self, t, prim):
        """Linear ramp from the previous target to this one over ramp_s.

        Ramping is a design choice, not a policy limitation: training resampled
        the command every 10 s as a step, so steps are in distribution -- but a
        0.5 s ramp costs nothing and keeps the first control cycles of each
        primitive out of the sharpest transient.
        """
        tgt = self._target(prim)
        if self.lim.ramp_s <= 0.0:
            return tgt
        a = min(1.0, max(0.0, (t - self._t_prim) / self.lim.ramp_s))
        return tuple(f + a * (g - f) for f, g in zip(self._from, tgt))

    def _advance(self, t):
        prev = self._target(self.current)
        self.i += 1
        self._from = prev
        self._t_prim = t
        self._yaw0 = None
        self._ob = None
        if self.current is not None:
            self.records.append(Record(self.current))

    def _turn_wz(self, remaining, mag, elapsed):
        """The closed-loop turn's rate: full until TAPER_DEG from the target, then in
        proportion to what is left, never below the floor that still moves; ramped in
        over ramp_s. plan.expected_turn_s() models exactly this; change them together."""
        taper = min(1.0, max(taper_floor(mag, self.lim.turn_min_wz), abs(remaining)
                             / max(1e-6, math.radians(TAPER_DEG))))
        sign = 1.0 if remaining > 0 else -1.0
        a = min(1.0, max(0.0, elapsed / self.lim.ramp_s)) if self.lim.ramp_s > 0 else 1.0
        return sign * mag * taper * a

    def _guard(self, t, percept):
        """Why a walk under camera guard must stop now, or None. Acts on single frames:
        the safety check does not wait for the confirmation that targets need."""
        if percept is None or percept.latest is None:
            return "no camera frames (is r1_perception running?)"
        age = percept.age(t)
        if age > self.lim.obj_stale_s:
            return "camera frames stale ({:.2f} s old, limit {:.2f} s)".format(
                age, self.lim.obj_stale_s)
        g = percept.latest.guard
        if g.get("person_near"):
            pm = g.get("person_m")
            return "person {} -- stopped".format(
                "{:.2f} m ahead".format(pm) if pm is not None else "close in view")
        if g.get("obstacle"):
            om = g.get("obstacle_m")
            return "obstacle {} in the walking corridor -- stopped".format(
                "{:.2f} m ahead".format(om) if om is not None else "close")
        return None

    # ------------------------------------------------------------------- step
    def step(self, t, yaw=None, bridge_state="RUNNING", percept=None):
        """Advance to wall-clock `t`. `yaw` is accumulated radians from
        YawTracker (None if not yet available). `percept` is a percept.Percept (None
        without a camera): turn_to / walk_to need it, and so does every walk when
        perception_guard is "all". Returns an Output."""
        if self.state in (DONE, ABORTED):
            return Output(0.0, 0.0, 0.0, self.state)

        if self._t_first is None:
            self._t_first = t
        self._t_last = t

        # The bridge's own health machine outranks the mission. A DEGRADED bridge
        # is already commanding damping; sending velocity into that is pointless
        # and hides the fault. Once running, anything but RUNNING -- including a
        # status stream that has gone quiet -- ends the plan.
        if self.state == RUNNING and bridge_state != "RUNNING":
            return self._abort("bridge left RUNNING ({})".format(bridge_state), t)

        if self.state == IDLE:
            if bridge_state == "DEGRADED":
                return self._abort("bridge is DEGRADED -- clear it (~/resume) first", t)
            if bridge_state != "RUNNING":
                # Wait, do not start: before the first ~/status arrives the state is
                # simply unknown, and the policy may still be filling its history.
                # Waiting for ever would hang any caller, so it is bounded.
                if t - self._t_first > self.lim.start_timeout_s:
                    why = ("no ~/status received at all -- is the stack up, and is "
                           "this terminal on the same ROS_DOMAIN_ID / "
                           "ROS_LOCALHOST_ONLY / RMW?"
                           if bridge_state in ("UNKNOWN", "NO_STATUS")
                           else "bridge stayed in {}".format(bridge_state))
                    return self._abort("never saw the bridge RUNNING within {:.0f} s: {}"
                                       .format(self.lim.start_timeout_s, why), t)
                return Output(0.0, 0.0, 0.0, IDLE, None)
            self.state = RUNNING
            self._advance(t)
            return Output(0.0, 0.0, 0.0, RUNNING,
                          "start: " + (self.current.summary() if self.current else ""))

        # Once the final primitive is done we are only holding zero. Re-entering
        # the per-primitive logic here would keep rewriting that primitive's
        # record (its actual_s would come out as duration + settle_s).
        if self._t_settle is not None:
            return self._finish(t)

        prim, rec = self.current, self.records[-1] if self.records else None
        if prim is None:
            return self._finish(t)

        elapsed = t - self._t_prim
        finished, note = False, None

        if prim.op in OBJECT_OPS:
            if prim.budget_s is not None and elapsed > prim.budget_s:
                rec.aborted = "{} took longer than its {:.0f} s budget".format(
                    prim.op, prim.budget_s)
                return self._abort(rec.aborted, t)
            out, finished, note = self._object_tick(t, yaw, prim, rec, percept)
            if out is not None:
                return out
        elif prim.op == "walk" and self.lim.perception_guard == "all":
            why = self._guard(t, percept)
            if why:
                rec.aborted = why
                return self._abort(why, t)

        if prim.op in OBJECT_OPS:
            pass
        elif prim.closed_loop:
            if yaw is None:
                if elapsed > 2.0:
                    return self._abort("no IMU yaw after 2 s -- is ~/imu publishing?", t)
                return Output(0.0, 0.0, 0.0, RUNNING, "waiting for IMU yaw")
            if self._yaw0 is None:
                self._yaw0 = yaw
            turned = yaw - self._yaw0
            want = math.radians(prim.yaw_deg)
            remaining = want - turned
            rec.achieved_deg = math.degrees(turned)
            reached = remaining <= 0.0 if want >= 0.0 else remaining >= 0.0
            if reached:
                finished = True
                err = math.degrees(turned) - prim.yaw_deg
                note = "turn done: {:+.1f} deg (target {:+.1f}, err {:+.1f})".format(
                    math.degrees(turned), prim.yaw_deg, err)
                if abs(err) > self.lim.yaw_tol_deg:
                    rec.aborted = None   # not a failure, but say so in the report
                    note += " -- OUTSIDE the {:.1f} deg tolerance".format(
                        self.lim.yaw_tol_deg)
            elif elapsed > (prim.expected_s or prim.duration_s or 1.0) \
                    * self.lim.turn_timeout_factor:
                rec.aborted = ("turn timed out at {:+.1f} of {:+.1f} deg"
                               .format(math.degrees(turned), prim.yaw_deg))
                return self._abort(rec.aborted, t)
            else:
                # Taper near the target so the turn does not overshoot on the last
                # control cycles, but never below a floor that still moves. With
                # the floor at 0.35 of the commanded rate, one 50 Hz cycle is
                # well under a tenth of a degree, so aiming at the target rather
                # than at a tolerance band costs nothing in overshoot.
                return Output(0.0, 0.0, self._turn_wz(remaining, abs(prim.wz), elapsed),
                              RUNNING)
        elif prim.op == "stop":
            # "stop" has no duration: it means zero velocity, now. Without this,
            # `duration_s is None` fell through and the primitive never ended --
            # which only shows up on a plan whose LAST step is a stop.
            finished = True
            note = "stop"
        else:
            if prim.duration_s is not None and elapsed >= prim.duration_s:
                finished = True

        if finished:
            rec.actual_s = elapsed
            if self.i + 1 >= len(self.prims):
                return self._finish(t, note)
            self._advance(t)
            nxt = self.current.summary() if self.current else ""
            vx, vy, wz = self._ramped(t, self.current)
            return Output(vx, vy, wz, RUNNING,
                          (note + " | next: " + nxt) if note else "next: " + nxt)

        vx, vy, wz = self._ramped(t, prim)
        return Output(vx, vy, wz, RUNNING)

    # ------------------------------------------------------- turn_to / walk_to
    # Phases, in a loop until done:
    #   settle  ramp to zero, then stand obj_settle_s: the frames that count are taken
    #           standing still
    #   look    wait for a confirmed sighting of the target (percept.choose_target) in
    #           frames captured after settling; give up after obj_look_timeout_s
    #   turn    by the camera's bearing, closed loop on the IMU like any turn
    #   walk    (walk_to) at most obj_segment_m, open loop, under the camera guard
    # The bearing is measured standing and turned on the IMU; the distance is measured
    # standing and walked open loop in short pieces, then measured again. Nothing is
    # steered while moving: the robot does not turn while walking.
    def _ob_out(self, vx, wz):
        self._ob["last"] = (vx, wz)
        return Output(vx, 0.0, wz, RUNNING)

    def _object_tick(self, t, yaw, prim, rec, percept):
        """One tick of a turn_to / walk_to. Returns (Output or None, finished, note):
        an Output while it runs (an ABORTED one included); (None, True, note) when done."""
        lim = self.lim
        if self._ob is None:
            g = prim.ground or {}
            self._ob = {"phase": "settle", "t0": t, "from": (0.0, 0.0), "last": (0.0, 0.0),
                        "expect": g.get("bearing_deg"), "turns": 0, "turns_since_walk": 0}
        ob = self._ob

        if ob["phase"] == "settle":
            e = t - ob["t0"]
            if e < lim.ramp_s + lim.obj_settle_s:
                a = min(1.0, e / lim.ramp_s) if lim.ramp_s > 0 else 1.0
                return self._ob_out((1 - a) * ob["from"][0], (1 - a) * ob["from"][1]), False, None
            ob["phase"], ob["look_t"] = "look", t

        if ob["phase"] == "look":
            frames = percept.since(ob["look_t"]) if percept is not None else []
            status, det, detail = PC.choose_target(
                frames, prim.target_cls, lim, ob["expect"], need_depth=prim.op == "walk_to")
            if status != PC.CONFIRMED:
                if t - ob["look_t"] <= lim.obj_look_timeout_s:
                    return self._ob_out(0.0, 0.0), False, None
                why = {PC.STALE: "only {} camera frames in {:.0f} s standing still (is "
                                 "r1_perception running?)".format(len(frames),
                                                                  lim.obj_look_timeout_s),
                       PC.NOT_SEEN: "lost the {}: {}".format(prim.target_cls, detail),
                       PC.AMBIGUOUS: "cannot tell which {}: {}".format(prim.target_cls, detail),
                       PC.LOW_CONF: "the {} is not clear enough: {}".format(prim.target_cls,
                                                                            detail),
                       PC.NO_DEPTH: "no depth reading on the {}".format(prim.target_cls)}[status]
                rec.aborted = why
                return self._abort(why, t), False, None
            rec.looks.append({"t": round(t, 3), "bearing_deg": det["bearing_deg"],
                              "range_m": det.get("range_m"), "conf": det.get("conf")})
            rec.final = det
            return self._object_decide(t, prim, rec, det)

        if ob["phase"] == "turn":
            tr = ob["turn"]
            e = t - tr["t0"]
            if yaw is None:
                if e > 2.0:
                    rec.aborted = "no IMU yaw after 2 s -- is ~/imu publishing?"
                    return self._abort(rec.aborted, t), False, None
                return self._ob_out(0.0, 0.0), False, None
            if tr["yaw0"] is None:
                tr["yaw0"] = yaw
            turned = yaw - tr["yaw0"]
            want = math.radians(tr["target"])
            remaining = want - turned
            if (remaining <= 0.0) if want >= 0.0 else (remaining >= 0.0):
                deg = math.degrees(turned)
                rec.turns.append(deg)
                ob["turns"] += 1
                ob["turns_since_walk"] += 1
                ob["expect"] = PC.bearing_after_turn(tr["target"], deg)
                ob.update(phase="settle", t0=t, **{"from": ob["last"]})
                return self._ob_out(*ob["last"]), False, None
            if e > tr["timeout"]:
                rec.aborted = "turn toward the {} timed out at {:+.1f} of {:+.1f} deg".format(
                    prim.target_cls, math.degrees(turned), tr["target"])
                return self._abort(rec.aborted, t), False, None
            return self._ob_out(0.0, self._turn_wz(remaining, tr["mag"], e)), False, None

        # walk
        w = ob["walk"]
        e = t - w["t0"]
        why = self._guard(t, percept)
        if why:
            rec.segments.append((w["vx"], e, w["vx"] * max(0.0, e - lim.ramp_s / 2.0)))
            rec.aborted = why
            return self._abort(why, t), False, None
        if e >= w["dur"]:
            est = w["vx"] * w["dur"]
            rec.segments.append((w["vx"], e, est))
            ob["expect"], _ = PC.after_walk(w["b"], w["r"], est)
            ob["turns_since_walk"] = 0
            ob.update(phase="settle", t0=t, **{"from": ob["last"]})
            return self._ob_out(*ob["last"]), False, None
        a = min(1.0, e / lim.ramp_s) if lim.ramp_s > 0 else 1.0
        return self._ob_out(a * w["vx"], 0.0), False, None

    def _object_turn(self, t, prim, bearing):
        lim = self.lim
        # The rate a plan turn of this size would get (translate(): slowed to last
        # min_primitive_s, never below turn_min_wz). A correction smaller than ~17 deg
        # then lasts under min_primitive_s: allowed here, and only here, because the
        # robot stands still before and after it (stage E plan, section 2).
        floor = lim.turn_min_wz if lim.turn_min_wz > 0 else 0.1
        mag = min(lim.cruise_wz, max(floor, math.radians(abs(bearing)) / lim.min_primitive_s))
        exp = expected_turn_s(bearing, mag, lim.ramp_s, lim.turn_min_wz)
        self._ob.update(phase="turn", turn={"target": bearing, "t0": t, "yaw0": None, "mag": mag,
                                            "timeout": exp * lim.turn_timeout_factor + 1.0})
        out = self._ob_out(0.0, 0.0)
        out.event = "turn {:+.1f} deg toward the {}".format(bearing, prim.target_cls)
        return out, False, None

    def _object_decide(self, t, prim, rec, det):
        lim, ob = self.lim, self._ob
        b = det["bearing_deg"]
        if prim.op == "turn_to":
            if abs(b) <= lim.obj_turn_tol_deg:
                return None, True, "facing the {}: {:+.1f} deg off by the camera".format(
                    prim.target_cls, b)
            if ob["turns"] > lim.obj_max_corrections:
                rec.note = "still {:+.1f} deg off after {} turns (tolerance {:.0f})".format(
                    b, ob["turns"], lim.obj_turn_tol_deg)
                return None, True, rec.note
            return self._object_turn(t, prim, b)
        r = det["range_m"]
        if abs(b) > lim.obj_align_tol_deg:
            if ob["turns_since_walk"] > lim.obj_max_corrections:
                rec.aborted = "could not face the {} ({:+.1f} deg off after {} turns)".format(
                    prim.target_cls, b, ob["turns_since_walk"])
                return self._abort(rec.aborted, t), False, None
            return self._object_turn(t, prim, b)
        left = r - prim.stop_at_m
        if left <= lim.obj_arrive_tol_m:
            # Arrived. A target off to one side drifts outward as the range closes
            # (-2 deg at 3 m is -6 deg at 1 m), so face it once more before finishing.
            if abs(b) > lim.obj_turn_tol_deg and ob["turns_since_walk"] <= lim.obj_max_corrections:
                return self._object_turn(t, prim, b)
            return None, True, "at the {}: {:.2f} m from it by depth (asked {:.2f})".format(
                prim.target_cls, r, prim.stop_at_m)
        if len(rec.segments) >= lim.obj_max_segments:
            rec.aborted = "still {:.2f} m from stopping after {} walks".format(
                left, len(rec.segments))
            return self._abort(rec.aborted, t), False, None
        # Equal pieces, none longer than obj_segment_m: 2.16 m is 3 x 0.72, not 1 + 1
        # and a 0.16 m remainder the robot cannot walk (the shortest walk is 0.2 m).
        seg = left / math.ceil(left / lim.obj_segment_m - 1e-9)
        vx = max(WALK_MIN_VX, min(lim.obj_vx, seg / lim.min_primitive_s))
        ob.update(phase="walk", walk={"vx": vx, "dur": seg / vx, "t0": t, "b": b, "r": r})
        out = self._ob_out(0.0, 0.0)
        out.event = "walk {:.2f} m at {:.2f} m/s toward the {} ({:.2f} m away, {:+.1f} deg)".format(
            seg, vx, prim.target_cls, r, b)
        return out, False, None

    # ---------------------------------------------------------------- endings
    def _finish(self, t, note=None):
        """Ramp to zero and hold it for settle_s before declaring DONE, so the
        process does not exit while the robot is still moving and let the deadman
        do the stopping."""
        if self._t_settle is None:
            self._t_settle = t
        if t - self._t_settle < self.lim.settle_s:
            return Output(0.0, 0.0, 0.0, RUNNING, note)
        self.state = DONE
        return Output(0.0, 0.0, 0.0, DONE, note or "plan complete")

    def _abort(self, reason, t):
        self.state = ABORTED
        self.abort_reason = reason
        if self.records and self.records[-1].aborted is None:
            self.records[-1].aborted = reason
            if self._t_prim is not None:
                self.records[-1].actual_s = max(0.0, t - self._t_prim)
        return Output(0.0, 0.0, 0.0, ABORTED, "ABORT: " + reason)

    def interrupt(self, reason, t=None):
        """An abort from outside the control loop (the operator's Ctrl-C). `t` is
        the executor's own clock; without it, the time of the last step() is used.
        Stage C found the caller passing 0.0 here, which recorded the interrupted
        step's actual_s as 0 (or negative) and made an interrupted walk report
        having walked nowhere."""
        if self.state in (DONE, ABORTED):
            return None
        if t is None:
            t = self._t_last if self._t_last is not None else (self._t_prim or 0.0)
        return self._abort(reason, t)

    def report(self):
        return {"executed": [r.to_dict() for r in self.records],
                "final_state": self.state,
                "abort_reason": self.abort_reason}
