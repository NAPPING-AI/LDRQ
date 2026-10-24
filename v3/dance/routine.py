"""The V3 dance: a 10 s loop of four bars, both feet on the floor throughout.

  bar 1  3 s  right arm out to the side, forearm up, waves; the waist turns a little right
  bar 2  3 s  the same with the left arm
  bar 3  2 s  both arms up and out to the sides; the knees dip twice
  bar 4  2 s  standing still in the default pose

Every bar starts and ends in R1's default standing pose (assets/r1/r1.py) with zero
joint velocity: a bar boundary is where the bridge may hand the robot back to the
walking policy without a jump (V3 plan 01 §1.4), so a normal end waits at most 3 s. The
hands never come in front of the camera; check_routine.py proves that against the
calibrated intrinsics.

The output is the generalized-coordinate CSV that BeyondMimic's motion converter reads
(v3/dance/csv_to_npz_r1.py): one row per frame, root position xyz, root quaternion xyzw,
then the 26 joints in ART_JOINTS order. The root is the pelvis. It stays upright and only
translates, so that the soles stay where they are while the knees dip -- solved here
with the URDF's own leg kinematics (PelvisSolver), not left to the policy.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import urdf_fk  # noqa: E402

# Isaac Lab's articulation order for R1 (deploy/interface/policy_interface.json
# "articulation"); the motion npz and the policy observation use this order.
ART_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_roll_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_yaw_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "head_pitch_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_knee_joint",
    "right_knee_joint", "head_yaw_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]

# assets/r1/r1.py init_state (Unitree's HOME_KEYFRAME); pelvis 0.731 m above the floor.
DEFAULT = {j: 0.0 for j in ART_JOINTS}
DEFAULT.update({
    "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3, "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.35, "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18, "right_shoulder_roll_joint": -0.18,
    "left_elbow_joint": 0.87, "right_elbow_joint": 0.87,
})
PELVIS_Z = 0.731

# Arm paths, for the left arm (the right arm mirrors: roll and yaw change sign). Each
# joint is a list of moves (t0, t1, target), seconds into the bar's rise, applied one
# after another from DEFAULT with a quintic blend; the bar's fall is its rise played
# backwards. The ORDER keeps the hand out of the camera: the arm swings out to the side
# and straightens, turns about its own axis while straight (the hand barely moves), and
# only then bends the forearm up. Bending before turning swings the hand across the lens.
WAVE_PATH = {
    "shoulder_pitch": [(0.0, 0.4, 0.0)],
    "shoulder_roll": [(0.0, 0.4, 1.3)],
    "elbow": [(0.0, 0.3, 1.5), (0.6, 1.0, 0.0)],
    "shoulder_yaw": [(0.25, 0.65, 1.6)],
}
WAVE_ELBOW_AMP = 0.35       # rad, the forearm swings about upright
WAVE_HZ = 1.5
WAVE_WAIST_YAW = 0.15       # rad, the torso turns toward the waving arm
UP_PATH = {
    "shoulder_pitch": [(0.0, 0.35, 0.0)],
    "elbow": [(0.0, 0.25, 1.3)],
    "shoulder_roll": [(0.1, 0.6, 2.0)],
}
DIP_KNEE = 0.2              # rad of extra knee bend at the bottom of a dip
DIP_TIMES = ((0.6, 1.0), (1.0, 1.4))    # within bar 3, seconds from the bar's start

MIRROR_SIGN = {"shoulder_pitch": 1.0, "shoulder_roll": -1.0, "shoulder_yaw": -1.0, "elbow": 1.0}

# (what, length s, rise s): the bar's pose is reached after `rise` and left `rise` before
# its end; in between it holds (and waves, or dips).
BARS = (("wave_right", 3.0, 1.0), ("wave_left", 3.0, 1.0), ("arms_up_dip", 2.0, 0.6), ("still", 2.0, 0.6))
LOOP_S = sum(b[1] for b in BARS)
BAR_STARTS = tuple(sum(b[1] for b in BARS[:i]) for i in range(len(BARS)))


def quintic(x):
    """0 -> 1 with zero velocity and acceleration at both ends."""
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def envelope(tau, length, rise):
    """0 at the bar's edges, 1 in its middle; tau = seconds into the bar."""
    return quintic(tau / rise) * quintic((length - tau) / rise)


def window(tau, t0, t1):
    """sin^2 bump on [t0, t1]: zero value and slope at both ends."""
    if tau <= t0 or tau >= t1:
        return 0.0
    return math.sin(math.pi * (tau - t0) / (t1 - t0)) ** 2


def _follow(start, moves, s):
    """Value at s seconds into the rise of a joint that starts at `start`."""
    value = start
    for t0, t1, target in moves:
        if s > t0:
            value += quintic((s - t0) / (t1 - t0)) * (target - value)
    return value


def _arm(q, side, path, tau, length):
    """One arm along `path` (given for the left arm), tau seconds into the bar."""
    s = min(tau, length - tau)                   # the fall is the rise backwards
    for part, moves in path.items():
        name = "{}_{}_joint".format(side, part)
        sign = 1.0 if side == "left" else MIRROR_SIGN[part]
        q[name] = _follow(DEFAULT[name], [(a, b, v * sign) for a, b, v in moves], s)


def bar_at(t):
    """(bar index, seconds into it) at t seconds into the dance."""
    t = t % LOOP_S
    for i in reversed(range(len(BARS))):
        if t >= BAR_STARTS[i]:
            return i, t - BAR_STARTS[i]
    return 0, t


def joints_at(t):
    """{joint: rad} at t seconds into the dance (the loop repeats every LOOP_S)."""
    bar, tau = bar_at(t)
    what, length, rise = BARS[bar]
    q = dict(DEFAULT)
    e = envelope(tau, length, rise)
    hold = window(tau, rise, length - rise)
    if what in ("wave_right", "wave_left"):
        side = "right" if what == "wave_right" else "left"
        _arm(q, side, WAVE_PATH, tau, length)
        wave = WAVE_ELBOW_AMP * math.sin(2.0 * math.pi * WAVE_HZ * (tau - rise)) * hold
        q["{}_elbow_joint".format(side)] += wave
        q["waist_yaw_joint"] = (-WAVE_WAIST_YAW if side == "right" else WAVE_WAIST_YAW) * e
    elif what == "arms_up_dip":
        _arm(q, "left", UP_PATH, tau, length)
        _arm(q, "right", UP_PATH, tau, length)
        dip = DIP_KNEE * sum(window(tau, a, b) for a, b in DIP_TIMES)
        for side in ("left", "right"):
            # hip + knee + ankle = 0 keeps the sole parallel to the (upright) pelvis
            q["{}_knee_joint".format(side)] += dip
            q["{}_hip_pitch_joint".format(side)] -= dip / 2.0
            q["{}_ankle_pitch_joint".format(side)] -= dip / 2.0
    return q


class PelvisSolver:
    """Plants the feet: where both ankle_roll_link frames are in DEFAULT, they stay.

    An upright pelvis can only translate, which fixes the mean of the two feet. Bending
    the knees also moves the feet a few mm toward or away from each other (the leg's
    joint axes are offset), so each leg's hip roll is corrected too, with the ankle roll
    turning back by the same angle to keep the sole flat."""

    def __init__(self, robot=None):
        self.robot = robot or urdf_fk.Robot()
        self.feet0 = self._feet(DEFAULT, np.array([0.0, 0.0, PELVIS_Z]))

    def _feet(self, q, p):
        out = self.robot.fk(q, base_p=p)
        return np.array([out["left_ankle_roll_link"][1], out["right_ankle_roll_link"][1]])

    def __call__(self, q):
        """(corrected joints, pelvis position) for joints q."""
        q = dict(q)
        for _ in range(3):                       # Newton on each leg's hip roll
            rel = self._feet(q, np.zeros(3))
            p = (self.feet0 - rel).mean(axis=0)
            err = rel + p - self.feet0           # (2, 3): left, right
            if np.abs(err[:, 1]).max() < 1e-5:
                break
            h = 1e-4
            for k, side in enumerate(("left", "right")):
                probe = dict(q)
                probe["{}_hip_roll_joint".format(side)] += h
                probe["{}_ankle_roll_joint".format(side)] -= h
                dy = (self._feet(probe, np.zeros(3))[k, 1] - rel[k, 1]) / h
                step = -(err[k, 1] - err[:, 1].mean()) / dy
                q["{}_hip_roll_joint".format(side)] += step
                q["{}_ankle_roll_joint".format(side)] -= step
        rel = self._feet(q, np.zeros(3))
        return q, (self.feet0 - rel).mean(axis=0)


def frames(fps=50.0, loops=3):
    """(t, root_pos (N,3), root_quat_xyzw (N,4), joints (N,26) in ART_JOINTS order)."""
    n = int(round(loops * LOOP_S * fps)) + 1      # include the final frame = the first pose
    solver = PelvisSolver()
    t = np.arange(n) / fps
    pos = np.zeros((n, 3))
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    q = np.zeros((n, len(ART_JOINTS)))
    for i, ti in enumerate(t):
        qi, pos[i] = solver(joints_at(ti))
        q[i] = [qi[j] for j in ART_JOINTS]
    return t, pos, quat, q


def write_csv(path, fps=50.0, loops=3):
    t, pos, quat, q = frames(fps, loops)
    rows = np.hstack([pos, quat, q])
    np.savetxt(str(path), rows, delimiter=",", fmt="%.6f")
    return len(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/v3/dance/routine_wave.csv")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--loops", type=int, default=3)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_csv(out, args.fps, args.loops)
    print("{}: {} frames, {:.1f} s at {:g} fps ({} loops of {:g} s)".format(
        out, n, (n - 1) / args.fps, args.fps, args.loops, LOOP_S))


if __name__ == "__main__":
    main()
