#!/usr/bin/env python3
"""Static checks of the dance (routine.py) before any training (V3 plan 01, T1).

  1 joint limits     every joint inside Isaac's soft limits (90 % of the URDF range)
  2 self-collision   arm meshes keep >= --min-gap from the torso, head, legs and each other
  3 camera           no point of the arms projects into the person-detection region of the
                     left head camera, for a spread of plausible camera mounts
  4 bar boundaries   every bar starts and ends in the default pose at rest
  5 rates            joint speed <= half the URDF velocity limit; acceleration <= --max-acc
  6 feet             the soles do not move (the pelvis is solved for that, this checks it)
  7 loop             the last frame equals the first, so the loop can repeat

Plain numpy on the URDF (urdf_fk.py): runs anywhere, including run_tests.sh.

Why check 3: perception reads a person's distance from the height of their box. An arm in
front of the lens cuts the box short, and a shorter box reads FARTHER -- the unsafe
direction. The dance must keep the arms out of the region the reflex looks at.
"""
from __future__ import annotations

import argparse
import itertools
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import routine  # noqa: E402
import urdf_fk  # noqa: E402

# Left head camera, from the stereo calibration the user supplied on 2026-10-23
# (OpenCV, 544x448, stereo RMS 0.29 px). The distortion is small enough that the
# projection below uses all five coefficients without a fisheye model.
IMG_W, IMG_H = 544, 448
K = np.array([[124.82108004, 0.0, 262.89587511], [0.0, 124.79267272, 216.99502464], [0.0, 0.0, 1.0]])
D = np.array([0.00557527, 0.00442321, 0.00072874, -0.00544967, -0.00097839])  # k1 k2 p1 p2 k3

# perception.yaml proximity_center_region: x0, y0, x1, y1 as fractions of the image
REGION = (0.25, 0.25, 0.75, 1.0)

# The lens is not in the URDF. Its nominal place in head_yaw_link: ~8 cm in front of the
# head-yaw axis (the head mesh's front face is at +9.1 cm), 3 cm to the left (half the
# 5.96 cm baseline), and 4.85 cm up, which puts it at the measured 1.20 m above the floor
# in the default pose. Check 3 must hold for every mount in this spread that agrees with
# what R1's own camera showed on 2026-10-23: in those frames the horizon is mid-image and
# the hanging robot's own hands are not in view, so a mount from which the DEFAULT pose
# already puts a hand in the region is not this robot's and is left out (and reported).
CAM_X = (0.05, 0.08, 0.10)
CAM_Y = (0.0298,)
CAM_Z = (0.03, 0.0485, 0.07)
# Pitch: the R1 frames put the horizon near mid-image (about level). Covered: 10 deg up to
# 10 deg down. Not covered: steeper -- at 20 deg down the default pose's hands already sit
# on the region's bottom edge. The B0 capture measures the real tilt from the floor; if it
# is outside this range, extend the list and rerun.
CAM_PITCH_DOWN_DEG = (-10.0, 0.0, 5.0, 10.0)

ARM_LINKS = ["{}_{}".format(s, l) for s in ("left", "right")
             for l in ("shoulder_roll_link", "shoulder_yaw_link", "elbow_link", "wrist_roll_link")]
BODY_LINKS = ["pelvis_link", "waist_roll_link", "waist_yaw_link", "head_pitch_link", "head_yaw_link",
              "left_hip_pitch_link", "right_hip_pitch_link", "left_hip_roll_link", "right_hip_roll_link",
              "left_hip_yaw_link", "right_hip_yaw_link", "left_knee_link", "right_knee_link"]
# links that touch by design (a joint between them), never checked against each other
ADJACENT = {("shoulder_roll_link", "waist_yaw_link")}


def project(points_cam):
    """OpenCV pinhole + Brown distortion; points (N,3) in the camera frame -> (N,2) px."""
    x, y = points_cam[:, 0] / points_cam[:, 2], points_cam[:, 1] / points_cam[:, 2]
    k1, k2, p1, p2, k3 = D
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)


def camera_pose(fk, x, y, z, pitch_down_deg):
    """(R, p) of the camera in the world: optical axis = head forward, tilted down."""
    R_head, p_head = fk["head_yaw_link"]
    p = p_head + R_head @ np.array([x, y, z])
    a = math.radians(pitch_down_deg)
    tilt = urdf_fk.axis_angle_matrix((0.0, 1.0, 0.0), a)       # +a about y = nose down
    # columns: camera x (image right) = -head y, camera y (image down) = -head z, z = head x
    R = R_head @ tilt @ np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return R, p


def arm_in_region(pts, R, p):
    """How many of the world points `pts` the camera (R, p) sees inside REGION."""
    cam = (pts - p) @ R                       # world -> camera frame
    front = cam[:, 2] > 0.02
    # beyond ~75 deg off-axis a point is outside this lens's image anyway
    front &= np.degrees(np.arctan2(np.hypot(cam[:, 0], cam[:, 1]), cam[:, 2])) < 75.0
    if not front.any():
        return 0
    return int(in_region(project(cam[front])).sum())


def in_region(px):
    x0, y0, x1, y1 = REGION
    return ((px[:, 0] >= x0 * IMG_W) & (px[:, 0] <= x1 * IMG_W)
            & (px[:, 1] >= y0 * IMG_H) & (px[:, 1] <= y1 * IMG_H))


def world_points(robot, fk, links, cache):
    out = {}
    for link in links:
        if link not in cache:
            cache[link] = robot.mesh_points(link)
        R, p = fk[link]
        out[link] = cache[link] @ R.T + p
    return out


def run(fps=25.0, min_gap=0.02, max_acc=60.0, verbose=True):
    robot = urdf_fk.Robot()
    solver = routine.PelvisSolver(robot)
    t, pos, quat, q = routine.frames(fps=fps, loops=1)
    names = routine.ART_JOINTS
    results = []

    def report(ok, label, detail):
        results.append((ok, label, detail))
        if verbose:
            print("{} {:<16} {}".format("PASS" if ok else "FAIL", label, detail))

    # 1 limits (Isaac clips the reference into the soft limits; a dance that leans on them
    #   would be clipped and look different from what was designed)
    worst = (1e9, None)
    for k, name in enumerate(names):
        lo, hi = robot.limits[name]
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * 0.9
        margin = min(q[:, k].min() - (mid - half), (mid + half) - q[:, k].max())
        worst = min(worst, (margin, name))
    report(worst[0] >= 0.0, "joint limits", "closest to a soft limit: {} ({:+.3f} rad of room)".format(
        worst[1], worst[0]))

    # 2 + 3 per frame
    cache = {}
    gap_min, gap_where = 1e9, None
    cam_hits, cam_worst = 0, None
    mounts = list(itertools.product(CAM_X, CAM_Y, CAM_Z, CAM_PITCH_DOWN_DEG))
    fk0 = robot.fk(routine.DEFAULT, base_p=np.array([0.0, 0.0, routine.PELVIS_Z]))
    arm0 = world_points(robot, fk0, ARM_LINKS, cache)
    pts0 = np.concatenate([arm0[l] for l in ARM_LINKS if "shoulder_roll" not in l])
    excluded = [m for m in mounts if arm_in_region(pts0, *camera_pose(fk0, *m))]
    mounts = [m for m in mounts if m not in excluded]
    for i in range(len(t)):
        qi = dict(zip(names, q[i]))
        fk = robot.fk(qi, base_p=pos[i])
        arm = world_points(robot, fk, ARM_LINKS, cache)
        body = world_points(robot, fk, BODY_LINKS, cache)
        for a_link, a_pts in arm.items():
            a_side, a_part = a_link.split("_", 1)
            others = [(l, p) for l, p in body.items() if (a_part, l) not in ADJACENT]
            others += [(l, p) for l, p in arm.items() if not l.startswith(a_side)]
            for o_link, o_pts in others:
                d = np.sqrt(((a_pts[:, None, :] - o_pts[None, :, :]) ** 2).sum(-1)).min()
                if d < gap_min:
                    gap_min, gap_where = d, (t[i], a_link, o_link)
        pts = np.concatenate([arm[l] for l in ARM_LINKS if "shoulder_roll" not in l])
        for m in mounts:
            n_hit = arm_in_region(pts, *camera_pose(fk, *m))
            if n_hit:
                cam_hits += 1
                if cam_worst is None:
                    cam_worst = (t[i], m, n_hit)
    report(gap_min >= min_gap, "self-collision",
           "smallest gap {:.3f} m (t={:.2f} s, {} vs {})".format(gap_min, *gap_where))
    pitches = sorted({m[3] for m in mounts})
    report(cam_hits == 0 and bool(mounts), "camera region",
           ("no arm point in the detection region: {} mounts (pitch {} deg down) x {} frames; {} mounts "
            "left out because the default pose would already show a hand there").format(
               len(mounts), "/".join("{:g}".format(p) for p in pitches), len(t), len(excluded))
           if cam_hits == 0 else "{} frame-mounts hit; first: t={:.2f} s mount {} ({} points)".format(
               cam_hits, *cam_worst))

    # 4 bar boundaries: default pose, zero velocity
    default = np.array([routine.DEFAULT[n] for n in names])
    dt = 1e-3
    worst_pose = worst_vel = 0.0
    for start, (_, length, _) in zip(routine.BAR_STARTS, routine.BARS):
        for tb in (start, start + length - 1e-9):
            qa = np.array([routine.joints_at(tb)[n] for n in names])
            qb = np.array([routine.joints_at(tb + dt)[n] for n in names])
            worst_pose = max(worst_pose, np.abs(qa - default).max())
            worst_vel = max(worst_vel, np.abs(qb - qa).max() / dt)
    report(worst_pose < 1e-6 and worst_vel < 1e-3, "bar boundaries",
           "max offset from default {:.1e} rad, max speed {:.1e} rad/s".format(worst_pose, worst_vel))

    # 5 rates
    fine_t, _, _, fine_q = routine.frames(fps=200.0, loops=1)
    vel = np.gradient(fine_q, fine_t, axis=0)
    acc = np.gradient(vel, fine_t, axis=0)
    vmax_ratio, vwho = 0.0, None
    for k, name in enumerate(names):
        vlim = robot.velocity_limits[name]
        r = np.abs(vel[:, k]).max() / vlim
        if r > vmax_ratio:
            vmax_ratio, vwho = r, name
    amax = np.abs(acc).max()
    report(vmax_ratio <= 0.5 and amax <= max_acc, "rates",
           "fastest {} at {:.0%} of its limit; max acceleration {:.1f} rad/s^2 (allowed {:g})".format(
               vwho, vmax_ratio, amax, max_acc))

    # 6 feet
    drift = 0.0
    for i in range(len(t)):
        fk = robot.fk(dict(zip(names, q[i])), base_p=pos[i])
        feet = np.array([fk["left_ankle_roll_link"][1], fk["right_ankle_roll_link"][1]])
        drift = max(drift, np.abs(feet - solver.feet0).max())
    report(drift < 0.005, "feet", "soles move at most {:.4f} m (allowed 0.005)".format(drift))

    # 7 loop
    jump = max(np.abs(q[-1] - q[0]).max(), np.abs(pos[-1] - pos[0]).max())
    report(jump < 1e-6, "loop", "last frame vs first: {:.1e}".format(jump))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fps", type=float, default=25.0, help="frames checked per second of the dance")
    ap.add_argument("--min-gap", type=float, default=0.02)
    ap.add_argument("--max-acc", type=float, default=60.0)
    args = ap.parse_args()
    results = run(args.fps, args.min_gap, args.max_acc)
    failed = [r for r in results if not r[0]]
    print("{} of {} checks passed".format(len(results) - len(failed), len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
