# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V3 T3: handing the robot between the dance policy and v2, as the bridge does.

Both policies run every control step on the same robot, the way the V3 runner runs them
on the Orin: v2 on its 85 x 5 history with the velocity command the conductor gives it,
the dance policy on its 137-float observation built the way the runner builds it
(v3/dance/dance_obs.py: the reference at the dance tick, its heading turned to the
robot's, the IMU, the joints, the last APPLIED action). Only the active policy's action
reaches the robot. Two scenarios, each over many envs with random times:

  normal     v2 steps in place for 1-3 s (random) -> the dance starts (tick 0) -> it is
             asked to end after 5-20 s (random) and hands back at the next bar boundary,
             as conductor.py does -> v2 stands (0, 0, 0) for 6 s. Covers the start from
             a random point of v2's gait and the planned end.
  emergency  v2 stands 1.5 s -> dance -> at a random time 0.3-10 s into it (any phase)
             v2 takes over on that step with (-0.3, 0, 0) for 1.5 s (the reflex), then
             (0, 0, 0) for 10 s.
  baseline   the same retreat and stand with no dance at all: v2 stands 1.8-11.5 s, then
             (-0.3, 0, 0) for 1.5 s and (0, 0, 0) for 10 s. What v2 does on its own, to
             tell the handover's share of the falls, the distance and the drift.

Measured per env: a fall (pelvis below 0.45 m or tilt above 60 deg) from SETTLE_S on --
an env down before that has fallen over from the reset that writes it into the default
pose (it happens to ~1 in 100 envs, v2 alone just the same), is left out and counted as
`spawn_excluded`; for an
emergency, how far it stepped back (pelvis, along the heading it had at the switch, from
the switch to the end of the retreat) and its drift over the 10 s stand; the dance tick
at the switch, for the worst phases.

Terminations and resets are off (a fall is measured, not reset), the episode never times
out, and the reference never resamples: the dance's command term is pinned to the tick.
Startup randomization is the dance task's, except motor strength: --kp_scale is applied
exactly (the robot runs at 1.3).

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p v3/sim/eval_handover.py --headless \\
        --dance logs/rsl_rl/r1_dance_v3/<run>/model_<n>.pt \\
        --motion outputs/v3/dance/routine_wave.npz [--num_envs 1024 --batches 4 --kp_scale 1.3]
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "v3" / "dance"))

parser = argparse.ArgumentParser(description="V3 T3: dance <-> v2 handovers in simulation.")
parser.add_argument("--dance", type=str, required=True, help="dance checkpoint (model_<n>.pt)")
parser.add_argument("--v2", type=str, default=str(_ROOT / "models" / "v2_drift_ft" / "policy.pt"))
parser.add_argument("--motion", type=str, default=str(_ROOT / "outputs" / "v3" / "dance" / "routine_wave.npz"))
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--batches", type=int, default=4, help="runs per scenario = num_envs x batches")
parser.add_argument("--scenarios", type=str, default="normal,emergency,baseline")
parser.add_argument("--kp_scale", type=float, default=1.3)
parser.add_argument("--retreat_vx", type=float, default=-0.3)
parser.add_argument("--retreat_s", type=float, default=1.5)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", type=str, default="", help="write the summary JSON here")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import r1_tracking  # noqa: E402  -- registers Isaac-Dance-R1-Play-v3
import routine  # noqa: E402

DT = 0.02
BAR_TICKS = [int(round(b[1] / DT)) for b in routine.BARS]
LOOP = sum(BAR_TICKS)
BOUNDARIES = [sum(BAR_TICKS[:i]) for i in range(len(BAR_TICKS))]
FALL_Z, FALL_TILT = 0.45, math.radians(60.0)
SETTLE_S = 1.0          # every scenario's first switch is later than this


# -------------------------------------------------------------- quaternions (w, x, y, z), batched
def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def qyaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_q(yaw):
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], -1)


def qmat2(q):
    """First two columns of the rotation matrix, flattened row by row: (N, 6)."""
    w, x, y, z = q.unbind(-1)
    m = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (x * z - w * y), 2 * (y * z + w * x)], -1)
    return m


class V2Obs(object):
    """v2's observation: 85-float frames, 5 of them, term-major, oldest first."""
    WIDTHS = (3, 3, 3, 26, 26, 24)

    def __init__(self, n, device):
        self.hist = None
        self.n, self.device = n, device

    def __call__(self, frame, reset_mask=None):
        if self.hist is None:
            self.hist = frame.unsqueeze(1).repeat(1, 5, 1)
        else:
            self.hist = torch.cat([self.hist[:, 1:], frame.unsqueeze(1)], 1)
            if reset_mask is not None and reset_mask.any():
                self.hist[reset_mask] = frame[reset_mask].unsqueeze(1).repeat(1, 5, 1)
        out, off = [], 0
        for w in self.WIDTHS:
            out.append(self.hist[:, :, off:off + w].reshape(self.n, -1))
            off += w
        return torch.cat(out, 1)


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg = parse_env_cfg("Isaac-Dance-R1-Play-v3", device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.commands.motion.motion_file = str(Path(args_cli.motion).resolve())
    env_cfg.episode_length_s = 1.0e6
    for name in ("time_out", "anchor_pos", "anchor_ori", "ee_body_pos"):
        setattr(env_cfg.terminations, name, None)
    env_cfg.events.actuator_gains = None          # motor strength: --kp_scale, exactly
    env_cfg.events.push_robot = None
    env = gym.make("Isaac-Dance-R1-Play-v3", cfg=env_cfg)
    wenv = RslRlVecEnvWrapper(env)
    uw = env.unwrapped
    dev = uw.device
    robot = uw.scene["robot"]
    cmd = uw.command_manager.get_term("motion")
    act_term = uw.action_manager.get_term("joint_pos")
    n = uw.num_envs
    pelvis = robot.body_names.index("pelvis_link")

    agent_cfg = r1_tracking.R1DancePPORunnerCfg()
    runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=None, device=dev)
    runner.load(args_cli.dance)
    dance_pi = runner.get_inference_policy(device=dev)
    v2_pi = torch.jit.load(args_cli.v2, map_location=dev).eval()

    if args_cli.kp_scale != 1.0:
        for act in robot.actuators.values():
            act.stiffness *= args_cli.kp_scale
            robot.write_joint_stiffness_to_sim(act.stiffness, joint_ids=act.joint_indices)

    mot = np.load(args_cli.motion)
    ref_jp = torch.tensor(mot["joint_pos"][:LOOP], device=dev, dtype=torch.float32)
    ref_jv = torch.tensor(mot["joint_vel"][:LOOP], device=dev, dtype=torch.float32)
    bodies = [str(b) for b in mot["body_names"]]
    ref_q = torch.tensor(mot["body_quat_w"][:LOOP, bodies.index("pelvis_link")], device=dev,
                         dtype=torch.float32)

    def dance_obs(tick, yaw0, last_act):
        k = torch.clamp(tick, min=0) % LOOP
        q_robot = robot.data.body_quat_w[:, pelvis]
        q_ref = qmul(yaw_q(yaw0), ref_q[k])
        rel = qmul(qconj(q_robot), q_ref)
        return torch.cat([ref_jp[k], ref_jv[k], qmat2(rel), robot.data.root_ang_vel_b,
                          robot.data.joint_pos - robot.data.default_joint_pos,
                          robot.data.joint_vel - robot.data.default_joint_vel, last_act], 1)

    def v2_frame(vcmd, last_act):
        return torch.cat([robot.data.root_ang_vel_b, robot.data.projected_gravity_b, vcmd,
                          robot.data.joint_pos - robot.data.default_joint_pos,
                          robot.data.joint_vel - robot.data.default_joint_vel, last_act], 1)

    def reset_all():
        env.reset()
        root = robot.data.default_root_state.clone()
        root[:, :3] += uw.scene.env_origins
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                       robot.data.default_joint_vel.clone())
        act_term.reset()
        cmd.time_steps[:] = 0
        uw.scene.write_data_to_sim()
        uw.sim.forward()
        uw.scene.update(0.0)

    results = {}
    checked_obs = None
    for scen in [s for s in args_cli.scenarios.split(",") if s]:
        rows = []
        for b in range(args_cli.batches):
            reset_all()
            g = torch.Generator(device="cpu").manual_seed(args_cli.seed * 100 + b)
            if scen == "normal":
                t_start = 1.0 + 2.0 * torch.rand(n, generator=g)
                t_ask_end = t_start + 5.0 + 15.0 * torch.rand(n, generator=g)
                t_total = float(t_ask_end.max()) + 3.0 + 6.0
            else:
                # baseline: the dance "starts" after the switch, i.e. never
                t_switch = 1.5 + 0.3 + 9.7 * torch.rand(n, generator=g)
                t_start = torch.full((n,), 1.5) if scen == "emergency" else torch.full((n,), 1.0e9)
                t_total = float(t_switch.max()) + args_cli.retreat_s + 10.0
            t_start, = (x.to(dev) for x in (t_start,))
            t_ask_end = t_ask_end.to(dev) if scen == "normal" else None
            t_switch = t_switch.to(dev) if scen != "normal" else None

            policy = torch.zeros(n, dtype=torch.long, device=dev)     # 0 walk, 1 dance
            tick = torch.full((n,), -1, dtype=torch.long, device=dev)
            ending = torch.zeros(n, dtype=torch.bool, device=dev)
            done_dance = torch.zeros(n, dtype=torch.bool, device=dev)
            yaw0 = torch.zeros(n, device=dev)
            last_act = torch.zeros(n, 24, device=dev)
            v2obs = V2Obs(n, dev)
            fell = torch.zeros(n, dtype=torch.bool, device=dev)
            spawn_fail = torch.zeros(n, dtype=torch.bool, device=dev)
            t_fell = torch.full((n,), float("nan"), device=dev)
            fell_policy = torch.full((n,), -1, dtype=torch.long, device=dev)
            t_end_dance = torch.full((n,), float("nan"), device=dev)
            sw_tick = torch.full((n,), -1, dtype=torch.long, device=dev)
            sw_pos = torch.zeros(n, 2, device=dev)
            sw_yaw = torch.zeros(n, device=dev)
            back = torch.full((n,), float("nan"), device=dev)
            stand_pos = torch.full((n, 2), float("nan"), device=dev)
            drift = torch.full((n,), float("nan"), device=dev)
            steps = int(math.ceil(t_total / DT))
            with torch.inference_mode():
                for k in range(steps):
                    t = k * DT
                    q_robot = robot.data.body_quat_w[:, pelvis]
                    pos = robot.data.body_pos_w[:, pelvis, :2] - uw.scene.env_origins[:, :2]
                    # --- the conductor's decisions for this step
                    start = (policy == 0) & ~done_dance & (t >= t_start - 1e-9)
                    if scen == "emergency":
                        stop_now = (policy == 1) & (t >= t_switch - 1e-9)
                    elif scen == "baseline":   # a "switch" from standing: v2 throughout
                        stop_now = ~done_dance & (t >= t_switch - 1e-9)
                    else:
                        ending |= (policy == 1) & (t >= t_ask_end - 1e-9)
                        stop_now = torch.zeros_like(start)
                    if bool(start.any()):
                        policy[start] = 1
                        tick[start] = -1
                    tick = torch.where(policy == 1, tick + 1, torch.full_like(tick, -1))
                    at_bound = torch.zeros_like(start)
                    for bnd in BOUNDARIES:
                        at_bound |= (tick % LOOP) == bnd
                    stop_normal = (policy == 1) & ending & at_bound & (tick > 0)
                    stop = stop_now | stop_normal
                    if bool(stop.any()):
                        sw_tick[stop] = tick[stop]
                        sw_pos[stop] = pos[stop]
                        sw_yaw[stop] = qyaw(q_robot[stop])
                        t_end_dance[stop] = t
                        policy[stop] = 0
                        tick[stop] = -1
                        done_dance |= stop
                    # yaw the dance is played at: the robot's until the first dancing step
                    yaw0 = torch.where((policy == 1) & (tick > 0), yaw0, qyaw(q_robot))
                    # --- v2's velocity command
                    vcmd = torch.zeros(n, 3, device=dev)
                    if scen != "normal":
                        in_retreat = done_dance & (t < t_end_dance + args_cli.retreat_s - 1e-9)
                        vcmd[in_retreat, 0] = args_cli.retreat_vx
                    # --- both policies, every step
                    a_v2 = v2_pi(v2obs(v2_frame(vcmd, last_act)))
                    dobs = dance_obs(tick, yaw0, last_act)
                    if checked_obs is None and bool(((policy == 1) & (tick == 0)).any()):
                        # Our observation against the task's own, where they must agree:
                        # the first dancing step, reference heading 0 (the task's).
                        idx = torch.nonzero((policy == 1) & (tick == 0))[:, 0]
                        cmd.time_steps[idx] = 0
                        mine = dance_obs(tick, torch.zeros_like(yaw0), last_act)[idx]
                        theirs = uw.observation_manager.compute_group("policy")[idx]
                        checked_obs = float((mine - theirs).abs().max())
                    a_dance = dance_pi(dobs)
                    applied = torch.where((policy == 1).unsqueeze(1), a_dance, a_v2)
                    cmd.time_steps[:] = torch.clamp(tick, min=0)
                    env.step(applied)
                    last_act = applied
                    cmd.time_steps[:] = torch.clamp(tick + 1, min=0)
                    # --- falls and distances
                    z = robot.data.body_pos_w[:, pelvis, 2]
                    gz = robot.data.projected_gravity_b[:, 2]
                    tilt = torch.acos(torch.clamp(-gz, -1.0, 1.0))
                    down = (z < FALL_Z) | (tilt > FALL_TILT)
                    if t < SETTLE_S:
                        spawn_fail |= down
                        continue
                    new_fall = down & ~fell & ~spawn_fail
                    if bool(new_fall.any()):
                        t_fell[new_fall] = t
                        fell_policy[new_fall] = policy[new_fall]
                    fell |= down
                    if scen != "normal":
                        pos = robot.data.body_pos_w[:, pelvis, :2] - uw.scene.env_origins[:, :2]
                        end_retreat = done_dance & torch.isnan(back) & (t + DT >= t_end_dance + args_cli.retreat_s - 1e-9)
                        if bool(end_retreat.any()):
                            d = pos - sw_pos
                            along = d[:, 0] * torch.cos(sw_yaw) + d[:, 1] * torch.sin(sw_yaw)
                            back[end_retreat] = -along[end_retreat]
                            stand_pos[end_retreat] = pos[end_retreat]
                        fin = done_dance & torch.isnan(drift) & (t + DT >= t_end_dance + args_cli.retreat_s + 10.0 - 1e-9)
                        if bool(fin.any()):
                            drift[fin] = torch.linalg.norm(pos - stand_pos, dim=1)[fin]
            for i in range(n):
                if bool(spawn_fail[i]):
                    rows.append({"spawn_excluded": True})
                    continue
                rows.append({"fell": bool(fell[i]), "switch_tick": int(sw_tick[i]),
                             "fell_after_switch_s": None if not bool(fell[i]) or math.isnan(float(t_end_dance[i]))
                             else round(float(t_fell[i] - t_end_dance[i]), 3),
                             "fell_while": {-1: None, 0: "walk", 1: "dance"}[int(fell_policy[i])],
                             "t_fell_s": None if not bool(fell[i]) else round(float(t_fell[i]), 2),
                             "t_start_s": round(float(t_start[i]), 2) if float(t_start[i]) < 1e8 else None,
                             "back_m": None if math.isnan(float(back[i])) else float(back[i]),
                             "drift_m": None if math.isnan(float(drift[i])) else float(drift[i])})
            print("[{}] batch {}/{}: falls {}/{} ({} left out: down before {:.1f} s)".format(
                scen, b + 1, args_cli.batches, int(fell.sum()), n - int(spawn_fail.sum()),
                int(spawn_fail.sum()), SETTLE_S), flush=True)
        results[scen] = rows

    summary = {"dance": args_cli.dance, "v2": args_cli.v2, "kp_scale": args_cli.kp_scale,
               "envs": n, "batches": args_cli.batches, "obs_check_max_abs": checked_obs}
    for scen, rows in results.items():
        excluded = sum(1 for r in rows if r.get("spawn_excluded"))
        rows = [r for r in rows if not r.get("spawn_excluded")]
        ok = [r for r in rows if not r["fell"]]
        s = {"runs": len(rows), "falls": len(rows) - len(ok), "spawn_excluded": excluded}
        falls = [r for r in rows if r["fell"]]
        if falls:
            s["falls_detail"] = [{"switch_tick": r["switch_tick"], "while": r["fell_while"],
                                  "t_fell_s": r["t_fell_s"], "t_dance_start_s": r["t_start_s"],
                                  "after_switch_s": r["fell_after_switch_s"]} for r in falls][:50]
        if scen != "normal":
            backs = np.array([r["back_m"] for r in ok if r["back_m"] is not None])
            drifts = np.array([r["drift_m"] for r in ok if r["drift_m"] is not None])
            if backs.size:
                s["back_m"] = {"mean": float(backs.mean()), "min": float(backs.min()),
                               "p05": float(np.percentile(backs, 5)), "max": float(backs.max())}
                bars = {}
                for r in ok:
                    if r["back_m"] is None or r["switch_tick"] < 0:
                        continue
                    k = r["switch_tick"] % LOOP
                    bar = max(i for i, bd in enumerate(BOUNDARIES) if k >= bd)
                    bars.setdefault(routine.BARS[bar][0], []).append(r["back_m"])
                s["back_m_by_bar"] = {k: {"n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v))}
                                      for k, v in bars.items()}
            if drifts.size:
                s["stand_drift_10s_m"] = {"mean": float(drifts.mean()), "max": float(drifts.max()),
                                          "p95": float(np.percentile(drifts, 95))}
        summary[scen] = s
    text = json.dumps(summary, indent=1)
    print(text, flush=True)
    if args_cli.out:
        Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.out).write_text(text + "\n")
    env.close()


if __name__ == "__main__":
    with torch.inference_mode():      # resets too: the buffers are made in inference mode
        main()
    simulation_app.close()
