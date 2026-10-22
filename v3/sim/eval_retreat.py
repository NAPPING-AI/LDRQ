# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V3: how the v2 policy answers the proximity reflex's command.

The reflex (r1_hw_bridge, V3) replaces whatever velocity command the robot is
walking on with (retreat_vx, 0, 0) for retreat_s, then with zero for good (the
robot stands, stepping in place). This script does the same to the policy in
simulation, from a set of starting motions, and measures what the bridge
cannot know in advance:

- overshoot: how far the robot keeps going forward after the switch (the
  margin the stop zone must leave);
- reverse_s: time from the switch until the body moves backward (> 0.05 m/s);
- net: displacement along the starting heading from the switch to the end of
  the retreat (negative = it ended behind where it was) -- the tape number;
- back: from the furthest-forward point to the end of the retreat;
- after the retreat, over the stand that follows: the drift and whether it
  settles (mean planar speed over its last second);
- peak tilt and the narrowest foot gap during the whole switch, and falls.

The switch is spread over a gait cycle: env i switches (i % 8) * 4 control
steps after the others (~0.65 s cycle at 1.55 Hz), so every phase is covered
and the table reports the mean and the worst env. The command is written into
the command term before env.step(), so the policy first sees it one control
step later -- as on the robot, where the bridge writes it into the next obs
frame.

Blocks run back to back in the same envs: warm up on the start command, switch,
retreat, stand. Envs that fell in a block are excluded from its distances.

Run from the project root (V3 code lives under v3/; nothing outside it changes):
    ~/IsaacLab/isaaclab.sh -p v3/sim/eval_retreat.py --headless --num_envs 256 \\
        --task Isaac-Velocity-Flat-R1-Play-v2 --checkpoint models/v2_drift_ft/model_4799.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

sys.path.insert(0, str(Path.home() / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

_DEFAULT_STARTS = "0,0,0; 0.3,0,0; 0.5,0,0; 1.0,0,0; -0.3,0,0; 0,0.2,0; 0,-0.2,0; 0,0,0.5; 0,0,-0.5"

parser = argparse.ArgumentParser(description="Measure the v2 policy's answer to the V3 retreat command.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-Play-v2", help="Name of the task.")
parser.add_argument("--starts", type=str, default=_DEFAULT_STARTS, help="Start commands 'vx,vy,wz; ...'.")
parser.add_argument("--retreat_vx", type=float, default=-0.3, help="Retreat command, m/s.")
parser.add_argument("--retreat_s", type=float, default=1.0, help="How long the retreat command is held.")
parser.add_argument("--ramp_s", type=float, default=0.0, help="Linear ramp into and out of the retreat.")
parser.add_argument("--stand_s", type=float, default=3.0, help="Zero command after the retreat.")
parser.add_argument("--warmup_s", type=float, default=3.0, help="On the start command before the first switch.")
parser.add_argument("--phases", type=int, default=8, help="Switch offsets spread over one gait cycle.")
parser.add_argument("--phase_step", type=int, default=4, help="Control steps between switch offsets.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--tag", type=str, default="", help="Suffix for the output filename.")
parser.add_argument("--out_dir", type=str, default="outputs/v3/sim", help="Where the report goes.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import math
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import tasks.r1_flat  # noqa: E402, F401  -- registers the task
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _command_at(k, switch, start, retreat, ramp_n, retreat_n):
    """Per-env command (n, 3) at control step k of a block; `switch` is each env's switch step."""
    t = (k - switch).float().unsqueeze(1)  # steps since the switch, negative before it
    if ramp_n > 0:
        a_in = torch.clamp(t / ramp_n, 0.0, 1.0)
        a_out = torch.clamp((t - retreat_n) / ramp_n, 0.0, 1.0)
    else:
        a_in = (t >= 0).float()
        a_out = (t >= retreat_n).float()
    cmd = start * (1.0 - a_in) + retreat * a_in
    return cmd * (1.0 - a_out)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = 1.0e6
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    # Commands are written per env every step: no resampling, heading or standing envs.
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)
    cmd_cfg.heading_command = False
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.rel_standing_envs = 0.0
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.command_ranges = None
    env_cfg.events.push_robot = None
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join(str(_PROJECT_ROOT), "logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    term = unwrapped.command_manager.get_term("base_velocity")
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    body_ids, _ = robot.find_bodies(_FEET, preserve_order=True)
    dt = unwrapped.step_dt
    n = unwrapped.num_envs
    dev = unwrapped.device
    starts = [tuple(float(x) for x in s.split(",")) for s in args_cli.starts.split(";") if s.strip()]
    retreat = torch.tensor([args_cli.retreat_vx, 0.0, 0.0], device=dev)
    ramp_n = int(round(args_cli.ramp_s / dt))
    retreat_n = int(round(args_cli.retreat_s / dt))
    stand_n = int(round(args_cli.stand_s / dt))
    warm_n = int(round(args_cli.warmup_s / dt))
    offsets = (torch.arange(n, device=dev) % args_cli.phases) * args_cli.phase_step
    switch = warm_n + offsets
    # the retreat ends at retreat_n + ramp_n after the switch (the ramp out starts at retreat_n)
    end_retreat = retreat_n + ramp_n
    block_n = warm_n + int(offsets.max()) + end_retreat + stand_n
    run_label = f"{os.path.basename(os.path.dirname(resume_path))} / {os.path.basename(resume_path)}"

    rows = []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for start in starts:
            start_t = torch.tensor(start, device=dev)
            fell = torch.zeros(n, dtype=torch.bool, device=dev)
            p0 = torch.zeros(n, 3, device=dev)
            q0 = torch.zeros(n, 4, device=dev)
            x_max = torch.full((n,), -float("inf"), device=dev)
            reverse_s = torch.full((n,), float("nan"), device=dev)
            net = torch.zeros(n, 2, device=dev)
            yaw_end = torch.zeros(n, device=dev)
            p_stand = torch.zeros(n, 3, device=dev)
            v_last = torch.zeros(n, device=dev)
            v_last_cnt = 0
            tilt_max = torch.zeros(n, device=dev)
            gap_min = torch.full((n,), float("inf"), device=dev)
            yaw0 = torch.zeros(n, device=dev)
            for k in range(block_n):
                kk = torch.full((n,), k, device=dev)
                term.vel_command_b[:] = _command_at(kk, switch, start_t, retreat, ramp_n, retreat_n)
                obs, _, dones, _ = env.step(policy(obs))
                fell |= dones.bool()
                d = robot.data
                since = k + 1 - switch  # steps since the switch, after this step
                at_switch = since == 0
                if at_switch.any():
                    p0[at_switch] = d.root_pos_w[at_switch]
                    q0[at_switch] = yaw_quat(d.root_quat_w[at_switch])
                    yaw0[at_switch] = _yaw(d.root_quat_w[at_switch])
                active = since > 0
                if not active.any():
                    continue
                rel = quat_rotate_inverse(q0, d.root_pos_w - p0)
                in_retreat = active & (since <= end_retreat)
                x_max = torch.where(in_retreat, torch.maximum(x_max, rel[:, 0]), x_max)
                back = in_retreat & torch.isnan(reverse_s) & (d.root_lin_vel_b[:, 0] < -0.05)
                reverse_s = torch.where(back, since.float() * dt, reverse_s)
                done_retreat = since == end_retreat
                net = torch.where(done_retreat.unsqueeze(1), rel[:, :2], net)
                yaw_end = torch.where(done_retreat, _wrap(_yaw(d.root_quat_w) - yaw0), yaw_end)
                p_stand = torch.where(done_retreat.unsqueeze(1), d.root_pos_w, p_stand)
                g = d.projected_gravity_b
                tilt = torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))
                tilt_max = torch.where(active, torch.maximum(tilt_max, tilt), tilt_max)
                feet = d.body_pos_w[:, body_ids]
                gap = quat_rotate_inverse(yaw_quat(d.root_quat_w), feet[:, 0] - feet[:, 1])[:, 1]
                gap_min = torch.where(active, torch.minimum(gap_min, gap), gap_min)
                if k >= block_n - int(round(1.0 / dt)):
                    v_last += torch.linalg.norm(d.root_lin_vel_b[:, :2], dim=1)
                    v_last_cnt += 1
            stand_drift = torch.linalg.norm((robot.data.root_pos_w - p_stand)[:, :2], dim=1)
            ok = ~fell
            row = {
                "start": start,
                "falls": int(fell.sum()),
                "envs": n,
                "overshoot": _stats(torch.clamp(x_max, min=0.0), ok),
                "reverse_s": _stats(reverse_s, ok & ~torch.isnan(reverse_s)),
                "never_reversed": int((ok & torch.isnan(reverse_s)).sum()),
                "net_x": _stats(net[:, 0], ok),
                "net_y": _stats(net[:, 1], ok),
                "back": _stats(x_max - net[:, 0], ok),
                "yaw_deg": _stats(torch.rad2deg(yaw_end), ok),
                "stand_drift": _stats(stand_drift, ok),
                "stand_speed_last_1s": _stats(v_last / max(v_last_cnt, 1), ok),
                "tilt_max_deg": _stats(torch.rad2deg(tilt_max), ok),
                "gap_min": float(gap_min[ok].min()) if ok.any() else float("nan"),
                "per_phase_net_x": [
                    float(net[ok & (offsets == o * args_cli.phase_step), 0].mean())
                    if (ok & (offsets == o * args_cli.phase_step)).any() else float("nan")
                    for o in range(args_cli.phases)
                ],
            }
            rows.append(row)
            print(f"[INFO] {json.dumps(row)}", flush=True)
            # a fallen env was reset mid-block; the next block's warm-up re-establishes its gait

    env.close()

    cfg_line = (
        f"retreat {args_cli.retreat_vx:+.2f} m/s for {args_cli.retreat_s:.2f} s, ramp {args_cli.ramp_s:.2f} s,"
        f" then zero for {args_cli.stand_s:.1f} s; switch spread over {args_cli.phases} offsets x"
        f" {args_cli.phase_step} steps; {n} envs; task {args_cli.task}; seed {env_cfg.seed}"
    )
    out = [
        "",
        "=" * 78,
        f"V3 retreat response: {run_label}",
        cfg_line,
        "=" * 78,
        "",
        "Distances in m along the heading at the switch; mean / worst env. 'worst' is the max for overshoot,"
        " reverse time, stand drift, stand speed and tilt, and the least-backward (max) for net and back.",
        "",
        "| start (vx, vy, wz) | falls | overshoot | reverse s | net x | back | net y | yaw deg |"
        " stand drift (3 s) | stand speed last 1 s | tilt max deg | min foot gap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["start"]
        out.append(
            f"| {s[0]:+.2f}, {s[1]:+.2f}, {s[2]:+.2f} | {r['falls']}/{r['envs']} |"
            f" {_fmt(r['overshoot'], 'max')} | {_fmt(r['reverse_s'], 'max')}"
            f"{' (' + str(r['never_reversed']) + ' never)' if r['never_reversed'] else ''} |"
            f" {_fmt(r['net_x'], 'max')} | {_fmt(r['back'], 'min')} | {_fmt(r['net_y'], 'absmax')} |"
            f" {_fmt(r['yaw_deg'], 'absmax')} | {_fmt(r['stand_drift'], 'max')} |"
            f" {_fmt(r['stand_speed_last_1s'], 'max')} | {_fmt(r['tilt_max_deg'], 'max')} | {r['gap_min']:+.3f} |"
        )
    out.append("")
    out.append("net x by switch offset (steps after the first; mean over envs):")
    for r in rows:
        s = r["start"]
        phases = " ".join(f"{v:+.3f}" for v in r["per_phase_net_x"])
        out.append(f"- {s[0]:+.2f}, {s[1]:+.2f}, {s[2]:+.2f}: {phases}")
    report = "\n".join(out)
    print(report, flush=True)
    out_dir = os.path.join(str(_PROJECT_ROOT), args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"retreat{'_' + args_cli.tag if args_cli.tag else ''}"
    with open(os.path.join(out_dir, stem + ".md"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump({"config": vars(args_cli) | {"checkpoint_path": resume_path}, "rows": rows}, f, indent=1,
                  default=str)
    print(f"[INFO] wrote {os.path.join(out_dir, stem)}.md/.json", flush=True)


def _yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def _stats(x, mask):
    v = x[mask]
    if v.numel() == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def _fmt(s, worst):
    if worst == "absmax":
        w = s["max"] if abs(s["max"]) >= abs(s["min"]) else s["min"]
    else:
        w = s[worst]
    return f"{s['mean']:+.3f} / {w:+.3f}"


if __name__ == "__main__":
    main()
    simulation_app.close()
