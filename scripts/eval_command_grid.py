# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold a checkpoint at a list of (vx, vy, wz) commands and measure what the body actually does.

eval_baseline.py only commands forward speeds and only reports forward error.
This is the v2 counterpart: every axis, including the zero command, which is
the case v2 exists to fix (v1 creeps forward at +0.047 m/s on it). Per block:

- mean body-frame vx, vy and yaw rate, against the command;
- displacement over the block along the starting heading, scaled to metres per
  10 s -- the number to compare with a tape measure on the robot;
- p10/p90 of the per-env mean vx, which tells a bias (narrow) from noise (wide);
- per-foot air fraction (about 0.4-0.5 = stepping; 0 = standing still);
- the smallest sideways gap between the feet (negative = legs crossed);
- falls.

With the Play task the conditions are nominal; with the training task each env
keeps its own randomization draw (pushes and the curriculum are switched off),
and the p10/p90 column then shows the spread across draws.

Blocks run back to back in the same envs, so a zero block placed after a
walking block measures the walk-to-stop transition too. Episodes don't time
out during the run.

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p scripts/eval_command_grid.py --headless --num_envs 64 \\
        --task Isaac-Velocity-Flat-R1-Play-v2 --checkpoint logs/rsl_rl/r1_flat_v2/<run_id>/model_3999.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

sys.path.insert(0, str(Path.home() / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

_DEFAULT_GRID = (
    "0,0,0; 0.05,0,0; 0.1,0,0; 0.3,0,0; 0.5,0,0; 1.0,0,0; 0,0,0;"
    " -0.1,0,0; -0.3,0,0; -0.6,0,0; 0,0,0;"
    " 0,0.2,0; 0,0.5,0; 0,-0.2,0; 0,-0.5,0; 0,0,0;"
    " 0,0,0.5; 0,0,1.0; 0,0,1.57; 0,0,-1.0; 0,0,0;"
    " 0.5,0.3,0; 0.5,0,0.5; 0,0,0"
)

parser = argparse.ArgumentParser(description="Measure body velocity, drift and gait at fixed (vx, vy, wz) commands.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to average over.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-Play-v2", help="Name of the task.")
parser.add_argument("--grid", type=str, default=_DEFAULT_GRID, help="Commands 'vx,vy,wz; vx,vy,wz; ...'.")
parser.add_argument("--steps", type=int, default=500, help="Recorded control steps per command (500 ~ 10s).")
parser.add_argument("--warmup", type=int, default=100, help="Steps discarded after each command switch (~2s).")
parser.add_argument("--tag", type=str, default="", help="Suffix for the output filename.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed (randomization draws, resets).")
parser.add_argument(
    "--delay", type=int, default=None, help="Fix the action delay at this many control steps (training draws 0-1)."
)
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
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import tasks.r1_flat  # noqa: E402, F401  -- registers the task
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _pin_command(env, vx: float, vy: float, wz: float) -> None:
    """Hold every env's command at (vx, vy, wz): collapse the ranges, no heading or standing envs."""
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.cfg.heading_command = False
    term.cfg.ranges.lin_vel_x = (vx, vx)
    term.cfg.ranges.lin_vel_y = (vy, vy)
    term.cfg.ranges.ang_vel_z = (wz, wz)
    term.cfg.rel_standing_envs = 0.0
    term.is_standing_env[:] = False
    term.vel_command_b[:] = torch.tensor([vx, vy, wz], device=term.vel_command_b.device)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # blocks run back to back; a timeout reset mid-block would restart the walk
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = 1.0e6
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    # The training task can be passed too, to measure under its randomization
    # (friction, mass, gains, delay, observation noise): commands stay pinned
    # and nothing pushes, but each env keeps its own random draw.
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.command_ranges = None
    env_cfg.events.push_robot = None
    if args_cli.delay is not None:
        env_cfg.actions.joint_pos.min_delay = args_cli.delay
        # the buffer must hold at least one step even for a zero lag
        env_cfg.actions.joint_pos.max_delay = max(args_cli.delay, 1) if args_cli.delay else 0
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
    contact = unwrapped.scene.sensors["contact_forces"]
    sensor_ids, _ = contact.find_bodies(_FEET, preserve_order=True)
    body_ids, _ = robot.find_bodies(_FEET, preserve_order=True)
    dt = unwrapped.step_dt
    n = unwrapped.num_envs
    dev = unwrapped.device
    grid = [tuple(float(x) for x in item.split(",")) for item in args_cli.grid.split(";") if item.strip()]
    run_label = f"{os.path.basename(os.path.dirname(resume_path))} / {os.path.basename(resume_path)}"

    rows = []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for vx, vy, wz in grid:
            _pin_command(env, vx, vy, wz)
            for _ in range(args_cli.warmup):
                obs, *_ = env.step(policy(obs))
            start_xy = robot.data.root_pos_w.clone()
            start_yaw = yaw_quat(robot.data.root_quat_w).clone()
            vel = torch.zeros(n, 3, device=dev)
            air = torch.zeros(n, 2, device=dev)
            min_gap = torch.full((n,), float("inf"), device=dev)
            falls = 0
            for _ in range(args_cli.steps):
                obs, _, dones, extras = env.step(policy(obs))
                d = robot.data
                vel[:, :2] += d.root_lin_vel_b[:, :2]
                vel[:, 2] += d.root_ang_vel_b[:, 2]
                air += (contact.data.current_contact_time[:, sensor_ids] <= 0.0).float()
                feet = d.body_pos_w[:, body_ids]
                gap = quat_rotate_inverse(yaw_quat(d.root_quat_w), feet[:, 0] - feet[:, 1])[:, 1]
                min_gap = torch.minimum(min_gap, gap)
                falls += int(dones.bool().sum())
            seconds = args_cli.steps * dt
            disp = quat_rotate_inverse(start_yaw, robot.data.root_pos_w - start_xy)
            mean_vel = vel / args_cli.steps
            p10, p90 = torch.quantile(mean_vel[:, 0], torch.tensor([0.1, 0.9], device=dev)).tolist()
            rows.append(
                {
                    "cmd": (vx, vy, wz),
                    "vel": mean_vel.mean(0).tolist(),
                    "disp": (disp[:, :2].mean(0) * 10.0 / seconds).tolist(),
                    "p10p90": (p10, p90),
                    "air": (air / args_cli.steps).mean(0).tolist(),
                    "min_gap": float(min_gap.min()),
                    "falls": falls,
                }
            )
            print(f"[INFO] {rows[-1]}", flush=True)

    env.close()

    out = [
        "",
        "=" * 78,
        f"Command grid: {run_label}",
        f"{n} envs, {args_cli.steps} steps ({args_cli.steps * dt:.1f}s) per command after {args_cli.warmup} warm-up,"
        f" task {args_cli.task}, action delay {env_cfg.actions.joint_pos.min_delay}-{env_cfg.actions.joint_pos.max_delay}"
        " steps, blocks back to back",
        "=" * 78,
        "",
        "| cmd (vx, vy, wz) | body vx | body vy | yaw rate | moved per 10 s (fwd, left) m | env vx p10/p90 |"
        " air L/R | min foot gap (m) | falls |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["cmd"]
        out.append(
            f"| {c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f} | {r['vel'][0]:+.3f} | {r['vel'][1]:+.3f} | {r['vel'][2]:+.3f} |"
            f" {r['disp'][0]:+.2f}, {r['disp'][1]:+.2f} | {r['p10p90'][0]:+.3f}/{r['p10p90'][1]:+.3f} |"
            f" {r['air'][0]:.2f}/{r['air'][1]:.2f} | {r['min_gap']:+.3f} | {r['falls']} |"
        )
    zero = [r for r in rows if r["cmd"] == (0.0, 0.0, 0.0)]
    if zero:
        worst = max(max(abs(r["disp"][0]), abs(r["disp"][1])) for r in zero)
        out.append("")
        out.append(f"zero command: worst drift {worst:.2f} m per 10 s over {len(zero)} zero blocks")
    report = "\n".join(out)
    # one flushed write: Isaac Sim's shutdown can eat buffered stdout
    print(report, flush=True)
    out_path = os.path.join(os.path.dirname(resume_path), f"command_grid{'_' + args_cli.tag if args_cli.tag else ''}.md")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"[INFO] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
