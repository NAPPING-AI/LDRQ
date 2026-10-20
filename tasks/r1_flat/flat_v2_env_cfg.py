# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R1 flat-ground velocity task, v2: backwards, sideways, faster turns, and no drift on a zero command.

Built on ``flat_env_cfg.R1FlatEnvCfg`` (v1, the deployed policy's task), which
is left unchanged so that run stays reproducible. Everything not listed here
is inherited: scene, actuators, observations (still 425 dims, so the exported
ONNX keeps v1's interface), domain randomization, terminations.

Why v1 creeps forward on a zero command -- measured on
``week04_nohead/model_2999.pt``, 64 envs, nominal sim, no DR:

- zero command: body vx +0.047 m/s (0.45 m per 10 s), the same in every env
  (p10/p90 +0.046/+0.047), both feet stepping (air fraction 0.46/0.46). It is
  a bias of the policy itself, already present in simulation.
- the same offset carries up the low end: 0.05 -> 0.090, 0.10 -> 0.137,
  0.20 -> 0.228, and fades by 0.3 m/s (PG-1 error 0.019 there). Negative
  commands, never trained, come out shifted the same way: -0.10 -> -0.045.

Three things in v1 let that bias survive training:

1. zero sat at the *edge* of the trained range (lin_vel_x in [0, 1]); every
   other command asked for forward motion;
2. only 2% of envs ever got a zero command (``rel_standing_envs``);
3. the tracking kernel is wide: exp(-err^2 / 0.25) pays 0.991 for a 0.047 m/s
   error, so removing the drift was worth under 1% of that term.

v2 changes, each answering one of those:

- commands span zero: lin_vel_x (-0.6, 1.0), lin_vel_y (-0.5, 0.5),
  ang_vel_z (-1.57, 1.57), with heading control on;
- 15% zero-command envs instead of 2%;
- a second, narrow linear-velocity tracking term (std 0.15) on top of v1's;
  the same 0.047 m/s error now costs ~9% of it;
- stepping is paid on a zero command too (``feet_swing_touchdown`` with
  ``min_command=0``), because the zero-command behavior asked for is stepping
  in place, not standing still;
- on a zero command, a penalty on the planar velocity *averaged over ~1 s*
  (``zero_command_drift``). Added after the first v2 run
  (``2026-10-16_15-51-33_v2``, without it) still crept at +0.026 m/s nominal:
  the narrow kernel sees instantaneous velocity, in which the step-to-step
  sway hides a steady 0.026 m/s. Fine-tuning that run for 800 iterations with
  this term (``2026-10-16_17-57-14_v2_drift_ft``) brought it to +0.014 m/s
  nominal and +0.007 m/s averaged over 256 randomization draws (p10/p90
  -0.006/+0.023). The term's value plateaued within 100 iterations: what is
  left is mostly the spread between draws, which a policy that cannot
  identify friction, mass and gains cannot remove.

  When the term comes in matters. On from iteration 0
  (``2026-10-16_18-31-18_v2_drift``) the robot learned to stand still on a
  zero command, and with it to ignore every command up to 0.1 m/s and
  0.5 rad/s, the speeds mission_ctl walks and turns at. Switched on at
  iteration 4000 inside one continuous run (``2026-10-16_20-36-55_v2_final``)
  it drifted more (+0.020 m/s) and fell on a zero command from some resets
  (~9 per 10 s in 128 envs, seed 2). The fine-tune above had neither problem
  (0 falls, 5 seeds x 256 envs), so it is the v2 policy and the recipe is its
  two stages;
- a foot-gap penalty, since sideways steps and fast turns can pass one leg
  through the other in a simulation without self-collisions. Threshold 0.07 m:
  v1's own gait gets as close as 0.090 m at 1.0 m/s (and 0.15 m stepping in
  place), so only a real near-miss or a crossing pays.

The v1 numbers above and the v2 acceptance numbers both come from
scripts/eval_command_grid.py.

Training is two stages (seed 42, deterministic: re-running stage 2 from the
same stage-1 checkpoint reproduced the policy bit for bit)::

    # stage 1: the gait, 4000 iterations (zero_command_drift weight 0)
    ~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --headless \
        --task Isaac-Velocity-Flat-R1-v2 --num_envs 4096 --run_name v2
    # stage 2: +800 iterations with the drift penalty, command ranges fully open
    ~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --headless \
        --task Isaac-Velocity-Flat-R1-v2 --num_envs 4096 --max_iterations 4800 \
        --run_name v2_drift_ft --resume --load_run <stage-1 run> --checkpoint model_3999.pt \
        env.curriculum.command_ranges.params.end_step=1 env.rewards.zero_command_drift.weight=-5.0
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from tasks.r1_flat.flat_env_cfg import R1FlatEnvCfg
from tasks.r1_flat.mdp import feet_lateral_gap_below, zero_command_drift

# Curriculum end points. The start already spans zero on x so the zero command
# is never the edge of the trained range, not even early on.
COMMAND_RANGES_INITIAL_V2 = {
    "lin_vel_x": (-0.3, 0.5),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.5, 0.5),
}
COMMAND_RANGES_FINAL_V2 = {
    "lin_vel_x": (-0.6, 1.0),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-1.57, 1.57),
}


@configclass
class R1FlatV2EnvCfg(R1FlatEnvCfg):
    """v2 training config."""

    def __post_init__(self):
        super().__post_init__()

        # -- commands --
        cmd = self.commands.base_velocity
        cmd.heading_command = True
        # With heading control on, a heading env's yaw-rate command is
        # 0.5 * heading error, clipped to ang_vel_z: it starts large and decays
        # to ~0 within a few seconds. The robot is driven with *constant* yaw
        # rates (mission_ctl's turns), so half the envs keep a directly sampled
        # ang_vel_z instead. 0.5 * pi = 1.57, so the heading envs can reach the
        # whole range.
        cmd.rel_heading_envs = 0.5
        cmd.heading_control_stiffness = 0.5
        cmd.rel_standing_envs = 0.15
        cmd.ranges.lin_vel_x = COMMAND_RANGES_INITIAL_V2["lin_vel_x"]
        cmd.ranges.lin_vel_y = COMMAND_RANGES_INITIAL_V2["lin_vel_y"]
        cmd.ranges.ang_vel_z = COMMAND_RANGES_INITIAL_V2["ang_vel_z"]
        cmd.ranges.heading = (-math.pi, math.pi)

        # Ramp over the first 1500 of 4000 iterations (24 control steps each).
        self.curriculum.command_ranges.params["initial_ranges"] = COMMAND_RANGES_INITIAL_V2
        self.curriculum.command_ranges.params["final_ranges"] = COMMAND_RANGES_FINAL_V2
        self.curriculum.command_ranges.params["end_step"] = 24 * 1500

        # -- rewards --
        rew = self.rewards
        rew.track_lin_vel_xy_fine = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.15},
        )
        rew.feet_swing_touchdown.params["min_command"] = 0.0
        # Weight 0 here: stage 1 trains the gait without it, stage 2 resumes
        # with weight -5 (commands in the module docstring). Switched on from
        # the start, the robot learned to stand still instead of stepping.
        rew.zero_command_drift = RewTerm(
            func=zero_command_drift,
            weight=0.0,
            params={"command_name": "base_velocity", "tau": 1.0},
        )
        rew.feet_lateral_gap = RewTerm(
            func=feet_lateral_gap_below,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
                ),
                "min_gap": 0.07,
            },
        )


@configclass
class R1FlatV2EnvCfg_PLAY(R1FlatV2EnvCfg):
    """Nominal conditions and the full command range, for play and evaluation (as v1's PLAY)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.actuator_gains = None
        self.events.push_robot = None
        self.actions.joint_pos.max_delay = 0
        self.curriculum.command_ranges = None
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = COMMAND_RANGES_FINAL_V2["lin_vel_x"]
        ranges.lin_vel_y = COMMAND_RANGES_FINAL_V2["lin_vel_y"]
        ranges.ang_vel_z = COMMAND_RANGES_FINAL_V2["ang_vel_z"]
