# The v2 policy: backward, sideways, faster turns, and no creep on a zero command

The policy deployed until v2 (`week04_nohead`, "v1") walked forward and turned on the
spot. On a zero command it kept stepping and slowly crept forward. v2 is a retrain of the
same task. It walks backward and sideways, turns three times faster, and steps in place
without the forward creep. It keeps v1's observation and action interface, so the
deployment path did not change.

**On the robot (2026-10-20):** installed with `policy_pack` and the engine rebuilt on the
Orin. The install only completes once the parity gate has passed. The operator reported
that the robot walks backward, walks sideways and holds its place on a zero command. The
session's logs and tape measurements are not in this repository.

| | v1 (`week04_nohead`) | v2 (`v2_drift_ft`) |
|---|---|---|
| command envelope | vx 0 → 1.0 m/s, vy 0, wz ±0.5 rad/s | vx −0.6 → 1.0 m/s, vy ±0.5 m/s, wz ±1.57 rad/s |
| zero command, simulation | +0.046 m/s forward, 0.45 m per 10 s | +0.014 m/s, 0.14 m per 10 s; +0.007 m/s averaged over 256 randomization draws |
| interface | 425 → 24 at 50 Hz | the same; same PD gains, same joint map |

Files: task [tasks/r1_flat/flat_v2_env_cfg.py](../tasks/r1_flat/flat_v2_env_cfg.py),
artifacts [models/v2_drift_ft/](../models/v2_drift_ft/README.md), measurement tool
[scripts/eval_command_grid.py](../scripts/eval_command_grid.py).

## Why v1 creeps forward

`scripts/eval_command_grid.py` holds each command for 10 s in 64 simulated envs, at
nominal conditions. On a zero command v1 moved forward at +0.046 m/s, with p10/p90 across
envs of +0.046/+0.047. Both feet were stepping (each foot airborne 46 % of the time).
The narrow spread rules out noise: this is a bias of the policy itself, and it is present
in simulation, so it is not a sim2real effect. The same offset runs up the low end of the
speed range: 0.05 → 0.090, 0.10 → 0.137, and it fades by 0.3 m/s. A one-step action delay
changed it only from 0.046 to 0.041, so it is not the latency model either.

Three things in v1's training let the bias survive:

1. **Zero was the edge of the trained range.** lin_vel_x was sampled from [0, 1.0], so
   every other command asked for forward motion.
2. **Only 2 % of envs ever got a zero command** (`rel_standing_envs`).
3. **The tracking reward was too wide to notice.** exp(−err²/0.25) pays 0.991 for a
   0.047 m/s error, so removing the drift was worth under 1 % of that term.

## What v2 changes

The observation and action interface are unchanged, and so are the actuators and the
randomization. These are the changes:

| | v1 | v2 | answers |
|---|---|---|---|
| commands | vx [0, 1.0], vy 0, wz ±0.5 | vx [−0.6, 1.0], vy ±0.5, wz ±1.57; heading control for half the envs | zero is inside the range |
| zero-command envs | 2 % | 15 % | |
| velocity tracking | exp kernel, std 0.5 | plus a narrow one, std 0.15, weight 0.5 | the wide kernel |
| swing reward on a zero command | not paid | paid | a zero command means stepping in place, not standing |
| drift on a zero command | — | the planar velocity averaged over ~1 s, penalized (weight −5), **second stage only** | the residual bias |
| feet closer than 0.07 m sideways | — | penalized | sideways steps can pass one leg through the other: the simulation has no self-collisions |

The averaged-velocity penalty is there because the narrow kernel was not enough. Stepping
in place sways the pelvis by several cm/s on every step, and a steady 0.026 m/s offset is
a small part of what an instantaneous kernel measures. Averaging over about a second
removes the sway and keeps the offset.

## Four runs, and why v2 trains in two stages

| run | what | zero command | other |
|---|---|---|---|
| `15-51-33_v2` | the changes above, without the drift penalty, 4000 iterations | +0.026 m/s | backward, sideways, fast turns all tracked, no falls |
| `17-57-14_v2_drift_ft` | **the v2 policy**: the run above, resumed for 800 iterations with the drift penalty | **+0.014 m/s**, stepping | 0 falls over 5 seeds × 256 envs |
| `18-31-18_v2_drift` | drift penalty from iteration 0 | 0.00 m/s, but **standing still** | ignored every command up to 0.1 m/s and 0.5 rad/s, the speeds `mission_ctl` uses |
| `20-36-55_v2_final` | one continuous run, penalty switched on at iteration 4000 | +0.020 m/s | fell on a zero command from some resets (about 9 per 10 s in 128 envs, one seed) |

- **When the penalty starts matters.** Switched on before the gait exists, the cheapest
  way to stop drifting is to stop stepping, and the robot learns a dead band around zero
  with it.
- **The two-stage run and the continuous run differ only by chance.** Training is
  deterministic: the continuous run's first 4000 iterations are bit-identical to the
  first run's, and re-running stage 2 from the same checkpoint reproduced the policy bit
  for bit. So the difference between the second and fourth rows is the random course of
  the last 800 iterations, and the policy kept is the one that measured better.
- **The remaining drift is mostly spread between randomization draws.** Over 256 draws
  of friction, mass, gains and delay, the mean is +0.007 m/s with p10/p90 −0.006/+0.023.
  The drift penalty's value stopped improving within 100 iterations. A policy that cannot
  identify friction, mass and gains cannot remove this part.

## v2 in simulation

64 envs, nominal conditions, 10 s per command, blocks back to back:

| command | v1 | v2 |
|---|---|---|
| 0 | +0.046 | +0.014 (stepping, air 0.47/0.47) |
| vx 0.1 / 0.5 / 1.0 | 0.137 / 0.502 / 0.982 | 0.107 / 0.490 / 1.017 |
| vx −0.3 / −0.6 | −0.269 / falls | −0.278 / −0.588 |
| vy +0.5 / −0.5 | 0 / 0 | +0.478 / −0.477 |
| wz 0.5 / 1.0 / 1.57 | 0.472 / 0.834 / 1.032 | 0.475 / 0.929 / 1.407 |
| narrowest foot gap | 0.090 m (forward 1.0 m/s; at −0.6 it fell) | 0.069 m (sideways at 0.5 m/s) |

The gait alternates (phase offset 0.50), is symmetric (swing-duration ratio 0.99) and
steps at 1.55 Hz.

## Deployment

- **Export and engine.**
  - `scripts/play_r1.py` exported the ONNX. It is the same graph as v1's, and the same
    size.
  - ONNX Runtime against PyTorch on the fixture's 512 inputs: max_abs 1.5e-05.
  - A TensorRT 10.7 FP32 engine on the dev box: 2.3e-05 (gate 1e-3).
  - The robot's engine is built on the Orin by the installer and gated against the same
    fixture.
- **Bundle.**
  - `policy_pack/make_bundle.py --task-cfg tasks/r1_flat/flat_v2_env_cfg.py --interface
    models/v2_drift_ft/interface --artifacts models/v2_drift_ft` builds it.
  - The envelope is parsed from `COMMAND_RANGES_FINAL_V2`.
  - `deploy/tools/dump_interface.py` and `dump_actuator_gains.py` gained `--run` and
    `--out` / `--spec`, so a policy that is not the deployed one gets its own interface
    files.
- **Bridge.**
  - No C++ change. Install step 5 rewrites the three `cmd_vel_*_range` lines of
    `bridge.yaml` from the bundle; at start-up the bridge prints
    `vx [-0.60, 1.00] vy [-0.50, 0.50] wz [-1.57, 1.57]`.
  - `joint_map.hpp` does not change, because the gains are v1's.
  - This was the first new policy to go through `policy_pack`; stage D had only
    re-installed v1.
- **Rollback** is `install_bundle.sh` with v1's bundle. Rehearsed on a copy of the
  release tree: afterwards `bridge.yaml` was byte-identical to before.

## The command layer

- **`walk` has a direction.** The syntax is `walk [forward|backward|left|right]
  <N>s|<N>m[@<speed>]`. The speed is a magnitude; the direction picks the axis and the
  sign.
- **The installed envelope decides which directions exist.** With v1 installed, backward
  is outside the trained range and sideways is refused as a pinned axis, exactly as
  before. With v2 installed, all four directions work.
- **Defaults and safety rules.**
  - Default speeds: `cruise_back_vx` 0.3 m/s and `cruise_vy` 0.2 m/s.
  - With `perception_guard: all`, backward and sideways walks are refused, because the
    camera looks forward.
  - A measured `v_cal` converts forward distances only.
- **English (`ask`): the prompt did not change.** The model already wrote backward, left
  and right walks, and the deterministic layer refused them. Now that layer executes them
  when the envelope allows, so the prompt id is the same and the earlier eval still
  applies.
- **A new frozen set, `mission_ctl/eval/nl_test5.jsonl`.** It has 20 cases: backward,
  sideways, sequences and fast turns, judged against the v2 envelope.
  - qwen3:1.7b on the CPU, as on the robot: 19/20 correct, **0 unsafe** (moving, but not
    as asked). The one miss was refused for the wrong reason.
  - The 100 earlier held-out cases score as in stage E.
  - Writing this set exposed a gap in the scorer: it compared plans without vy, so left
    and right would have counted as the same motion. It compares vy now.

## What this does not show

- **No tape measurement of the real zero-command drift is recorded here**, so the robot
  number to compare with the simulator's 0.14 m per 10 s is still missing.
- **`kp_scale` 1.3 is v1's demo gain.** v2 was trained at the same gains and has not been
  swept on the robot.
- **`turn_min_wz` (0.15 rad/s) was measured with v1.**
- **Distances are still time × commanded speed.** There is no odometry.

## Reproducing

```bash
# stage 1: 4000 iterations, drift penalty weight 0
~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --headless \
    --task Isaac-Velocity-Flat-R1-v2 --num_envs 4096 --run_name v2
# stage 2: +800 iterations with the drift penalty, command ranges open from the start
~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --headless \
    --task Isaac-Velocity-Flat-R1-v2 --num_envs 4096 --max_iterations 4800 \
    --run_name v2_drift_ft --resume --load_run <stage-1 run> --checkpoint model_3999.pt \
    env.curriculum.command_ranges.params.end_step=1 env.rewards.zero_command_drift.weight=-5.0
# measure
~/IsaacLab/isaaclab.sh -p scripts/eval_command_grid.py --headless --num_envs 64 \
    --task Isaac-Velocity-Flat-R1-Play-v2 --checkpoint models/v2_drift_ft/model_4799.pt
```
Each stage takes about 2 h and 25 min on an RTX 2080 Ti.
