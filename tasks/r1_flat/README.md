# tasks/r1_flat/

`Isaac-Velocity-Flat-R1-v0`: R1 walking on flat ground, tracking a commanded forward
speed and yaw rate. The v1 policy
([`models/week04_nohead/`](../../models/week04_nohead/)) was trained on this task as it
stands. `Isaac-Velocity-Flat-R1-v2` builds on it and trained the v2 policy
([`models/v2_drift_ft/`](../../models/v2_drift_ft/README.md)).

| file | what's in it |
|---|---|
| `__init__.py` | `gym.register()` for `Isaac-Velocity-Flat-R1-v0` and its `-Play-v0` variant (fewer envs, no randomization), both with the PPO config as `rsl_rl_cfg_entry_point` |
| `flat_env_cfg.py` | `R1FlatEnvCfg`: scene, commands, actions, observations, randomization events, rewards, terminations, command curriculum. Each config class's docstring says why it is the way it is |
| `mdp.py` | R1-specific terms not in Isaac Lab: the touchdown-gated swing reward, the command-range curriculum, the action-level control delay |
| `flat_v2_env_cfg.py` | `R1FlatV2EnvCfg`, task `Isaac-Velocity-Flat-R1-v2`, the policy of release v2.0.0: v1 plus backwards, sideways and faster turns, and no drift on a zero command. See [v2](#v2) |
| `symmetry.py` | the left/right mirror map for PPO's symmetry augmentation |
| `agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters, seed 42, 3000 iterations, symmetry augmentation on; v2 runs 4000 + an 800-iteration second stage and logs to `logs/rsl_rl/r1_flat_v2/` |

## The task in one table

| | |
|---|---|
| **action** | 24 joint-position targets (legs 2×6, waist 2, arms 2×5; the head is not actuated), per-group scale `0.25 × effort / stiffness`, at 50 Hz (`sim.dt = 0.005`, `decimation = 4`) |
| **policy observation** | 85 floats per frame, proprioception only (angular velocity, gravity direction, command, joint positions and velocities, last action), the last 5 frames stacked → **425**. No base linear velocity: the robot cannot measure it |
| **critic observation** | adds privileged simulator state (base linear velocity, external wrench, foot friction): asymmetric actor-critic |
| **commands** | forward 0 → 1.0 m/s, yaw rate ±0.5 rad/s, reached by a curriculum over the first 1000 iterations; **sideways is pinned to 0 and there is no backward**, which is why the robot refuses both |
| **randomization** | friction 0.6–1.2, body mass ±10 %, actuator gains ×0.85–1.15, pushes ±0.5 m/s every 10–15 s, control delay 0–1 step |
| **rewards** | velocity tracking; a swing reward paid only at touchdown for a 0.15–0.45 s swing; penalties including feet held up, foot slide, joint limits, upper-body and hip drift, falling |
| **termination** | base below 0.51 m, tilt beyond 0.7 rad, or the 20 s episode ends |

## Traps this task already walked into

Each is fixed in the code and explained where it is fixed; listed so nobody undoes one:

- **`feet_air_time_positive_biped` is maximized by standing on one leg.** It is not used.
  The swing reward is paid at touchdown instead (`mdp.py`). W03 spent three training runs
  on it; see [`experiments/README.md`](../../experiments/README.md).
- **Any actuated joint without a reward on it becomes a balance aid.** The head ended up
  pinned at its limit, so it was taken out of the action space (24 actions, not 26).
- **Actuator limits come from the robot's spec, not from what makes standing pass.**
  The W02 values were 2.5–3× stronger than the hardware.
- **Stiff leg gains need a fine timestep and an implicit actuator.** With the W02 gains
  (~1200 N·m/rad) the robot collapsed at any coarser step than `dt = 0.002`, whatever the
  gains; with an explicit PD actuator it collapsed even then. Since the gains were
  corrected to the hardware's 100/40 (W04), `dt = 0.005` is stable and is what the
  deployed policy trained at. If you raise the gains, re-check standing with
  `scripts/inspect_r1.py` before training.

Longer write-ups of each are in [`scripts/README.md`](../../scripts/README.md#known-pitfalls).

## v2

v1 creeps forward on a zero command: **+0.047 m/s, 0.45 m per 10 s**, the same in every
simulated env, with both feet stepping. So it is the policy's own bias, already present in
simulation. Three things let it survive training:
- zero was the edge of the trained range (forward 0 → 1.0);
- only 2 % of envs ever saw a zero command;
- the tracking reward is wide enough that the drift cost it under 1 %.

v2 keeps v1's observation and action interface (425 → 24) and changes only the commands
and five rewards:

| | v1 | v2 |
|---|---|---|
| forward | 0 → 1.0 m/s | −0.6 → 1.0 m/s |
| sideways | 0 | ±0.5 m/s |
| yaw rate | ±0.5 rad/s | ±1.57 rad/s, heading control on for half the envs |
| zero-command envs | 2 % | 15 % |
| velocity tracking | exp kernel, std 0.5 | the same, plus a narrow one (std 0.15, weight 0.5) |
| swing reward on a zero command | not paid | paid: a zero command means stepping in place |
| drift on a zero command | not checked | penalized on the velocity averaged over ~1 s, so the step-to-step sway does not hide it; **only in a second stage, 800 iterations resumed from the 4000-iteration gait**: on from the start, the robot learned to stand still and to ignore commands up to 0.1 m/s and 0.5 rad/s |
| feet closer than 0.07 m sideways | not checked | penalized (no self-collisions in sim) |

The reasoning for each line is in the module docstring of `flat_v2_env_cfg.py`. Both
policies are measured with `scripts/eval_command_grid.py`. Results, the four runs and the
robot: [docs/v2_policy.md](../../docs/v2_policy.md).

## History

W02: the task skeleton and standing; W03: rewards and the first real gait; W04: domain
randomization, the observation history and the final 24-action / 425-observation
interface. Weeks are explained in [docs/project_history.md](../../docs/project_history.md).
