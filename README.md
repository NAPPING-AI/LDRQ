# LDRQ v2: a Unitree R1 humanoid that walks in four directions, on English commands

[![tests](https://github.com/NAPPING-AI/LDRQ/actions/workflows/tests.yml/badge.svg)](https://github.com/NAPPING-AI/LDRQ/actions/workflows/tests.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[中文](README_zh.md)

A walking policy for the **Unitree R1** humanoid, trained in Isaac Lab, deployed on the
robot's own Jetson Orin NX, measured against the simulator, and commanded in English
through a small language model running on the robot.

> **This is the `v2` branch.** The project has three versions, one per branch: `main`
> (v1), `v2` (this one) and `v3` (V3). The overview of all three, with each version's
> architecture, is the README on [`main`](https://github.com/NAPPING-AI/LDRQ/tree/main).

<p align="center">
  <a href="docs/media/ldrq_pitch.mp4"><img src="docs/media/cover.jpg" width="720" alt="LDRQ: LLM-Driven Robotic Query-to-action. The Unitree R1 walking on the real robot, beside its simulation in Isaac Lab"></a>
  <br><em>Click for the 3-minute video (<a href="docs/media/ldrq_pitch.mp4">docs/media/ldrq_pitch.mp4</a>): training, deployment,
  the sim2real measurement, and English commands on the real robot.</em>
</p>

## v2 (this branch)

### What v2 is

v2 is v1 with a retrained walking policy, `v2_drift_ft` (run
`2026-10-16_17-57-14_v2_drift_ft`):
- it walks **backward** (to 0.6 m/s) and **sideways** (±0.5 m/s), and turns on the spot at
  up to 1.57 rad/s, three times faster than v1;
- on a zero command it steps in place without creeping forward: 0.14 m per 10 s in
  simulation, against v1's 0.45 m;
- it keeps v1's interface (425 inputs → 24 joint targets at 50 Hz) and gains, so the
  bridge's C++, the joint map and the install path are unchanged.

Around the policy:
- the command layer walks in four directions (`walk backward 1m`, `walk left 3s`), from v2's
  command range, which the installer writes into the bridge's configuration;
- the language model moves to the CPU (4 threads), leaving the GPU to the control loop;
- the vision node from stage E (`perception/`, YOLOX-Nano on the head camera's stream)
  is in the code, for the steps `turn_to` and `walk_to`. Its engine and its coexistence
  with the policy passed on the robot; the two steps themselves have not run there yet,
  so `perception` is `off` in the quick start, as in the release.

Released as [v2.0.0](https://github.com/kedianwen/LDRQ/releases/tag/v2.0.0) and deployed on
the robot since 2026-10-20. This branch is that release. The full write-up is
[docs/v2_policy.md](docs/v2_policy.md).

### Architecture

![v2 architecture](docs/system_architecture_v2.svg)

An English sentence becomes a plan once, before the robot moves: the model only
transcribes, deterministic code checks the plan against what the policy was trained on,
and the operator confirms. Below that, the command layer (10 Hz), the policy (50 Hz) and
the motor loop (500 Hz) each have their own safety checks.

Compared with v1:
- one more policy file, and a wider command range in the bridge;
- the model on the CPU: with the detector and the policy sharing the GPU, the policy's
  inference p99 stayed under 0.6 ms;
- the vision node, which reads the head camera's left image over the robot's network
  (UDP RTP, H264, port 5002), decodes it on the Jetson's hardware decoder and runs
  YOLOX-Nano as a TensorRT FP16 engine (3.7 ms per frame).

Source: [docs/system_architecture_v2.dot](docs/system_architecture_v2.dot). The page
[docs/system_architecture.md](docs/system_architecture.md) explains the v1 system of
2026-09-30, which is the same below the command layer; the vision node is
[docs/stageE_perception.md](docs/stageE_perception.md).

### Development order

| step | dates (2026) | what was built | result |
|---|---|---|---|
| v1 (W01–W08, stages A–D) | to 10-02 | the v1 system and release v1.1.0; see `main` | — |
| stage E, dev box | to 10-14 | the vision node: YOLOX-Nano, tracker, `turn_to` / `walk_to` steps, the camera guard; English prompt with the two steps | tests and simulated trials on the dev box |
| stage E, robot | 10-14 | engine and parity on the Orin; the model moved to the CPU; coexistence of policy, detector and model | FP16 engine passes; policy p99 < 0.6 ms with everything running |
| v2, diagnosis | 10-16 | v1's creep on a zero command measured as a bias of the policy (+0.046 m/s, in simulation) | zero was the edge of the trained range |
| v2, training | 10-16 | four runs; the kept one trains the gait for 4000 iterations, then adds the drift penalty for 800 | creep 0.14 m per 10 s; 0 falls over 5 seeds × 256 envs |
| v2, package | 10-16 to 10-20 | bundle, command layer in four directions; the vision node reads the head camera's stream (`source: stereo`) | dev box: bundle verified, ONNX vs PyTorch 1.5e-5 |
| v2, robot | 10-20 | installed with `policy_pack`, engine rebuilt on the Orin; release v2.0.0 | walks backward and sideways, holds its place on a zero command |

### Key parameters

What changed from v1 is in bold; everything else is v1's (see `main`).

| parameter | value | where it is set |
|---|---|---|
| interface | 425 → 24 at 50 Hz, as v1 | `models/v2_drift_ft/interface/` |
| PD gains, `kp_scale` | as v1 (legs 100 / 2 ...); **1.3** on the robot | `interface/actuator_gains.json`, launch argument |
| **command range** | **vx −0.6 → 1.0 m/s, vy ±0.5 m/s, wz ±1.57 rad/s** | `COMMAND_RANGES_FINAL_V2` in [tasks/r1_flat/flat_v2_env_cfg.py](tasks/r1_flat/flat_v2_env_cfg.py); written into `bridge.yaml` by the installer |
| **zero-command envs** | **15 %** (v1: 2 %) | `flat_v2_env_cfg.py` |
| **tracking reward** | v1's kernel (std 0.5) **plus a narrow one, std 0.15, weight 0.5** | same |
| **drift penalty** | **planar velocity averaged over ~1 s on a zero command, weight −5, from iteration 4000 only** | same; `zero_command_drift` in `tasks/r1_flat/mdp` |
| **foot-gap penalty** | **feet closer than 0.07 m sideways** | same |
| **training** | **4000 + 800 iterations** (checkpoint `model_4799.pt`), 4096 envs, seed 42 | docstring of `flat_v2_env_cfg.py` |
| **command layer** | **cruise backward 0.3 m/s, sideways 0.2 m/s**; forward 0.4 m/s, turns 0.4 rad/s | `mission_ctl/config/mission.yaml` |
| **model placement** | **CPU, 4 threads** (`llm_num_gpu: 0`, `llm_num_thread: 4`) | same |
| vision (stage E) | YOLOX-Nano, TensorRT **FP16**; one frame per 4 policy steps (12.5 Hz); confidence 0.30; 3 frames to confirm; a walk stops for a person within 1.2 m or an obstacle within 0.6 m | `perception/config/perception.yaml` |
| camera | head stereo, left image, UDP RTP H264 on port 5002, 544 × 448; depth from `/dev/video-dep` (after `sudo unitree_depth_start`), which `walk_to` needs | same (`source: stereo`) |

### Intermediate files

| file | made by | in git |
|---|---|---|
| `logs/rsl_rl/r1_flat/2026-10-16_15-51-33_v2/` (stage 1), `..._17-57-14_v2_drift_ft/` (stage 2) | `scripts/train_r1.py` | no (GB); the kept policy is in `models/v2_drift_ft/` |
| `models/v2_drift_ft/model_4799.pt`, `policy.pt`, `policy.onnx`, `parity_fixture.bin` | training, `scripts/play_r1.py` export, the fixture tool | yes (`SHA256SUMS`) |
| `models/v2_drift_ft/interface/` (interface, gains) | `deploy/tools/dump_interface.py --run`, `dump_actuator_gains.py` | yes |
| command-grid tables (drift, backward, sideways, turns) | `scripts/eval_command_grid.py` | the numbers are in [docs/v2_policy.md](docs/v2_policy.md) |
| policy bundles `bundles/<run_id>/` (v2, and v1 for rollback) | `policy_pack/make_bundle.py --task-cfg tasks/r1_flat/flat_v2_env_cfg.py ...` | no; built into the quick-start archive |
| `deploy/artifacts/policy_fp32.plan`, `installed.json` | `install_bundle.sh`, **on the robot**, after parity | no |
| `bridge.yaml` command range | `install_bundle.sh` step 4, from the bundle's `command_envelope.json` | no: the committed file keeps v1's, and the installer rewrites it |
| `perception/models/yolox_nano.onnx`, `dog.jpg` | `perception/models/fetch_yolox.sh` (sha256-pinned download) | no; in the quick-start archive |
| `perception/models/yolox_fixture.bin` | `perception/tools/make_fixture.py` (YOLOX's sample image, as in stage E) | no; in the quick-start archive |
| `perception/models/yolox_fp16.plan` | `r1_build_engine --fp16`, **on the robot** | no |
| stage E results (engines, coexistence, simulated trials, English eval) | stage E tools | written up in [docs/stageE_perception.md](docs/stageE_perception.md) |

### Quick start

**Without a robot** (any machine with Python 3.8+; no GPU, no ROS):

```bash
git clone -b v2 https://github.com/NAPPING-AI/LDRQ.git && cd LDRQ

# every check that needs no GPU, ROS or robot: command layer, English front end,
# objects (stage E), perception, eval sets, policy packaging, both policies' SHA256,
# Markdown links, dry runs. CI runs the same script.
./run_tests.sh

# a plan with the v2 policy's command range: backward and sideways exist
printf '{"vx": [-0.6, 1.0], "vy": [-0.5, 0.5], "wz": [-1.57, 1.57]}' > /tmp/v2_env.json
python3 mission_ctl/r1_mission_cli.py run "walk backward 1m; walk left 2s; turn right 180@1.0" --dry-run --envelope /tmp/v2_env.json

# an object step in a simulated room: turn to the chair, walk to 1.2 m from it
python3 mission_ctl/r1_mission_cli.py run "walk_to the chair stop_at 1.2; turn left 90" --dry-run --world mission_ctl/tests/worlds/room.json
```

To try the English front end as well, run [Ollama](https://ollama.com) with
`ollama pull qwen3:1.7b`, then
`python3 mission_ctl/r1_mission_cli.py ask "walk backward for 3 seconds, then turn left" --dry-run --envelope /tmp/v2_env.json`.

**On the robot:** [quickstart/ldrq-v2-quickstart.tar.gz](quickstart/ldrq-v2-quickstart.tar.gz)
(about 9 MB) is everything v2 runs on the robot: the bridge and policy node as source,
built there; `policy_pack`; `mission_ctl`; the model server's scripts; the v2 policy bundle
and v1's for rollback; the vision node with YOLOX-Nano and its parity fixture; and a
self-test. The language model's runtime and weights (about 1.8 GB) are too large for git;
the guide says where to get them. [quickstart/make_quickstart.sh](quickstart/make_quickstart.sh)
rebuilds the archive from the committed tree.

### Robot guide

[quickstart/ROBOT_GUIDE.md](quickstart/ROBOT_GUIDE.md) (中文
[quickstart/ROBOT_GUIDE_zh.md](quickstart/ROBOT_GUIDE_zh.md)): from copying the archive
to the robot to walking backward and sideways on an English command, upgrading from v1,
rolling back, and the optional vision checks. Read
[*Running on the robot*](#running-on-the-robot) below first.

## Results at a glance

| what | result |
|---|---|
| speed tracking in simulation, 0.5–1.0 m/s | worst error **0.037 m/s** (gate 0.15), no falls |
| TensorRT on the robot against PyTorch | max difference **1.3e-5** over 512 inputs (gate 1e-3) |
| policy inference on the Orin, in the 50 Hz loop | **~0.45 ms** of a 20 ms step |
| continuous walking without the gantry | **≥ 79.7 s** |
| tolerated actuator stiffness, `kp_scale` | simulator **0.80–2.00+**, real robot **1.10–1.50**; the gap is posture, not joint tracking |
| closed-loop turns on the robot | 10/10 within **1.6°** by the IMU |
| English sentence → checked plan, held-out test sets | **77/80** on the Orin, 0.65 s median; 11/11 instructions right end to end |
| swapping the policy | **one command**, 8 checked steps; 3/3 bad bundles refused before any change |
| v2 policy: zero command, simulation | creep **0.45 → 0.14 m per 10 s** (v1 → v2); backward to 0.6 m/s, sideways ±0.5 m/s, turns to 1.4 rad/s, no falls |
| v2 on the robot | walks **backward and sideways** and holds its place on a zero command; installed by the same one command |

The full account is [docs/technical_report.md](docs/technical_report.md); how the work was
organized and what each step found is [docs/project_history.md](docs/project_history.md).

## Layout

| folder | what it is | built in | start with |
|---|---|---|---|
| `assets/` | the R1 robot definition: URDF, meshes, Isaac Lab `ArticulationCfg`; the USD is generated, not tracked | W01–W04 | [assets/r1/README.md](assets/r1/README.md) |
| `tasks/` | the flat-ground velocity-tracking task `Isaac-Velocity-Flat-R1-v0`: scene, observations, rewards, randomization, PPO config; `-v2`, its retrain for backward, sideways and no creep | W02–W04, v2 | [tasks/r1_flat/README.md](tasks/r1_flat/README.md) |
| `scripts/` | Isaac Lab tools: URDF→USD, inspection, train/play/evaluate, gait diagnosis, config-drift check, the simulated `kp_scale` sweep, figures, demo video | W01 → stage A | [scripts/README.md](scripts/README.md) |
| `experiments/` | one record per training run (config + final metrics), for comparing runs | W03–W04 | [experiments/runs.md](experiments/runs.md) |
| `models/` | **the two policies** (checkpoint, TorchScript, ONNX, parity fixture): v2, deployed since 2026-10-20, and v1 | W04, v2 | [models/v2_drift_ft/README.md](models/v2_drift_ft/README.md), [models/week04_nohead/README.md](models/week04_nohead/README.md) |
| `deploy/` | the on-robot runtime: TensorRT engine builder and parity check, the C++/ROS 2 policy node and hardware bridge, measurement probes | W05–W08 | [deploy/README.md](deploy/README.md) |
| `policy_pack/` | swap the deployed policy with one command: bundle on the dev box, eight checked install steps on the robot | stage D | [policy_pack/README.md](policy_pack/README.md) |
| `mission_ctl/` | the command layer: time / angle / speed plans over the bridge's topics, IMU-closed turns, `ask` for English, and its offline eval | stages B–C | [mission_ctl/README.md](mission_ctl/README.md) |
| `llm/` | the model server on the robot: Ollama for JetPack 5, health checks, the control-loop coexistence test | stage C | [llm/README.md](llm/README.md) |
| `perception/` | object detection (YOLOX-Nano, TensorRT) for `turn_to` / `walk_to`: the node, calibration and evaluation tools; robot pending | stage E | [perception/README.md](perception/README.md) |
| `release/` | the robot-side release: install and usage READMEs (English, 中文), the robot self-test, and `make_release.sh`, which builds the release assets from the committed tree | v1.1.0, v2.0.0 | [release/README.md](release/README.md) |
| `docs/` | results: the technical report, per-stage write-ups, figures, robot evidence | all | [docs/README.md](docs/README.md) |
| `quickstart/` | the v2 quick-start archive for the robot, the script that builds it, and the robot guide (English, 中文) | 2026-10-26 | [quickstart/ROBOT_GUIDE.md](quickstart/ROBOT_GUIDE.md) |

The W07 snapshot of the robot's deploy tree (Orin-built binaries and engines, extracted
to `~/kdw_deploy` on the robot) is a [GitHub Release](https://github.com/kedianwen/LDRQ/releases/tag/orin-snapshot-w07),
not a file in the repository; `bash deploy/tools/fetch_orin_snapshot.sh` downloads it and
checks its SHA256.

## Running on the robot

What a reader with an R1 needs to know before anything moves. Each item cost a session
when it was missed:

1. **Switch the robot to developer mode first.** Otherwise the factory controller keeps
   writing `rt/lowcmd` against ours, and the result looks like a sim2real gap
   (`deploy/tools/probe_cpp/probe_lowcmd` checks for a second writer).
2. **Build the TensorRT engine on the Orin and pass `r1_parity_check` there**, every
   time. Engines are not portable, and a wrong engine runs at full speed with no error.
3. **Use a gantry** for the first runs, and keep the e-stop in reach. The bridge's output
   is off by default (`enable_output:=false`).
4. **Write numeric launch arguments as floats** (`kp_scale:=1.3`, `min_control_rate_hz:=55.0`);
   an integer kills the bridge at start-up.
5. **Lock the clocks before measuring latency** (`deploy/tools/w08_preflight.sh --lock-clocks`).
6. **Copy code to the robot as `.tar.gz`, not `.zip`**: a zip drops the execute bits.

The robot runs JetPack 5.1.1, TensorRT 8.5.2, ROS 2 foxy and Python 3.8, all different
from the training machine; see [*Hardware target*](#hardware-target).

## Reproducing

What each result needs, and the command that regenerates it. "GPU" means the Isaac Lab
dev box below; "robot" means an R1 with the `deploy/` stack; "none" runs on any machine
with Python 3.8+.

**Once after cloning (GPU):** the robot's USD is a build artifact, not tracked —
generate it from the URDF before any Isaac Lab script, or every one of them fails with
`USD file not found ... assets/r1/usd/R1.usd`:

```bash
conda activate env_isaaclab
~/IsaacLab/isaaclab.sh -p scripts/convert_r1_urdf.py --headless
```

The conversion is done once it prints `[INFO] USD articulation has 26 joints:` and the
joint list (about a minute); on this machine Isaac Sim's shutdown can then spin for many
minutes. `assets/r1/usd/R1.usd` is already written at that point, so Ctrl-C is safe.
Also note that `isaaclab.sh` exits 0 even when the Python script raised, so check the
output (each script prints `[ok]` lines or writes its report), not the exit code.

| result | needs | command |
|---|---|---|
| unit tests: command layer, policy packaging, model checksums | none | `./run_tests.sh` |
| stage A figure, from the committed data | none (matplotlib) | `python3 scripts/plot_gain_sweep.py --sim docs/stageA_kp_sweep_sim.json --real docs/stageA_kp_sweep_real.json --out docs/stageA_kp_sweep.png` |
| training config still matches the deployed run | GPU | `~/IsaacLab/isaaclab.sh -p scripts/verify_repro.py --headless --run 2026-08-19_11-03-32_week04_nohead` |
| speed-tracking table (PG-1) | GPU | `~/IsaacLab/isaaclab.sh -p scripts/eval_baseline.py --headless --num_envs 64 --checkpoint models/week04_nohead/model_2999.pt` |
| simulated half of the `kp_scale` sweep | GPU (~3 min) | `~/IsaacLab/isaaclab.sh -p scripts/sweep_gain_robustness.py --headless` |
| simulation clip for the demo | GPU | `~/IsaacLab/isaaclab.sh -p scripts/record_demo_sim.py --headless --out outputs/demo/sim_sequence.mp4` |
| demo video | none (ffmpeg) | `python3 scripts/make_demo_video.py --out ... --clip ... --figure docs/stageA_kp_sweep_wide.png` (see its docstring) |
| retrain the policy | GPU (~1.5 h) | see *Reproducing a training run* below |
| retrain v2 (two stages) | GPU (~2.5 h) | [docs/v2_policy.md](docs/v2_policy.md#reproducing) |
| v2 command grid: zero-command drift, backward, sideways, turns | GPU | `~/IsaacLab/isaaclab.sh -p scripts/eval_command_grid.py --headless --num_envs 64 --task Isaac-Velocity-Flat-R1-Play-v2 --checkpoint models/v2_drift_ft/model_4799.pt` |
| engine + parity on the robot | robot | `deploy/README.md` *Quick start* |
| real half of the `kp_scale` sweep | robot | `python3 deploy/tools/probe_cpp/gain_sweep_real.py --plan`, then `--record <KP>` per point, then `--collect` |

**Not in git, deliberately:** training logs (`logs/`, several GB — the deployed run's
artifacts are in `models/` instead), TensorRT engines (not portable; rebuilt on each
target), and the raw real-robot recordings (~80 MB; the reduced per-recording metrics are
`docs/stageA_kp_sweep_real.json`).

## Reproducing a training run

Everything needed to reproduce a run is pinned: the seed is set explicitly in
`tasks/r1_flat/agents/rsl_rl_ppo_cfg.py` (not inherited from an rsl_rl default),
and Isaac Lab dumps the fully-resolved configs to
`logs/rsl_rl/r1_flat/<run_id>/params/{env,agent}.yaml` at launch. Each archived
run in `experiments/<run_id>/` keeps those configs plus a `pip freeze` snapshot.

**Where the hyperparameters actually live.** They are Python dataclasses
(`tasks/r1_flat/flat_env_cfg.py`, `.../agents/rsl_rl_ppo_cfg.py`), and the
`params/*.yaml` beside each run are a *dump* of the resolved config — editing
that yaml changes nothing. This is Isaac Lab's design, and it keeps type
checking and IDE navigation over the config, but on its own a dump is weaker
than a spec: it records what happened without anything checking that the code
still produces it.

`scripts/verify_repro.py` closes that. It rebuilds the config from current code
and diffs it field by field against a run's archived yaml, so the dump becomes a
contract the repository is tested against:

```bash
~/IsaacLab/isaaclab.sh -p scripts/verify_repro.py --headless \
    --run 2026-08-19_11-03-32_week04_nohead
# RESULT: PASS -- current code reproduces this run's configuration exactly
```

It exits non-zero on drift and writes `experiments/<run_id>/repro_check.txt`.
Fields that vary per invocation (env count, device, run name) are reported
separately from real drift, and values the scene resolves at construction time
(`{ENV_REGEX_NS}` placeholders, terrain env count/spacing) are normalized rather
than ignored. Run it against a superseded run to see it work: `week04_hwspec`
reports exactly the two fields the head removal changed.

```bash
conda activate env_isaaclab

# train (W04 final config: hardware-spec actuators + obs history +
# domain randomization + command curriculum + symmetry augmentation, head not
# actuated -> 24-dim action, 425-dim policy observation).
# A single run -- the command curriculum counts env steps, so splitting this
# into train-then-resume shifts the ramp and does not reproduce the same curve.
~/IsaacLab/isaaclab.sh -p scripts/train_r1.py --headless \
    --num_envs 4096 --max_iterations 3000 --run_name week04_nohead

# watch it
tensorboard --logdir logs/rsl_rl/r1_flat

# evaluate: speed-tracking table (PG-1), then per-foot gait statistics
~/IsaacLab/isaaclab.sh -p scripts/eval_baseline.py --headless --num_envs 64 \
    --checkpoint logs/rsl_rl/r1_flat/<run_id>/model_2999.pt
~/IsaacLab/isaaclab.sh -p scripts/diagnose_gait.py --headless --num_envs 32 \
    --checkpoint logs/rsl_rl/r1_flat/<run_id>/model_2999.pt

# archive config + final metrics into experiments/
python scripts/archive_run.py <run_id>
```

`--resume` re-interprets `--max_iterations` as an *absolute* target rather than
an additional count — see `scripts/train_r1.py`'s docstring.

## Environment

Isaac Lab 2.1.0 + Isaac Sim 4.5.0 + torch 2.5.1+cu118, conda env `env_isaaclab`.
All scripts run through `~/IsaacLab/isaaclab.sh -p ...` from this directory (the
project root) — see each script's docstring/header for the exact invocation.

## Hardware target

R1 is the **EDU version** with an onboard 8-core Jetson Orin NX (40-100 TOPS), so
the policy runs on the robot itself with no external compute board.

The robot's software stack is **not** this dev box's, and every difference has
bitten at least once: JetPack 5.1.1, **TensorRT 8.5.2**, **ROS 2 foxy**, CUDA
11.4, and `unitree_hg` messages. A TensorRT `.plan` is tied to the TensorRT
version, GPU architecture and driver that built it, so **the artifact that ships
to the robot is the ONNX, and the engine is built on the Orin**. `deploy/` keeps
both toolchains working from one tree; see `deploy/README.md`.

## Project history

The work ran as a 12-week plan (weeks **W01–W08**, then four **stages A–D** in place of
W09–W12) in three milestones, followed by stage E (vision, open) and the v2 policy, with pass/fail gates such as **PG-2** and requirements such
as **FR-Q3**. Those labels appear in commits and documents; all of them, the timeline, the
status of each stage and the finding from each week that no metric showed are in
[docs/project_history.md](docs/project_history.md). The definition-of-done checklist with
evidence is [docs/dod.md](docs/dod.md).

## License

Apache License 2.0 ([LICENSE](LICENSE)), except the R1 URDF and meshes in `assets/r1/`,
which are Unitree Robotics' files under their BSD 3-Clause license
([assets/r1/LICENSE](assets/r1/LICENSE)), and files adapted from Isaac Lab, which keep its
BSD 3-Clause header. Details: [NOTICE](NOTICE).
