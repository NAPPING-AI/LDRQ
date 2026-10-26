# LDRQ: a Unitree R1 humanoid from simulation to English commands

[![tests](https://github.com/kedianwen/LDRQ/actions/workflows/tests.yml/badge.svg)](https://github.com/kedianwen/LDRQ/actions/workflows/tests.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![release](https://img.shields.io/github/v/release/kedianwen/LDRQ)](https://github.com/kedianwen/LDRQ/releases)

[中文简介](README_zh.md)

A walking policy for the **Unitree R1** humanoid, trained in Isaac Lab, deployed on the
robot's own Jetson Orin NX, measured against the simulator, and commanded in English
through a small language model running on the robot.

<p align="center">
  <a href="docs/media/ldrq_pitch.mp4"><img src="docs/media/cover.jpg" width="720" alt="LDRQ: LLM-Driven Robotic Query-to-action. The Unitree R1 walking on the real robot, beside its simulation in Isaac Lab"></a>
  <br><em>Click for the 3-minute video (<a href="docs/media/ldrq_pitch.mp4">docs/media/ldrq_pitch.mp4</a>): training, deployment,
  the sim2real measurement, and English commands on the real robot.</em>
</p>

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
| V3: a dance and a person reflex, simulation | handing the dancing robot to v2: **0 falls** in 2017 normal and 2044 emergency handovers |
| V3 on the robot (gantry, motors on) | dances on request, ends on a bar, steps back when a person comes within **0.9 m**, refuses to start with someone within 1 m; both policies **≤ 2.7 ms** per step together (p99) |

The full account is [docs/technical_report.md](docs/technical_report.md); how the work was
organized and what each step found is [docs/project_history.md](docs/project_history.md).

## Versions

| version | where | what changed | status |
|---|---|---|---|
| v1 | release [v1.1.0](https://github.com/kedianwen/LDRQ/releases/tag/v1.1.0) | the W04 walking policy, the robot runtime, the English command layer, one-command policy install | released |
| v2 | release [v2.0.0](https://github.com/kedianwen/LDRQ/releases/tag/v2.0.0), branch `master` | a retrained walking policy: no creep on a zero command, backward and sideways walking | released, deployed on the robot since 2026-10-20 |
| V3 | branch `v3`, directory [v3/](v3/README.md) | a second policy that dances, and a reflex that hands the robot to v2 to step back from a person | in development; ran on the robot on the gantry, 2026-10-24 |

**v2: walking that holds still and goes backward** ([docs/v2_policy.md](docs/v2_policy.md)).
On a zero command, v1 crept forward about 0.45 m every 10 s. Zero was the edge of its
trained range (forward only), and only 2 % of its training environments ever got a zero
command. v2 is a two-stage retrain on the same observation and action interface:
1. The command range is widened to span zero (backward to −0.6 m/s, sideways ±0.5 m/s,
   turns ±1.57 rad/s), and 15 % of environments get a zero command.
2. A penalty on the drift (the velocity averaged over about a second) is added only after
   the gait has formed. Added from the start, it taught the policy to stand still and
   ignore small commands.

Results in simulation: creep falls to 0.14 m per 10 s; the policy walks backward and
sideways and turns three times faster than v1, with no falls. v2 passes through the same one-command install as v1 (engine
built on the Orin, gated by parity). On the robot it walks backward and sideways and holds
its place on a zero command. Everything above the policy is unchanged, except that the
command layer reads v2's wider command range.

**V3: a dance, and stepping back from people** ([v3/README.md](v3/README.md)). Two
policies run side by side in the same process on the Orin, and both are inferred every
control step:
- **v2** for walking;
- **a dance policy** ([v3/models/dance_wave](v3/models/dance_wave/README.md)), trained with
  BeyondMimic's motion tracking to follow a 10 s routine on the spot.

A state machine in the hardware bridge (the conductor) chooses which policy's output reaches
the motors:
- `ask "dance for 20 seconds"` starts a dance only if nobody is within 1 m and the robot
  has stood still for 1 s;
- a normal end waits for the end of a bar;
- if a person comes within 0.9 m of the dancing robot, v2 takes over on the same control
  step, steps back for 1.5 s (about 0.33 m) and stands until the operator confirms a new
  plan;
- the same reflex protects walking and standing, triggering at 0.6 m.

The person's distance comes from the size of their detection box. The head camera's
depth is filled in up close, so it reports obstacles that are not there. Depth computed
from the two head images is under way, for obstacles. All V3 code lives under `v3/`, so
v1 and v2 stay reproducible as released.

## How it fits together

![System architecture with V3](docs/system_architecture_v3.svg)

An English sentence becomes a plan once, before the robot moves: the model only
transcribes, deterministic code checks the plan against what the policy was trained on,
and the operator confirms. Below that, the command layer (10 Hz), the policies (50 Hz) and
the motor loop (500 Hz) each have their own safety checks. The diagram is the V3 system.
v2 is the same picture without the dance policy and the conductor
([v2-era diagram](docs/system_architecture.svg); details and the logic diagram:
[docs/system_architecture.md](docs/system_architecture.md)).

**Vision** ([perception/](perception/README.md), [docs/stageE_perception.md](docs/stageE_perception.md);
V3's fork [v3/perception/](v3/perception/README.md)). The R1 EDU has no camera wired to
the Orin. The head's stereo module streams its images to it over the robot's network.
The `r1_perception` node works in four steps:
1. It receives the left camera's H264 stream (UDP RTP, port 5002, 544 × 448, about
   10 Hz) and decodes it with the Jetson's hardware decoder.
2. It runs YOLOX-Nano on every frame with a TensorRT FP16 engine built on the Orin: 3.7 ms
   per frame, with detections identical to FP32's. The detector shares the GPU with the
   walking policy, and the policy's p99 stays under 0.6 ms with both running.
3. It confirms a detection only after several hits in a row.
4. It publishes two topics:
   - `/r1_perception/detections` (JSON): each object's bearing and range, for the steps
     `turn_to` ("face the chair") and `walk_to` ("walk to the chair and stop 1.5 m from
     it"), and a guard that stops a walk when someone is close or the camera goes quiet;
   - `/r1_perception/proximity`, which V3's reflex acts on.

A person's distance comes from the height of their detection box. That ratio is
calibrated without a tape measure: a helper stands at arm's length, then one step further.
The module's own depth cannot be used up close: it fills the pixels it cannot match with
near values. Depth computed from the left and right images is in development.

Status:
- the person reflex runs on the robot (V3, 2026-10-24);
- the engine, parity and coexistence with the policy passed on the robot;
- `turn_to` and `walk_to` are tested on the dev box, and their robot runs are pending.

## Quick start

Pick the row that matches what you have. Everything in the first two rows runs on any
machine with Python 3.8+, with no GPU, no ROS and no robot.

| you want to… | do this | time |
|---|---|---|
| **understand the project** | read [docs/technical_report.md](docs/technical_report.md) (the whole project in one document), then [docs/system_architecture.md](docs/system_architecture.md) (how a sentence becomes motor commands) | 30 min |
| **run something now** | the commands below | 2 min |
| **train or evaluate in simulation** | an NVIDIA GPU with Isaac Lab 2.1 / Isaac Sim 4.5 → [*Reproducing*](#reproducing) and [*Reproducing a training run*](#reproducing-a-training-run) | 1 h to set up, 1.5 h per training run |
| **run it on an R1** | an R1 EDU (onboard Orin NX) → download the robot release ([Releases](https://github.com/kedianwen/LDRQ/releases): `release_v2.0.0.tar.gz` + the model archive) and follow its README ([release/README.md](release/README.md), 中文 [release/README_zh.md](release/README_zh.md)); read [*Running on the robot*](#running-on-the-robot) first | an hour |

```bash
git clone https://github.com/kedianwen/LDRQ.git && cd LDRQ

# every check that needs no GPU, ROS or robot (~3 s, standard library only):
# command layer, English front end, eval sets, coexistence selftest,
# policy packaging, the deployed policy's SHA256, Markdown links, a dry run.
# CI runs the same script.
./run_tests.sh

# what the robot says it can do, generated from the deployed configuration
python3 mission_ctl/r1_mission_cli.py capability

# a plan, compiled, checked and simulated against a stand-in robot; nothing moves
python3 mission_ctl/r1_mission_cli.py run "walk 3s@0.3; turn left 90" --dry-run

# the same with the v2 policy's command range: backward and sideways exist
printf '{"vx": [-0.6, 1.0], "vy": [-0.5, 0.5], "wz": [-1.57, 1.57]}' > /tmp/v2_env.json
python3 mission_ctl/r1_mission_cli.py run "walk backward 1m; walk left 2s" --dry-run --envelope /tmp/v2_env.json
```

To try the English front end as well, run [Ollama](https://ollama.com) with
`ollama pull qwen3:1.7b`, then
`python3 mission_ctl/r1_mission_cli.py ask "walk forward for 3 seconds, then turn left" --dry-run`.
It prints the parsed plan, or the reason it refuses. The plotting and video tools need
`pip install -r requirements.txt`; [requirements.txt](requirements.txt) also says where
the simulation, TensorRT and robot dependencies come from.

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
| `perception/` | vision: the head camera's stream, object detection (YOLOX-Nano, TensorRT FP16) for `turn_to` / `walk_to`, the node, calibration and evaluation tools; engine and coexistence passed on the robot, the two steps pending there | stage E | [perception/README.md](perception/README.md) |
| `release/` | the robot-side release: install and usage READMEs (English, 中文), the robot self-test, and `make_release.sh`, which builds the release assets from the committed tree | v1.1.0, v2.0.0 | [release/README.md](release/README.md) |
| `docs/` | results: the technical report, per-stage write-ups, figures, robot evidence | all | [docs/README.md](docs/README.md) |
| `v3/` | V3, isolated from v1/v2: the dance policy and its training, the two-policy runner and bridge (forks), the proximity reflex, mission_ctl with a `dance` step, robot install and update scripts | V3 | [v3/README.md](v3/README.md) |

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
| training config still matches v1's run | GPU | `~/IsaacLab/isaaclab.sh -p scripts/verify_repro.py --headless --run 2026-08-19_11-03-32_week04_nohead` |
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
