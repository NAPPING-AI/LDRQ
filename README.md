# LDRQ V3: a Unitree R1 humanoid that dances on request and steps back from people

[![tests](https://github.com/NAPPING-AI/LDRQ/actions/workflows/tests.yml/badge.svg)](https://github.com/NAPPING-AI/LDRQ/actions/workflows/tests.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[中文](README_zh.md)

A walking policy for the **Unitree R1** humanoid, trained in Isaac Lab, deployed on the
robot's own Jetson Orin NX, measured against the simulator, and commanded in English
through a small language model running on the robot. V3 adds a second policy that dances,
and a reflex that steps the robot back from a person who comes too close.

> **This is the `v3` branch.** The project has three versions, one per branch: `main`
> (v1), `v2` and `v3` (V3, this one). The overview of all three, with each version's
> architecture, is the README on [`main`](https://github.com/NAPPING-AI/LDRQ/tree/main).

<p align="center">
  <a href="docs/media/ldrq_pitch.mp4"><img src="docs/media/cover.jpg" width="720" alt="LDRQ: LLM-Driven Robotic Query-to-action. The Unitree R1 walking on the real robot, beside its simulation in Isaac Lab"></a>
  <br><em>Click for the 3-minute video (<a href="docs/media/ldrq_pitch.mp4">docs/media/ldrq_pitch.mp4</a>): training, deployment,
  the sim2real measurement, and English commands on the real robot.</em>
</p>

## V3 (this branch)

### What V3 is

V3 is v2 with three additions. v2's walking policy is unchanged.
- **A dance policy**, `dance_wave` (run `2026-10-23_16-51-44_wave`), trained with
  BeyondMimic's motion tracking to follow a 10 s routine on the spot: the right arm waves,
  then the left, both arms up with two knee dips, then standing still. Both feet stay on
  the floor.
- **Two policies in one process.** The policy runner infers v2 and the dance policy on
  every control step, in one CUDA context. A state machine in the bridge, the
  *conductor*, decides which policy's output reaches the motors.
- **A person reflex** in the bridge, once per control step. When a person comes within
  0.6 m (0.9 m while dancing), v2 takes over on that same step, backs off for 1.5 s (about
  0.33 m) and stands until the operator confirms a new plan. Neither the language model nor
  the command layer is in this loop.

For the operator, `ask "dance for 20 seconds"`:
- starts a dance only if nobody is within 1 m and the robot has stood still for 1 s;
- ends it at the end of a bar;
- hands the robot to v2 to step back if someone comes close;
- `stop dancing` or Ctrl-C ends it at the next bar.

**Status: complete.** On 2026-10-24 V3 ran on the robot in the gantry, first with the
output off, then with the motors on. The bridge stayed RUNNING throughout. The robot danced
and handed back to v2 at the end of a bar, stopped at the next bar on Ctrl-C, and stepped
back from the dance when a person came within 0.79–0.88 m. A dance asked for with someone
too close was refused. These gantry runs completed V3, and with it the project's
engineering (2026-10-26). Left out, and known limits of what was delivered:
- V3 has run in the gantry only, never untethered;
- the reflex acts on people only: obstacles would need depth computed from the two head
  images, which was started on the dev box and not finished;
- one `DEGRADED: joint_target stale`, seen once with the output off, was not explained.

All V3 code lives under [v3/](v3/README.md), whose README is the detailed account.

### Architecture

![V3 architecture](docs/system_architecture_v3.svg)

An English sentence becomes a plan once, before the robot moves: the model only
transcribes, deterministic code checks the plan, and the operator confirms. Below that,
the command layer (10 Hz), the policies (50 Hz) and the motor loop (500 Hz) each have
their own safety checks.

What V3 adds to v2's system:
- **mission_ctl** has a `dance` step. It sends the dance request as a heartbeat, and ends
  the plan when the reflex steps the robot back.
- **The bridge** carries the reflex (OFF, ARMED, RETREAT, STAND) and the conductor. Each
  control step it reads `/r1_perception/proximity` and writes the velocity command into
  v2's observation. It then passes on the targets of the policy that drives.
- **The runner** builds both observations and runs both engines every step: at most
  2.7 ms together at p99, against a 4 ms budget.

Source: [docs/system_architecture_v3.dot](docs/system_architecture_v3.dot). The v2 system
of 2026-10-20 is drawn on the `v2` branch. The page
[docs/system_architecture.md](docs/system_architecture.md) explains the v1 system of
2026-09-30, which is the same below the command layer.

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
near values. Depth computed from the left and right images was started on the dev box and
not finished.

### Development order

| step | dates (2026) | what was built | result |
|---|---|---|---|
| v1, v2 | to 10-20 | the walking system and the v2 policy; see `main` and `v2` | — |
| forks | 10-22 | `v3/`: forks of perception, mission_ctl, the bridge and the runner from release v2.0.0, run on the v2 policy | v1 and v2 stay reproducible as released |
| retreat in simulation | 10-22 | how v2 answers a −0.3 m/s command from standing, turning and walking; the speed-dependent stop distance | 0 falls; 0.33 m from standing |
| the reflex | 10-22 | reference state machine (Python), its golden ticks, the bridge's C++ port, mission_ctl's link, perception's proximity topic | C++ matches the reference tick for tick |
| first robot session | 10-23 | the head camera's depth fills near pixels: phantom obstacles at 0.05–0.35 m | people only, distance from box height (`person_box`), calibrated without tape |
| the dance | 10-23 | the routine and its checks on the URDF; BeyondMimic tracking task; `dance_wave` (6000 iterations, 4096 envs); export, fixture | ONNX parity 5.3e-5 (dev box) |
| two policies | 10-23 | the conductor (reference and C++), the runner with two engines, the handover in simulation, the `dance` step and its eval sets | 0 falls in 2017 normal and 2044 emergency handovers |
| robot install | 10-24 | `install_v3.sh` makes `~/r1-robot-v3` from the v2 tree; both engines built and gated on the Orin | parity v2 2.3e-05, dance 4.6e-05 |
| robot, output off | 10-24 | two engines' timing, the dance, a person walking up, refusals | ≤ 2.7 ms per step together (p99); handover on the same step |
| robot, motors on (gantry) | 10-24 | the dance, Ctrl-C, a person walking up, refusal | RUNNING throughout; stepped back at 0.79–0.88 m |
| fixes | 10-24 to 10-26 | the reply for a dance step; the dry run's bridge stepped at 50 Hz; depth from the two images (dev box, started, not finished) | — |
| close | 10-26 | the gantry runs accepted as the end of the engineering work; release v3.0.1 | project complete |
| head fix | 10-28 | the bridge holds the head where it is when driving starts (at the URDF zero it turned ~43° right); release v3.0.2 | head still on the robot, gantry, motors on |

### Key parameters

| parameter | value | where it is set |
|---|---|---|
| walking policy | v2 (`v2_drift_ft`), unchanged: 425 → 24 at 50 Hz | `models/v2_drift_ft/` |
| dance policy | 137 inputs (no history) → 24 joint targets at 50 Hz; v2's action joints, scales, default pose and gains | [v3/models/dance_wave/interface.json](v3/models/dance_wave/interface.json) |
| dance routine | a 10 s loop of 4 bars; handover only at a bar boundary | `v3/dance/routine.py` |
| dance training | BeyondMimic tracking, Isaac Lab 2.1.0, 6000 iterations, 4096 envs, v2's randomization added | `v3/dance/r1_tracking.py`, `train_dance.py` |
| `kp_scale` on the robot | 1.3, as v2 (one set of gains for both policies) | launch argument |
| reflex: retreat | person within **0.6 m** (obstacle 0.45 m) on 2 frames in a row → (−0.3, 0, 0) m/s for 1.5 s, then stand until a reset | `v3/ros2_ws/src/r1_hw_bridge/config/bridge.yaml` `reflex_*` |
| reflex: limits | at most 2 retreats in 10 s; another needs the person beyond trigger + 0.2 m; frames older than 0.3 s block forward walking | same |
| stop distance | max(floor, trigger + 0.10 + 1.5 · v · latency + 1.3 · brake(v)); floor person 1.2 m, obstacle 0.6 m; latency 0.25 s | same; `v3/reflex/stop_distance.py` |
| conductor | start: nobody within **1.0 m** and 1 s standing still; while dancing the reflex triggers at **0.9 m**; heartbeat 0.5 s; at most 63 s | `conductor.hpp` defaults (`dance_*` parameters) |
| dance step | 5 to 60 s, 20 s if not said; refused without the reflex | `v3/mission_ctl/config/mission.yaml` (`dance_*`) |
| person distance | `person_box`: distance = 0.465 / (box height ÷ image height); a box touching top and bottom = right in front | `v3/perception/config/perception.yaml` (`person_box_k`, calibrated 2026-10-24) |
| reflex in mission_ctl | `reflex: on`, `reflex_obstacles: off` (people only) | `mission.yaml` |
| launch | `reflex_enable:=true dance_enable:=true dance_engine:=... dance_reference:=...` (both default false) | `r1_stack.launch.py` |

### Intermediate files

| file | made by | in git |
|---|---|---|
| `v3/dance/` routine CSV → BeyondMimic motion `.npz` (with a replay video) | `routine.py`, `csv_to_npz_r1.py` (Isaac) | no, regenerated |
| `logs/rsl_rl/r1_dance_v3/2026-10-23_16-51-44_wave/` | `v3/dance/train_dance.py` | no (GB) |
| `v3/models/dance_wave/` `model_5999.pt`, `policy.pt`, `policy.onnx`, `interface.json` | training, `export_dance.py` | yes (`SHA256SUMS`) |
| `v3/models/dance_wave/dance_ref.bin` | the reference the runner plays: one loop, 500 frames | yes |
| `v3/models/dance_wave/parity_fixture.bin` | `deploy/tools/make_fixture.py --in-dim 137` | yes |
| `v3/reflex/golden.csv`, `golden_dance.csv` | `reflex/cases.py`, `dance_cases.py`: every tick of every case, the contract the C++ is tested against | yes |
| retreat and handover tables | `v3/sim/eval_retreat.py`, `eval_handover.py`, `summarize_retreat.py` | the numbers are in [v3/README.md](v3/README.md) |
| `r1_v3_<n>_<date>.tar.gz` | `v3/tools/make_robot_package.sh` | no; the quick-start archive holds one |
| `~/r1-robot-v3/` (a new tree), `v3_models/dance_wave/policy_fp32.plan` | `install_v3.sh`, **on the robot**, gated by parity | no |
| `perception/models/yolox_fp16.plan` | `r1_build_engine --fp16`, **on the robot** | no |
| `~/orin_commissioning/v3*_stack.log`, `*_reflex_log.csv` (one line per control step), `*_perception.log` | the stack and the perception node on the robot | no |
| stereo recordings (`stereo_capture.py`) | the head cameras' RTP packets, for the two-image depth | no: they can show people's faces |

### Quick start

**Without a robot** (any machine with Python 3.8+ and numpy; no GPU, no ROS):

```bash
git clone -b v3 https://github.com/NAPPING-AI/LDRQ.git && cd LDRQ
./run_tests.sh                        # v1 and v2: everything that needs no GPU, ROS or robot
cd v3 && ./run_tests.sh               # V3: reflex, conductor, C++ against the golden files, dance, mission_ctl

# a dance in a simulated empty room, with the bridge's reflex and conductor simulated: DONE
python3 mission_ctl/r1_mission_cli.py run "dance 12s" --dry-run --world mission_ctl/tests/worlds/room.json
# the same while a person walks up: v2 takes over and steps back
python3 mission_ctl/r1_mission_cli.py run "dance 12s" --dry-run --world mission_ctl/tests/worlds/approach.json
```
The C++ checks in `v3/run_tests.sh` need g++ with C++17; the bridge itself builds on a
dev box against a fake robot (see [v3/README.md](v3/README.md#on-a-dev-box)).

**On the robot, from the release:** [v3.0.2](https://github.com/NAPPING-AI/LDRQ/releases/tag/v3.0.2)
has the code in one archive and one install script; the language model is v3.0.0's archive,
the same qwen3:1.7b ([v3.0.0](https://github.com/NAPPING-AI/LDRQ/releases/tag/v3.0.0)).
`install.sh` installs V3 the way the robot was installed on 2026-10-24. Its README
is [v3/release/README.md](v3/release/README.md) (中文 [README_zh.md](v3/release/README_zh.md)): install,
every session, commands, acceptance checks, calibration, troubleshooting, rollback.
`v3/release/make_release_v3.sh` builds it from the committed tree.

**On the robot, from this branch:** [quickstart/ldrq-v3-quickstart.tar.gz](quickstart/ldrq-v3-quickstart.tar.gz)
(about 17 MB) holds two parts:
- `base/`, the v2 robot tree: the bridge and policy node as source, `policy_pack`,
  `mission_ctl`, the model server's scripts, the v2 and v1 bundles, and the vision node
  with YOLOX-Nano and its parity fixture;
- `v3_package/`, V3 as `v3/tools/make_robot_package.sh` packs it, with the dance policy:
  `install_v3.sh`, the V3 code in the robot tree's layout, and the hash lists the
  installer checks.

This is the path the robot took on 2026-10-24: v2 installed in a tree, then `install_v3.sh`
makes a new tree, `~/r1-robot-v3`, from it, and leaves the v2 tree as it was. The same
archive passed `install_v3.sh --rehearsal` on the dev box: all 8 steps, the source tree
unchanged, V3's tests green in the new tree. The language model's runtime and weights
(about 1.8 GB) are too large for git; the guide says where to get them.
[quickstart/make_quickstart.sh](quickstart/make_quickstart.sh) rebuilds the archive from the
committed tree.

### Robot guide

[quickstart/ROBOT_GUIDE.md](quickstart/ROBOT_GUIDE.md) (中文
[quickstart/ROBOT_GUIDE_zh.md](quickstart/ROBOT_GUIDE_zh.md)) covers the full path:
1. the v2 base, the vision engine and V3's install;
2. the camera and the tape-free calibration;
3. the checks with the output off;
4. the checks in the gantry with the motors on.

Each step says what to check and when to stop. Read
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
| V3: a dance and a person reflex, simulation | handing the dancing robot to v2: **0 falls** in 2017 normal and 2044 emergency handovers |
| V3 on the robot (gantry, motors on) | dances on request, ends on a bar, steps back when a person comes within **0.9 m**, refuses to start with someone within 1 m; both policies **≤ 2.7 ms** per step together (p99) |

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
| `perception/` | vision: the head camera's stream, object detection (YOLOX-Nano, TensorRT FP16) for `turn_to` / `walk_to`, the node, calibration and evaluation tools; engine and coexistence passed on the robot, the two steps pending there | stage E | [perception/README.md](perception/README.md) |
| `release/` | the robot-side release: install and usage READMEs (English, 中文), the robot self-test, and `make_release.sh`, which builds the release assets from the committed tree | v1.1.0, v2.0.0 | [release/README.md](release/README.md) |
| `docs/` | results: the technical report, per-stage write-ups, figures, robot evidence | all | [docs/README.md](docs/README.md) |
| `v3/` | V3, isolated from v1/v2: the dance policy and its training, the two-policy runner and bridge (forks), the proximity reflex, mission_ctl with a `dance` step, robot install and update scripts | V3 | [v3/README.md](v3/README.md) |
| `quickstart/` | the V3 quick-start archive for the robot (the v2 base and the V3 package), the script that builds it, and the robot guide (English, 中文) | 2026-10-26 | [quickstart/ROBOT_GUIDE.md](quickstart/ROBOT_GUIDE.md) |

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
W09–W12) in three milestones, followed by stage E (vision), the v2 policy and V3 (a dance and the person reflex); the
engineering closed on 2026-10-26. The plan used pass/fail gates such as **PG-2** and requirements such
as **FR-Q3**. Those labels appear in commits and documents; all of them, the timeline, the
status of each stage and the finding from each week that no metric showed are in
[docs/project_history.md](docs/project_history.md). The definition-of-done checklist with
evidence is [docs/dod.md](docs/dod.md).

## License

Apache License 2.0 ([LICENSE](LICENSE)), except the R1 URDF and meshes in `assets/r1/`,
which are Unitree Robotics' files under their BSD 3-Clause license
([assets/r1/LICENSE](assets/r1/LICENSE)), and files adapted from Isaac Lab, which keep its
BSD 3-Clause header. Details: [NOTICE](NOTICE).
