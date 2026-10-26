# R1 robot stack v3.0.0 (V3): a dance, and stepping back from people

This archive is everything that runs **on the robot** (a Unitree R1 EDU, onboard Jetson
Orin NX) for V3. 中文说明: [README_zh.md](README_zh.md).

V3 is the v2 walking system with two additions:
- **A dance.** `ask "dance for 20 seconds"` runs a second policy, trained to follow a
  10 s routine on the spot: the right arm waves, then the left, both arms up with two knee
  dips, then standing still. Both feet stay on the floor.
- **A person reflex.** When a person comes within **0.6 m** of the robot (**0.9 m** while
  it dances), the v2 walking policy takes over on that same control step. It steps back
  for 1.5 s (about 0.33 m) and stands until the operator confirms a new plan.

Everything v2 does still works:
- walking forward, backward and sideways, and turning on the spot;
- English instructions through a small language model on the robot's CPU, checked by code
  and confirmed by the operator before anything moves.

Source, design notes and results: <https://github.com/NAPPING-AI/LDRQ>, branch `v3`
(`v3/README.md` is the detailed account).

> **Status: pre-release.** V3 has run on the robot **in a gantry only**, with the output off
> and with the motors on (2026-10-24). It has not run untethered. The reflex acts on
> **people only**, not on objects. The camera looks forward only, so the robot backs off
> into space it cannot see.
>
> **Read [Safety](#safety) before anything moves.** This is research code for a robot that
> can fall and can hurt people.

## Contents

| release asset | size | what it is |
|---|---|---|
| `release_v3.0.0.tar.gz` | about 0.5 GB | this README, `install.sh`, the code, both walking bundles (v2; v1 for rollback), the dance policy, YOLOX-Nano, the Ollama language-model server for JetPack 5 |
| `qwen3-1.7b_r1-robot-v3.0.0.tar.gz` | about 1.3 GB | the language model, qwen3:1.7b (Q4_K_M), in Ollama's format |
| `SHA256SUMS` | — | checksums of the two archives |

Both archives extract into the same directory, `r1-robot-v3.0.0/`:

```
r1-robot-v3.0.0/
  README.md, README_zh.md       this file (English / 中文)
  install.sh                    installs V3 into v3/ (below), in five checked steps
  selftest.sh                   checks that need no robot motion
  base/                         the v2 robot tree V3 is installed from, and the rollback:
    deploy/                       C++/ROS 2 bridge and TensorRT policy node (built on the robot)
    bundles/                      v2 (2026-10-16_17-57-14_v2_drift_ft) and v1 (rollback) walking policies
    policy_pack/                  installs a bundle: configs, build, engine, parity gate
    mission_ctl/                  the v2 command layer
    perception/                   the vision node; models/: YOLOX-Nano and its parity fixture
    llm/                          ollama_ctl.sh, the Ollama runtime, the model (second archive)
  v3_package/                   V3: install_v3.sh, the V3 code (bridge, runner, mission_ctl,
                                perception, reflex, dance), the dance policy, hash lists
  v3/                           created by install.sh: the V3 tree you run
  LICENSE, NOTICE, THIRD_PARTY_NOTICES.md
```

| policy | what it does | interface |
|---|---|---|
| `2026-10-16_17-57-14_v2_drift_ft` (v2, walking) | forward to 1.0 m/s, backward to 0.6 m/s, sideways ±0.5 m/s, turns to 1.57 rad/s; steps in place on a zero command | 425 → 24 joint targets, 50 Hz |
| `2026-10-23_16-51-44_wave` (dance) | the 10 s routine on the spot, looped; hands back to v2 only at the end of a bar | 137 → 24 joint targets, 50 Hz, same joints and gains as v2 |

## Requirements

| | |
|---|---|
| robot | **Unitree R1 EDU** with the onboard Jetson Orin NX 16 GB and the head stereo camera module |
| system | **JetPack 5.1.1** (L4T R35.3.1): TensorRT 8.5.2, CUDA 11.4, Ubuntu 20.04, Python 3.8 with numpy; g++ |
| ROS | **ROS 2 foxy** in `/opt/ros/foxy` |
| Unitree SDK | **unitree_sdk2** built and installed under `/usr/local` (`/usr/local/include/unitree/`, `/usr/local/lib/libunitree_sdk2.a`). The bridge is compiled against it. It is not part of this release |
| the Unitree app | on a phone: the head camera's stream has to be switched on after every boot |
| disk / memory | about 6 GB free disk; about 2.5 GB free memory while the model is loaded |
| network | none on the robot; a PC to download the release and copy it over SSH |
| people and gear | an operator with the emergency stop in hand; a helper who walks up to the robot; a gantry; 1 m clear behind the robot, 2.5 m clear in front |

## Install

**Next to other trees.** V3 installs into its own directory, `~/r1-robot-v3.0.0`. Trees
from earlier releases or development installs are left as they are:
`~/r1-robot-v1.1.0`, `~/r1-robot-v2.0.0` and `~/r1-robot-v3`. Stop every running stack and
every model server first. For example, `bash ~/r1-robot-v2.0.0/llm/ollama_ctl.sh stop`:
all servers use the same port.

**1. On a PC: download and check.**
```bash
sha256sum -c SHA256SUMS          # both archives: OK
scp release_v3.0.0.tar.gz qwen3-1.7b_r1-robot-v3.0.0.tar.gz unitree@<robot-ip>:~/
```
Copy them as `.tar.gz`. A zip loses the files' execute permissions.

**2. On the robot: extract both into the same place, and self-test.**
```bash
cd ~
tar xzf release_v3.0.0.tar.gz
tar xzf qwen3-1.7b_r1-robot-v3.0.0.tar.gz      # adds r1-robot-v3.0.0/base/llm/models/
bash ~/r1-robot-v3.0.0/selftest.sh             # every line "ok"; moves nothing
```
`selftest.sh` checks:
- every file against its hash list;
- the v2 tree's tests;
- the model and its runtime;
- V3's own tests: the reflex and the conductor against their golden files (the C++
  included), the dance observation, the vision code, the command layer and three dry runs.

Lines marked `warn` are about the robot's system (JetPack, ROS, unitree_sdk2, TensorRT).
They do not fail the test, but the install needs all four.

**3. Install (about 15 minutes; most of it builds engines).** Use a **new terminal that has
not sourced any `env.sh`**:
```bash
bash ~/r1-robot-v3.0.0/install.sh
```
It prints `[n/5]` for each step, and any failure stops it. The log is
`~/orin_commissioning/v3.0.0_install_<time>.log`.

| step | what it does | expect |
|---|---|---|
| 1 checks | a clean terminal, no stack running, `selftest.sh` | `selftest: OK` |
| 2 v2 policy | `base/policy_pack/install_bundle.sh` with the v2 bundle. It writes v2's command range, builds the bridge and the policy node in `base/`, builds the engine **on this robot**, and checks it against the bundle's reference | `PARITY PASS`, `max_abs` about 1e-05 (limit 1e-3) |
| 3 vision | builds the YOLOX-Nano engine (FP16, about 7 minutes) and checks it against its fixture | `PARITY PASS (detections gate ...)` |
| 4 V3 | `v3_package/install_v3.sh --src base --dst v3`, in eight sub-steps: copy `base/` → lay V3 over the copy → show the config differences → build → CycloneDDS check and V3's tests → the dance engine built here, and both engines' parity | `joint_map.hpp identical`; config diffs only the expected ones; `CycloneDDS from /usr/local/lib: OK`; two `PARITY PASS`; `source tree unchanged: OK` |
| 5 summary | where V3 is, the dance engine's fingerprint | **write down `fnv1a=0x...`**: the stack prints it at every start |

- Steps 2 and 3 are skipped if they were done before, so running `install.sh` again after a
  failure in step 4 is fine.
- Step 4 refuses an existing `v3/`. Move it away first:
  `mv ~/r1-robot-v3.0.0/v3 ~/r1-robot-v3.0.0/v3.old`.
- **Expected config differences** at step 4 (`[5/8]`):
  - `mission.yaml` gains `reflex: on`, `reflex_obstacles: off`, the `dance*` lines, and
    `obj_stop_min_m` 0.9;
  - `perception.yaml` gains `proximity_*` and `person_box_*`;
  - `bridge.yaml` gains the `reflex_*` block.

  Any other `<` line is a local change in `base/`: copy it into the file in `v3/` by hand.
- The engine fingerprints differ from build to build: TensorRT picks kernels by timing them.
  Parity is what proves the numbers.

## Every session

### Before the robot moves, every power-on

1. **Put the robot in developer mode** with its handheld remote. Otherwise Unitree's own
   motion controller keeps commanding the motors at the same time as this stack. The
   result looks like a bad policy: sway, tremor, grinding joints.
2. **Lock the clocks** (needs sudo, lasts until reboot):
   `bash ~/r1-robot-v3.0.0/v3/deploy/tools/w08_preflight.sh --lock-clocks`. It must report
   0 FAIL.
3. **Switch the head camera's stream on**, in the Unitree app: **video_hub off**, **Stereo
   patch PC1 on**. The app forgets this at every boot.
4. Hang the robot in the gantry, feet on the floor, a little slack in the rope.

### In every terminal, first

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1      # before env.sh, in EVERY terminal
source ~/r1-robot-v3.0.0/v3/deploy/env.sh
R=~/r1-robot-v3.0.0/v3; M=$R/mission_ctl/r1_mission_cli.py; PC=$R/perception/r1_perception_cli.py
PM=$R/perception/models; D=$R/v3_models/dance_wave; O=~/orin_commissioning; mkdir -p $O
```
Terminals with different `ROS_DOMAIN_ID` settings cannot see each other. The symptom is a
stack that seems to ignore every command.

### Check the camera stream

```bash
pgrep -af stereo_capture                                  # must print nothing; else kill <PID>
python3 $R/perception/tools/stereo_capture.py --probe     # 3 s: ports 5002 and 5003, about 10 fps each
```
0 packets: set the app again (video_hub off, Stereo patch PC1 on), then probe again. Do not
go on without frames: with the reflex on, no frames means no forward walking and no dance,
and on 2026-10-24 a session without frames ended in `DEGRADED`.

### Start the services

| terminal | what | command |
|---|---|---|
| 1 | the language model | `bash $R/llm/ollama_ctl.sh start && bash $R/llm/ollama_ctl.sh health` |
| 2 | vision | `cd $R && python3 $PC run --engine $PM/yolox_fp16.plan 2>&1 \| tee $O/v3_perception.log` |
| 3 | the stack (below) | |
| 4 | the lines that matter | `tail -f $O/v3_stack.log \| grep --line-buffered -E "DANCE\|REFLEX:\|DEGRADED"` |
| 5 | your commands | `python3 $M ...` |

- **Terminal 1:** `health` must say `OK`. The model runs on 4 CPU threads, leaving the GPU
  to the policies and the detector.
- **Terminal 2:** the node's stats line should show about 12.5 Hz.

**Terminal 3, the stack. Always first with the output off.** The motors receive nothing;
the policies, the conductor and the reflex run as they would:
```bash
PYTHONUNBUFFERED=1 ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0 \
    reflex_enable:=true dance_enable:=true \
    dance_engine:=$D/policy_fp32.plan dance_reference:=$D/dance_ref.bin \
    reflex_log:=$O/v3_reflex_log.csv 2>&1 | tee $O/v3_stack.log
```
Before going on, check:
- the status line says `RUNNING | obs 50.0 Hz ... cmd 500.0 Hz`, with no `DEGRADED`;
- the line `dance engine ... fnv1a=0x...` matches the fingerprint written down at install;
- the line `cmd_vel envelope (as trained): vx [-0.60, 1.00]  vy [-0.50, 0.50]  wz [-1.57, 1.57]`
  shows v2's range.

Then Ctrl-C, and start the same command with `enable_output:=true` (log `$O/v3m_stack.log`;
use the same name in terminal 4). Watch the robot stand on v2 for 10 s before giving any
command.

Notes on the launch arguments:
- **Write numeric arguments with a decimal point** (`1.3`, `1.0`). An integer such as
  `kp_scale:=1` stops the bridge at start-up.
- `iface` is the robot's interface on its `192.168.123.x` network: `eth10` on the lab robot.
  Check yours with `ip -br addr`.
- `kp_scale` multiplies the joint stiffness, for both policies. The real robot walked
  stably with v1 from 1.10 to 1.50. Use **1.3**, as every V3 run did.
- `reflex_enable` and `dance_enable` default to `false`. Without them the stack is plain
  v2: no reflex, no dance, and `ask "dance"` is refused. `dance_enable` needs
  `reflex_enable`.

If the status line says `DEGRADED`:
1. Save the reason: `grep "DEGRADED:" $O/v3_stack.log`.
2. With the output **off**: Ctrl-C and start the stack again. Do not use the `ros2` CLI to
   resume.
3. Still DEGRADED after a restart: stop, and bring the logs back.
4. With the motors **on**: emergency stop, see [Safety](#safety).

## Use

### Commands

Scripts (`run`) need no model; `ask` takes English. Both print the plan and wait for `y`.
Add `--log $O/ask_log.jsonl` to keep a record of each request and its plan.

| what | `run` script | `ask` example |
|---|---|---|
| walk forward / backward / sideways, by time or distance | `walk 3s@0.2`, `walk backward 1m@0.2`, `walk left 3s@0.15` | `"Walk backward 1 meter at 0.2 meters per second."` |
| turn on the spot, closed on the IMU heading | `turn left 90`, `turn right 180@1.0` | `"Turn left 90 degrees."` |
| stand (steps in place) | `stand 5s` | `"Stand for 5 seconds."` |
| **dance**, on the spot (5–60 s; 20 s if not said) | `dance 30s` | `"Dance for 30 seconds."` |
| several steps in one plan | `stand 3s; walk 1m@0.3; turn right 90` | `"Stand for 3 seconds, then walk forward 1 meter at 0.3 meters per second, then turn right 90 degrees."` |
| what the robot can do, from the installed configuration | `python3 $M capability` | |

`turn_to` and `walk_to` ("face the chair", "walk to the chair and stop 1.5 m from it") are
in the command layer, but they **have not been run on the robot yet**. Leave them for
later sessions with the output off.

What `ask` does, in order:
1. The model, on the CPU, transcribes the sentence into steps (about 1–3 s).
2. Code checks every step against what the policies were trained to do.
3. The plan is printed with its speeds, times and angles.
4. **You press `y` or `N`.** Read the plan before you press `y`.
5. The robot executes it.
6. A short report says what was done, for example `1. Danced on the spot for 30.4 s.`, or
   why it stopped: `Interrupted: stepped back because of a person 0.82 m away -- standing
   still until a new instruction`.

| it can | it refuses, and says why |
|---|---|
| walk forward up to 1.0 m/s, backward up to 0.6 m/s, sideways up to 0.5 m/s (defaults 0.4, 0.3, 0.2) | walk faster than that; walk diagonally in one step |
| turn on the spot up to 1.57 rad/s, to a given angle | turn while walking |
| dance 5 to 60 s, when nobody is within 1 m | dance while doing anything else ("dance while walking forward"); dance with the reflex off |
| stand still | a step under 2 s; more than 20 steps, 120 s or 30 m in one plan |

- **Every `ask` stands alone.** "Again", "back" or "the other way" are refused. Say the
  direction and the number, with its unit, every time.
- `--dry-run` prints the plan's command trace without moving anything.

### What the dance and the reflex do

**A dance starts** only when all of these hold:
- the reflex is armed;
- a fresh camera frame shows nobody within 1.0 m;
- the robot has stood still on v2 for 1 s.

**A dance ends** at the next bar boundary (a bar is 2–3 s) when:
- its time is up;
- the operator presses Ctrl-C;
- the `ask` process stops sending its heartbeat (0.5 s);
- 63 s have passed;
- the camera's frames go stale.

**The reflex**, every control step (50 Hz):

| state | meaning |
|---|---|
| ARMED | normal. The command layer's velocity reaches the policy, except that forward walking is blocked inside the stop distance (at least 1.2 m from a person, more at speed) or when the camera's frames are older than 0.3 s. Backward, sideways and turning pass |
| RETREAT | a person within 0.6 m (0.9 m while dancing) on 2 frames in a row: v2 drives (−0.3, 0, 0) m/s for 1.5 s. While dancing, v2 takes over on the same control step |
| STAND | after a retreat: v2 steps in place. It is held until you confirm a new plan **and** the person is further than the trigger. A person who stays close gets a robot that stays standing. At most 2 retreats in 10 s, then it stays latched until a new plan |

Neither the language model nor the command layer is in this loop. The bridge decides on its
own, and the plan in progress ends with the reason.

## First runs: acceptance checks

Run these in order, in the gantry, the first time and after any change. **Part A with the
output off, part B with the motors on, and part B only if part A passed.** Read terminal 4
and the reply of each command.

### Part A: output off (about 40 minutes)

| # | do | pass |
|---|---|---|
| A1 | nobody within 2 m: `python3 $M ask "dance for 10 seconds"`, `y` | terminal 4: `walk -> dance, DANCING`, then `ENDING`, then `dance -> walk, OFF (danced; back to standing)`. The reply: `1. Danced on the spot for 1x.x s.`, `Done.` |
| A2 | 3 times, 15 s apart: `ask "dance for 30 seconds"`. The helper starts 2.5 m in front, and once the dance runs walks slowly **straight towards the robot** to arm's length | on the same line: `DANCE: policy dance -> walk ... stepped back` and `REFLEX: stepping back: person too close -- person 0.xx m`. The reply: `Interrupted: stepped back because of a person ...` |
| A3 | the helper stays at arm's length: `ask "dance"` | refused: `1. Did not dance. Interrupted: the bridge did not dance: ...too close` |
| A4 | `ask "dance for 30 seconds"`, then Ctrl-C a few seconds in | `ENDING`, then `dance -> walk` within 3 s; the reply says `Interrupted: operator interrupt (Ctrl-C)` |
| A5 | `run "walk 3s@0.2"`, `run "walk backward 2s@0.2"`, `run "turn left 90"` | each `Done.` |

For each A2, write down the distance in the `REFLEX` line. **Pass:**
- A1, A3, A4 and A5 right;
- A2 three times handed over on the same line, with the right reply;
- no Traceback anywhere.

### Part B: motors on, in the gantry (about 60 minutes)

In the gantry the feet carry less load, so the dance looks different from on the floor.
This part judges **safety only**: no fall, no flailing legs at the handover, no rope
suddenly taking the weight.

| # | do | pass |
|---|---|---|
| B1 | `ask "dance for 30 seconds"`, 3 times, 15 s standing in between | each back to standing after 30–33 s; no slipping feet, no buckling legs, no sudden load on the rope |
| B2 | a scripted person at 0.5 m in the middle of a dance, 10 times (below) | each time v2 takes over at that moment, steps back about 1.5 s and stands; 0 anomalies |
| B3 | the helper walks up from straight in front, 5 times, as in A2 | as B2; the reply says `stepped back because of a person` |
| B4 | Ctrl-C during a dance, 2 times | `ENDING`, and standing within 3 s, at the end of a bar |

**B2 without a person.**
1. Stop perception first (Ctrl-C in terminal 2): two publishers of the proximity topic
   would fight.
2. Each time, in one terminal start the scripted camera:
   ```bash
   python3 $R/v3_tools/dance_devbox_camera.py --seconds 45 --timeline "0:none,12:0.5,14:none"
   ```
   and **within 5 s**, in another (`run` skips the model and `--yes` the confirmation):
   ```bash
   python3 $M run "dance 30s" --yes
   ```
3. At second 12 the camera prints `person 0.50 m`. At that moment terminal 4 shows the
   handover and the `REFLEX` line, and the robot steps back.
4. Wait for the 45 s to end before the next run.

Put the person at 12, 15, 18, 21, 24, 13, 16, 19, 22 and 25 s, so that it lands on
different parts of the dance. When the scripted camera stops, the bridge has no frames and
holds the robot standing; this is expected. After the 10 runs, start perception again.

## Person distance calibration

A person's distance is `person_box_k / (box height / image height)`.
`person_box_k: 0.465` in `perception/config/perception.yaml` was calibrated on the lab
robot in the gantry (camera about 1.1 m above the floor). Calibrate again if the camera
sits differently. It needs no tape: a helper faces the robot and stretches an arm out
until the fingertips just touch its chest (about 0.65 m), then takes one normal step back
(about 1.3 m). With the stack and perception running:

```bash
python3 $R/perception/tools/person_box_cal.py --seconds 5 --distance 0.65 --csv $O/v3_cal.csv
python3 $R/perception/tools/person_box_cal.py --seconds 5 --distance 1.3  --csv $O/v3_cal.csv
python3 $R/perception/tools/person_box_cal.py --from-csv $O/v3_cal.csv --write
```
`--write` keeps the old file as `perception.yaml.before_cal`. Restart perception afterwards.
`python3 $R/perception/tools/prox_watch.py` shows for a few seconds what the reflex would
act on.

## Safety

- **The emergency stop is in the operator's hand** whenever the motors are on.
- **Gantry only.** V3 has not been run untethered. Do not take it off the gantry before
  the untethered checks have been written and passed.
- **The robot backs off into space it cannot see.** Keep 1 m clear behind it.
- **People only.** The reflex does not see objects (the camera's depth is unreliable up
  close). An operator watches for obstacles.
- **Stop at once** on any odd noise, shaking, flailing legs, a rope that suddenly takes the
  weight, or `DEGRADED` with the motors on. E-stop, write down the time and what you saw,
  do not retry, and bring the logs back.
- Video that will be published must show no faces and have its audio removed.

## Stopping

| what | effect |
|---|---|
| **Ctrl-C** in the `ask` / `run` terminal | walking: zero velocity at once, and a report of how far the step got. Dancing: `ENDING`, back to v2 at the end of the bar (≤ 3 s) |
| the `ask` process dies | no heartbeat for 0.5 s: a dance ends at the bar; a walk's command decays to zero within 500 ms |
| a person too close | RETREAT, then STAND (above) |
| the bridge detects a fault (stale sensor data, a policy stops answering, a non-finite target, the control rate under 45 Hz for 3 s) | **DEGRADED**: the motors go to damping |
| anything unexpected | **the emergency stop** |

To finish a session:
1. Make sure the robot is hanging and still.
2. Ctrl-C the stack (terminal 3), then perception (terminal 2).
3. `bash $R/llm/ollama_ctl.sh stop`.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `install.sh` step 1: "this terminal has sourced a tree's env.sh" | open a new terminal; source nothing before `install.sh` |
| step 2 fails with `unitree/...: No such file` | unitree_sdk2 is not installed under `/usr/local` |
| `PARITY FAILED` (step 2 or 4) | do not run that engine. Check JetPack 5.1.1 / TensorRT 8.5.2, and bring the log back |
| step 4: `joint_map.hpp ... differs` | `base/` was changed by hand or with another bundle: install the v2 bundle into `base/` again (step 2) |
| step 4 stops at the CycloneDDS check | the bridge links the wrong DDS library: unitree_sdk2's CycloneDDS must be in `/usr/local/lib` |
| perception gets 0 frames, then the bridge goes `DEGRADED` | the app settings (they reset at every boot), or `stereo_capture.py` still holds ports 5002/5003 |
| forward walks stop at once, or never start | the reflex has no fresh frames: start perception (terminal 2), or start the stack without `reflex_enable` / `dance_enable` for plain v2 walking |
| `ask "dance"` refused: "no status from the bridge on /r1_hw_bridge/reflex" | the stack was started without `reflex_enable:=true` |
| a dance never starts | someone or something within 1 m in the camera's view, or the robot did not stand still for 1 s |
| `DEGRADED: joint_target stale` right after start-up | seen once with the output off; a restart cleared it. Restart, and bring the log back if it recurs |
| the bridge dies at start: `InvalidParameterTypeException` | a numeric launch argument written as an integer: use `1.3`, `1.0` |
| commands have no effect, or `ask` waits for the bridge | `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` differ between terminals, or `env.sh` was not sourced |
| `ask` refuses: "another publisher on ~/cmd_vel" | another `ask` / `run` is still running; only one may command the robot |
| the model server will not start: port in use | another tree's server is running: stop it with that tree's `ollama_ctl.sh stop` |
| `ollama_ctl.sh health` says the model did not answer | `bash $R/llm/ollama_ctl.sh restart` (the runtime occasionally starts without serving) |
| sway, tremor, grinding joints | developer mode is off: Unitree's controller is also commanding the motors |
| `ros2 topic list` crashes (`bad_alloc`) | a known clash between ROS foxy's and the Unitree SDK's DDS libraries on this robot. Nothing here needs the `ros2` CLI except `ros2 launch` |

## What to bring back after a session

```bash
cd ~ && tar czf v3_logs_$(date +%m%d).tar.gz orin_commissioning/v3*
```
It holds:
- the install log;
- the stack logs;
- the reflex log: one line per control step, with state, cause, commands and distances;
- the perception log.

Add a note for each check: the result, the trigger distances, anything unexpected and
roughly when.

## Roll back, remove

- **Back to v2:** in each terminal `source ~/r1-robot-v3.0.0/base/deploy/env.sh` instead,
  and start the stack without the `reflex_*` / `dance_*` arguments. `base/` is an installed
  v2 tree; `install.sh` changes it only in steps 2 and 3 (the v2 install, the vision
  engine), and step 4 checks that it stays unchanged.
- **Back to v1:**
  `bash ~/r1-robot-v3.0.0/base/policy_pack/install_bundle.sh ~/r1-robot-v3.0.0/base/bundles/2026-08-19_11-03-32_week04_nohead`
  (in `base/`, with its `env.sh` sourced). Install the v2 bundle again before reinstalling V3.
- **Remove:** `bash ~/r1-robot-v3.0.0/v3/llm/ollama_ctl.sh stop`, then
  `rm -rf -- ~/r1-robot-v3.0.0`. Nothing was installed outside that directory except logs in
  `~/orin_commissioning/`.

## Licenses

The code is Apache-2.0 (`LICENSE`, `NOTICE`). The Ollama runtime, the CUDA libraries it
bundles, the Qwen3 model and YOLOX-Nano keep their own licenses; see
`THIRD_PARTY_NOTICES.md`.
