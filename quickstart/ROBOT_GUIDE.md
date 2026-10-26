# V3 on the robot: from the quick-start archive to a dance and the person reflex

For the operator of a **Unitree R1 EDU** (onboard Jetson Orin NX). The install takes about
an hour; the checks with the output off about 40 minutes; the checks in the gantry with the
motors on about 60 minutes. Two people: an operator with the e-stop, and a helper who walks
up to the robot. 中文版: [ROBOT_GUIDE_zh.md](ROBOT_GUIDE_zh.md).

V3 is v2 plus a dance policy and a person reflex:
- `ask "dance for 20 seconds"` dances on the spot (only if nobody is within 1 m);
- a person within **0.9 m** of the dancing robot (0.6 m when it walks or stands) makes v2
  take over on the same control step, step back for 1.5 s (about 0.33 m) and stand.

This guide follows the procedure that ran on the robot on 2026-10-24, from a fresh start.

> **Safety first.** This is research code for a robot that can fall and hurt people.
> - Keep the emergency stop in your hand whenever the motors are on.
> - Every run in this guide is **in the gantry**: untethered runs have not been done with V3.
> - The robot steps back **into space the camera cannot see**: clear 1 m behind it.
> - The reflex acts on **people only**, not on objects; an operator watches for obstacles.
> - On any odd noise, shaking, flailing legs, a rope that suddenly takes the weight, or
>   `DEGRADED` with the motors on: **e-stop at once**, write down the time and what you
>   saw, and do not retry. Stop the session there.

## What you need

| | |
|---|---|
| robot | Unitree R1 **EDU**, onboard Orin NX 16 GB, head stereo module |
| robot system | JetPack 5.1.1 (TensorRT 8.5.2, CUDA 11.4, Ubuntu 20.04, Python 3.8 with numpy), ROS 2 foxy in `/opt/ros/foxy`, **unitree_sdk2** under `/usr/local` (not in the archive) |
| the Unitree app | on a phone, to switch the camera stream on after every boot |
| disk / memory | about 6 GB free disk; about 2.5 GB free memory while the language model is loaded |
| a PC | to copy files to the robot over SSH; the robot itself needs no internet |
| people and gear | operator with the e-stop, a helper; a gantry; 1 m clear behind the robot, 2.5 m clear in front |

## Part A: install (once)

### A1. On the PC: copy the archive to the robot

From a clone of this branch (`git clone -b v3 https://github.com/NAPPING-AI/LDRQ.git`):

```bash
cd LDRQ/quickstart
sha256sum -c ldrq-v3-quickstart.tar.gz.sha256        # OK
scp ldrq-v3-quickstart.tar.gz unitree@<robot-ip>:~/
```
Copy it as `.tar.gz`. A zip loses the files' execute permissions.

### A2. On the robot: extract and self-test the base

Open a **new terminal that has not sourced any `env.sh`**, with no stack running:
```bash
cd ~ && tar xzf ldrq-v3-quickstart.tar.gz      # creates ~/ldrq-v3/{base,v3_package}
cat ~/ldrq-v3/VERSION ~/ldrq-v3/v3_package/VERSION
bash ~/ldrq-v3/base/selftest.sh
```
- `base/` is the v2 robot tree that V3 is installed from.
- `v3_package/` holds V3 and its installer.
- Every self-test line must be `ok`, except two `warn` lines about the language model
  until A3 is done.

### A3. The language model (needed for `ask`, not for `run`)

The model server (Ollama for JetPack 5) and the model (qwen3:1.7b) are about 1.8 GB, too
large for git. Put them into `base/` **before A6**: the V3 tree links them from there.

**A. The robot already has a release tree** (`~/r1-robot-v2.0.0` or `~/r1-robot-v1.1.0`):
```bash
R0=~/r1-robot-v2.0.0                      # or ~/r1-robot-v1.1.0
ln -s $R0/llm/ollama ~/ldrq-v3/base/llm/ollama
ln -s $R0/llm/models ~/ldrq-v3/base/llm/models
```
**B. Otherwise**, on the PC, download `release_v2.0.0.tar.gz`,
`qwen3-1.7b_r1-robot-v2.0.0.tar.gz` and `SHA256SUMS` from
<https://github.com/kedianwen/LDRQ/releases/tag/v2.0.0>. Check them with
`sha256sum -c SHA256SUMS` and copy both to the robot. Then:
```bash
mkdir -p ~/llm_tmp && cd ~/llm_tmp
tar xzf ~/release_v2.0.0.tar.gz r1-robot-v2.0.0/llm/ollama
tar xzf ~/qwen3-1.7b_r1-robot-v2.0.0.tar.gz
mv r1-robot-v2.0.0/llm/ollama r1-robot-v2.0.0/llm/models ~/ldrq-v3/base/llm/
```
**C. No model.** `run` takes the same plans as scripts (`run "dance 20s"`); `ask` does not
work.

Stop any other model server first; they all use port 11434.

### A4. Install the v2 policy into the base (about 2 minutes)

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/ldrq-v3/base/deploy/env.sh
bash ~/ldrq-v3/base/policy_pack/install_bundle.sh ~/ldrq-v3/base/bundles/2026-10-16_17-57-14_v2_drift_ft
```
These are eight checked steps; any failure stops it. Step 7 is the parity gate: `max_abs`
around 1e-05 against a 1e-3 limit. Step 4 writes v2's command range into the base's
`bridge.yaml`.

### A5. Build the vision engine in the base (about 7 minutes)

Same terminal:
```bash
B=$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner
PM=~/ldrq-v3/base/perception/models
( cd $PM && sha256sum -c SHA256SUMS && sha256sum -c yolox_fixture.sha256 )
$B/r1_build_engine --onnx $PM/yolox_nano.onnx --plan $PM/yolox_fp16.plan --fp16
python3 ~/ldrq-v3/base/perception/r1_perception_cli.py engine-check --engine $PM/yolox_fp16.plan --gate detections
```
It must end in `PARITY PASS`. FP16 is the robot's engine: on TensorRT 8.5, FP32 misses
the raw gate by a hair, with identical detections.

### A6. Install V3 (about 5 minutes)

**Open a new terminal** (the installer refuses a terminal that sourced an `env.sh`):
```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
bash ~/ldrq-v3/v3_package/install_v3.sh --src ~/ldrq-v3/base --dst ~/r1-robot-v3
```
It prints `[n/8]` for each step and writes its log under `~/orin_commissioning/`.

| step | expect |
|---|---|
| 1 preflight | disk space; package files match `MANIFEST.sha256` |
| 2 compare | informational: `bridge.yaml` differs from v2.0.0 (v2's ranges, written in A4); anything else listed was changed by hand in the base |
| 3 copy | the base copied; `llm/ollama`, `llm/models` linked (if A3 was done) |
| 4 overlay | `joint_map.hpp identical (v2's gains)`; `v3_models/: dance_wave` |
| 5 configs | only the expected differences: `reflex: on`, `dance*`, `obj_stop_min_m 0.9`; `proximity_*`, `person_box_*`; the `reflex_*` block. **No other `<` line** |
| 6 build | the bridge and the runner build |
| 7 checks | `CycloneDDS from /usr/local/lib: OK`; every V3 test line `ok` |
| 8 engines | the v2 engine and the dance engine, each `PARITY PASS`; **write down the dance engine's `fnv1a=`** |
| end | `source tree unchanged: OK`; `V3 installed: .../r1-robot-v3` |

If it stops at any step, bring the whole output back and change nothing. `~/ldrq-v3/base` is
only read: going back to v2 is sourcing its `env.sh`.

## Part B: every session

### B1. Before anything moves

1. **Developer mode** on the handheld remote. Otherwise Unitree's controller writes the
   motors too, and it looks like a bad policy.
2. **Lock the clocks** once per boot (sudo):
   `bash ~/r1-robot-v3/deploy/tools/w08_preflight.sh --lock-clocks` → 0 FAIL.
3. **The camera stream, after every boot.** In the Unitree app: **video_hub off**,
   **Stereo patch PC1 on**.
4. The robot in the gantry, feet on the floor, rope slack. The camera ends up about
   1.1–1.2 m above the floor.

In **every** terminal, first:
```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/r1-robot-v3/deploy/env.sh
R=~/r1-robot-v3; M=$R/mission_ctl/r1_mission_cli.py; PC=$R/perception/r1_perception_cli.py
PM=$R/perception/models; O=~/orin_commissioning; D=$R/v3_models/dance_wave; mkdir -p $O
```

**Check the camera before starting perception.** On 10-24 a session lost its first hour to
this, with perception at 0 frames and the bridge `DEGRADED`:
```bash
pgrep -af stereo_capture                                   # must print nothing; else kill <PID>
python3 $R/perception/tools/stereo_capture.py --probe      # 3 s: ports 5002 and 5003 about 10 fps each
```
0 packets: set the app again (video_hub off, Stereo patch PC1 on) and probe again.

### B2. Start the services

Terminal 1, the model (skip with option C):
```bash
bash $R/llm/ollama_ctl.sh start && bash $R/llm/ollama_ctl.sh health
```
Terminal 2, perception:
```bash
cd $R && python3 $PC run --engine $PM/yolox_fp16.plan 2>&1 | tee $O/v3_perception.log
```
Terminal 3, the stack, **output off**:
```bash
PYTHONUNBUFFERED=1 ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0 \
    reflex_enable:=true dance_enable:=true \
    dance_engine:=$D/policy_fp32.plan dance_reference:=$D/dance_ref.bin \
    reflex_log:=$O/v3_reflex_log.csv 2>&1 | tee $O/v3_stack.log
```
- The status line must say `RUNNING`.
- The start-up line `dance engine ... fnv1a=...` must match the value written down in A6.
- Numeric arguments need a decimal point (`1.3`, not `1`).
- `iface` is the robot's `192.168.123.x` interface: `eth10` on the lab robot.

If it says `DEGRADED`, first save the reason:
`grep "DEGRADED:" $O/v3_stack.log`. Then Ctrl-C terminal 3 and start the stack again (with
the output off this is safe; do not use the `ros2` CLI to resume). Still DEGRADED after a
restart: stop, and bring the logs back.

Terminal 4, the lines that matter:
```bash
tail -f $O/v3_stack.log | grep --line-buffered -E "DANCE|REFLEX:|DEGRADED"
```

### B3. Person distance calibration (optional)

`person_box_k: 0.465` in `perception/config/perception.yaml` was calibrated on the lab
robot in the gantry on 2026-10-24. Calibrate again if your camera height differs. It needs
no tape:
1. The helper faces the robot and stretches an arm out until the fingertips just touch
   its chest: about 0.65 m.
2. Then the helper takes one normal step back: about 1.3 m.

```bash
python3 $R/perception/tools/person_box_cal.py --seconds 5 --distance 0.65 --csv $O/v3_cal.csv
python3 $R/perception/tools/person_box_cal.py --seconds 5 --distance 1.3  --csv $O/v3_cal.csv
python3 $R/perception/tools/person_box_cal.py --from-csv $O/v3_cal.csv --write
```
`--write` keeps the old file as `perception.yaml.before_cal`. Restart perception
(terminal 2) afterwards.

## Part C: checks with the output off (about 40 minutes)

The motors receive nothing, but the policies, the conductor and the reflex run as they
would. Every check reads terminal 4 and the reply of `ask`.

| # | do | pass |
|---|---|---|
| ① | nobody within 2 m: `python3 $M ask "dance for 10 seconds"`, press `y` | terminal 4: `walk -> dance, DANCING`, then `ENDING`, then `dance -> walk, OFF (danced; back to standing)`. The reply is `1. Danced on the spot for 1x.x s.` and `Done.` No Traceback |
| ② | 3 times, 15 s apart: `ask "dance for 30 seconds"`. The helper starts about 2.5 m in front, and once the dance runs walks slowly **straight towards the robot** to arm's length | on the same line, `DANCE: policy dance -> walk ... stepped back` and `REFLEX: stepping back: person too close -- person 0.xx m`. The reply: `Interrupted: stepped back because of a person ...` |
| ②b | the helper stays at arm's length: `ask "dance"` | refused: `1. Did not dance. Interrupted: the bridge did not dance: ...too close` |
| ③ | `ask "dance for 30 seconds"`; a few seconds in, Ctrl-C in the `ask` terminal | `ENDING`, then `dance -> walk` within 3 s; the reply says `Interrupted: operator interrupt (Ctrl-C)` |
| ④ | (record only) the helper stands still at 1.0–1.1 m: `ask "dance for 10 seconds"`, 5 times | write down each result: refused / danced and stepped back within 2 s / danced to the end |

Write down for each ② the distance in the `REFLEX` line, and whether the helper came from
straight in front. **Pass** (needed before part D): ① right; ② three times handed over on
the same line with the right reply; ③ right; no Traceback anywhere.

## Part D: in the gantry, motors on (about 60 minutes)

Only after part C passed. In the gantry the feet carry less load, so the dance looks
different from on the floor. **This part checks only safety:** no falls, no flailing legs
at the handover, no rope suddenly taking the weight. It does not judge the dance.

Terminal 3: Ctrl-C the output-off stack, then start it with `enable_output:=true` (log
names `v3m_...`). Wait for `RUNNING`. Watch the robot stand on v2 for 10 s, and go on only if
it is calm.

| # | do | pass |
|---|---|---|
| ① | `ask "dance for 30 seconds"`, 3 times, 15 s standing in between | each back to standing after 30–33 s; no slipping feet, no buckling legs, no sudden load on the rope |
| ② | a scripted person at 0.5 m in the middle of a dance, 10 times (below) | each time, v2 takes over at that moment, steps back for about 1.5 s and stands; 0 anomalies |
| ③ | the helper walks up from straight in front, 5 times, as in part C ② | as ②; the reply says `stepped back because of a person` |
| ④ | Ctrl-C during a dance, 2 times | `ENDING`, and standing within 3 s at the end of a bar |

**② without a person.** Stop perception first (Ctrl-C in terminal 2): two publishers of the
proximity topic would fight. Then, each time, in terminal 5 start the scripted camera:
```bash
python3 $R/v3_tools/dance_devbox_camera.py --seconds 45 --timeline "0:none,12:0.5,14:none"
```
and **within 5 s**, in terminal 6 (`run` skips the model and the `y`):
```bash
python3 $M run "dance 30s" --yes
```
At second 12 the camera prints `person 0.50 m`. Terminal 4 must show the handover and the
`REFLEX` line at that moment, and the robot steps back about 1.5 s, then stands. Wait for the
scripted camera to end (45 s) before the next one. Use the person times 12, 15, 18, 21, 24,
13, 16, 19, 22 and 25 s, so that they land on different parts of the dance (for example
`"0:none,15:0.5,17:none"`). After the scripted camera stops, the bridge has no frames and
holds the robot standing; this is expected. After the 10 runs, start perception again in
terminal 2.

## Stopping and what each state means

| what | effect |
|---|---|
| **Ctrl-C** in the `ask` / `run` terminal | walking: zero velocity at once. Dancing: `ENDING`, back to v2 at the end of the bar (≤ 3 s) |
| the `ask` process dies | the dance heartbeat lapses (0.5 s): the dance ends at the bar; a walk's command decays to zero in 500 ms |
| a person within the trigger | RETREAT: v2 steps back 1.5 s at −0.3 m/s, then STAND until a new plan is confirmed and the way is clear |
| no camera frames (older than 0.3 s) | forward walking is blocked; a dance does not start, and a running one ends at the bar |
| the bridge sees a fault | **DEGRADED**: motors to damping. With the motors on: e-stop, write it down, stop the session |
| anything unexpected | **the emergency stop** |

Finish: the robot hanging and still, Ctrl-C the stack (terminal 3), then perception
(terminal 2), then `bash $R/llm/ollama_ctl.sh stop`.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `install_v3.sh` stops at step 1: "this terminal has sourced a tree's env.sh" | open a new terminal; do not source anything before the installer |
| step 1: "has no perception/" | `--src` must be `~/ldrq-v3/base` |
| step 4: `joint_map.hpp ... differs` | the base does not have v2's install: A4 |
| step 5 shows other `<` lines | a local value in the base's config: write it into the new file in `~/r1-robot-v3` |
| step 8: `PARITY FAILED` | do not run that engine; bring the log back |
| perception 0 frames, then the bridge `DEGRADED` | the app settings (they reset at every boot), or `stereo_capture.py` still holds ports 5002/5003 |
| `ask "dance"` refused: "no status from the bridge on /r1_hw_bridge/reflex" | the stack was started without `reflex_enable:=true` (dancing needs the reflex) |
| a dance never starts | someone or something within 1 m in the camera's view, or the robot did not stand still for 1 s |
| `DEGRADED: joint_target stale` right after start-up (seen once, output off) | restart the stack; it did not recur. Bring the log back if it does |
| the bridge dies at start: `InvalidParameterTypeException` | a numeric launch argument written as an integer |
| commands have no effect | `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` differ between terminals, or `env.sh` not sourced |
| `ros2 topic list` crashes (`bad_alloc`) | the known DDS clash on this robot; nothing here needs the `ros2` CLI except `ros2 launch` |

## Back to v2, and removing V3

- **Back to v2:** `source ~/ldrq-v3/base/deploy/env.sh` instead of V3's. Nothing in the
  base was changed. Start the stack there without the `reflex_*` / `dance_*` arguments.
- **Remove V3:** `rm -rf -- ~/r1-robot-v3`. The base stays usable. To remove everything,
  stop the model server, then `rm -rf -- ~/ldrq-v3 ~/r1-robot-v3`. If option A was used,
  the release tree it links to is left as it was.

## What to bring back

```bash
cd ~ && tar czf v3_logs_$(date +%m%d).tar.gz orin_commissioning/v3*
```
And a note for each check: the result, the trigger distances, anything unexpected and
roughly when. Video that will be published must show no faces and have its audio removed;
stereo recordings with people in them stay private.
