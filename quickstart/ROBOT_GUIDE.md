# v2 on the robot: from the quick-start archive to walking backward and sideways

For the operator of a **Unitree R1 EDU** (onboard Jetson Orin NX). About an hour the first
time, two people. 中文版: [ROBOT_GUIDE_zh.md](ROBOT_GUIDE_zh.md).

v2 walks forward (to 1.0 m/s), **backward** (to 0.6 m/s) and **sideways** (±0.5 m/s), turns
on the spot (to 1.57 rad/s), and steps in place on a zero command without creeping. The
v1 policy is in the archive for rollback (§12).

> **Safety first.** This is research code for a robot that can fall and hurt people.
> Keep the emergency stop in your hand whenever the motors are on. Hang the robot in a
> gantry, with a little slack in the rope, for every first run. **The camera only looks
> forward and nothing watches behind or beside the robot:** clear at least 2 m behind and
> to the sides before a backward or sideways walk. Read §6 before anything moves.

## What you need

| | |
|---|---|
| robot | Unitree R1 **EDU**, with the onboard Orin NX 16 GB |
| robot system | JetPack 5.1.1 (TensorRT 8.5.2, CUDA 11.4, Ubuntu 20.04, Python 3.8), ROS 2 foxy in `/opt/ros/foxy`, **unitree_sdk2** installed under `/usr/local` (the bridge is compiled against it; it is not in the archive) |
| disk / memory | about 5 GB free disk; about 2.5 GB free memory while the language model is loaded |
| a PC | to copy files to the robot over SSH; the robot itself needs no internet |
| people and gear | an operator with the e-stop, a second person; a gantry |

**Upgrading from v1:** v2 goes into its own directory, `~/ldrq-v2`. A v1 tree
(`~/ldrq-v1` or `~/r1-robot-v1.1.0`) is left as it is. Stop the v1 stack and its model
server first (`bash <v1 tree>/llm/ollama_ctl.sh stop`): the new server refuses to start
while the port is taken.

## 1. On the PC: copy the archive to the robot

From a clone of this branch (`git clone -b v2 https://github.com/NAPPING-AI/LDRQ.git`):

```bash
cd LDRQ/quickstart
sha256sum -c ldrq-v2-quickstart.tar.gz.sha256        # OK
scp ldrq-v2-quickstart.tar.gz unitree@<robot-ip>:~/
```
Copy it as `.tar.gz`. A zip loses the files' execute permissions.

## 2. On the robot: extract and self-test

```bash
cd ~ && tar xzf ldrq-v2-quickstart.tar.gz             # creates ~/ldrq-v2/
bash ~/ldrq-v2/selftest.sh
```
Every line must be `ok`, except two `warn` lines about the language model until step 3 is
done, and `warn` lines about the system if this is not the robot. `selftest.sh` moves
nothing and needs no stack. It checks:
- every file against `FILES.sha256`;
- the command layer's and the English front end's tests;
- the eval sets against the shipped configuration;
- both policy bundles, and that bad bundles are refused;
- the robot system: JetPack 5, ROS 2 foxy, unitree_sdk2, TensorRT 8.

## 3. The language model (needed for `ask`, not for `run`)

The model server (Ollama for JetPack 5, about 0.5 GB) and the model (qwen3:1.7b, about
1.3 GB) are too large for git. v1 and v2 use the same ones. Take one of these:

**A. The robot already has a release tree** (`~/r1-robot-v2.0.0` or `~/r1-robot-v1.1.0`):
link its runtime and model; they are only read.
```bash
R=~/r1-robot-v2.0.0                       # or ~/r1-robot-v1.1.0
ln -s $R/llm/ollama ~/ldrq-v2/llm/ollama
ln -s $R/llm/models ~/ldrq-v2/llm/models
```

**B. Otherwise**, download from the v2.0.0 release
(<https://github.com/kedianwen/LDRQ/releases/tag/v2.0.0>) on the PC:
`release_v2.0.0.tar.gz`, `qwen3-1.7b_r1-robot-v2.0.0.tar.gz` and `SHA256SUMS`. Check
them with `sha256sum -c SHA256SUMS`, copy both to the robot, then:
```bash
mkdir -p ~/llm_tmp && cd ~/llm_tmp
tar xzf ~/release_v2.0.0.tar.gz r1-robot-v2.0.0/llm/ollama
tar xzf ~/qwen3-1.7b_r1-robot-v2.0.0.tar.gz
mv r1-robot-v2.0.0/llm/ollama r1-robot-v2.0.0/llm/models ~/ldrq-v2/llm/
```

**C. No model.** Skip this step and step 5. `run` takes the same plans as scripts
(`run "walk backward 1m@0.2; walk left 3s@0.15"`); `ask` will not work.

Run `bash ~/ldrq-v2/selftest.sh` again: the two `warn` lines about the model become `ok`.

## 4. Build and install the v2 policy (about 2 minutes)

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/ldrq-v2/deploy/env.sh
bash ~/ldrq-v2/policy_pack/install_bundle.sh ~/ldrq-v2/bundles/2026-10-16_17-57-14_v2_drift_ft
```
`install_bundle.sh` runs eight checked steps, and any failure stops it:
1. Validate the bundle.
2. Show what will change: for v2, the three `cmd_vel_*_range` lines of `bridge.yaml`.
3. Stage the files.
4. Generate the joint map and the configs (the joint map is v1's: same gains).
5. `colcon build` the bridge and the policy node (about 30 s).
6. Build the TensorRT engine **on this robot** (about 25 s).
7. **Check the engine against the bundle's reference** (the parity gate). Expect `max_abs`
   around 1e-05 against a 1e-3 limit.
8. Write `deploy/artifacts/installed.json`.

The fingerprint printed at the end changes from build to build; parity is what proves the
numbers.

## 5. Start the language model (skip with option C)

```bash
bash ~/ldrq-v2/llm/ollama_ctl.sh start
bash ~/ldrq-v2/llm/ollama_ctl.sh warm
# expect: "health: OK -- qwen3:1.7b answered in X s, and the plan is right"
bash ~/ldrq-v2/llm/ollama_ctl.sh status     # the model is loaded; it runs on the CPU
```
`mission_ctl/config/mission.yaml` places the model on 4 CPU threads (`llm_num_gpu: 0`,
`llm_num_thread: 4`), so the GPU stays with the walking policy. The server listens on
`127.0.0.1:11434` only and runs as your user, with no sudo and no systemd. Leave
`ollama_ctl.sh watchdog` off while the robot moves.

## 6. Every power-on, before the robot moves

1. **Put the robot in developer mode** with its handheld remote. Otherwise Unitree's own
   motion controller keeps commanding the motors at the same time as this stack. It looks
   like a bad policy: sway, tremor, grinding joints.
2. **Lock the clocks** (sudo, lasts until reboot):
   `bash ~/ldrq-v2/deploy/tools/w08_preflight.sh --lock-clocks`. It must report 0 FAIL.
3. Hang the robot in the gantry with a little slack in the rope. Clear 2 m behind it and to
   both sides.

In **every** terminal, first:
```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1      # before env.sh
source ~/ldrq-v2/deploy/env.sh
M=~/ldrq-v2/mission_ctl/r1_mission_cli.py; O=~/orin_commissioning; mkdir -p $O
```
Terminals with different `ROS_DOMAIN_ID` settings cannot see each other: the stack seems to
ignore every command.

## 7. Start the stack (terminal 1)

**First with the output off.** The motors receive nothing; everything else runs:
```bash
ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0 2>&1 | tee $O/v2_stack_off.log
```
Wait for `RUNNING | obs 50.0 Hz ... cmd 500.0 Hz` (about 5 s), with no `DEGRADED`. The
start-up lines must show v2's range:
`cmd_vel envelope (as trained): vx [-0.60, 1.00]  vy [-0.50, 0.50]  wz [-1.57, 1.57]`.
Then Ctrl-C and start it again with `enable_output:=true` (log `$O/v2_stack_on.log`).

- **Write numeric arguments with a decimal point** (`1.3`, `1.0`, `55.0`). An integer such
  as `kp_scale:=1` stops the bridge at start-up.
- `iface` is the interface on the robot's `192.168.123.x` network: `eth10` on the lab
  robot. Check with `ip -br addr`.
- `kp_scale`: v2 was trained at v1's gains and run at **1.3**, as v1. Its own stable range
  has not been swept.

## 8. Command it (terminal 2)

Scripts, no model needed:
```bash
python3 $M capability                                  # what the robot can do, from the installed bundle
python3 $M run "walk backward 1m@0.2; walk left 3s@0.15"
```
English (`--log` keeps a record of each request and its plan):
```bash
python3 $M ask "Walk forward for 3 seconds at 0.2 meters per second." --log $O/ask_log.jsonl
python3 $M ask "Walk backward 1 meter at 0.2 meters per second." --log $O/ask_log.jsonl
python3 $M ask "Sidestep to the left for 3 seconds." --log $O/ask_log.jsonl
python3 $M ask "Turn left 90 degrees." --log $O/ask_log.jsonl
```
What `ask` does, in order:
1. The model, on the CPU, transcribes the sentence into steps (about 1–3 s).
2. Code checks every step against the policy's training range.
3. The plan is printed with its speeds, times and angles.
4. **You press `y` or `N`.** Read the plan before you press `y`.
5. The robot executes it.
6. A short report says what was done.

| it can | it refuses, and says why |
|---|---|
| walk forward up to 1.0 m/s, backward up to 0.6 m/s, sideways up to 0.5 m/s (defaults 0.4, 0.3 and 0.2) | walk faster than that; walk diagonally in one step |
| turn on the spot, up to 1.57 rad/s, to a given angle, closed on the IMU heading | turn while walking |
| stand still (it steps in place) | a step under 2 s; more than 20 steps, 120 s or 30 m in one plan |
| walk a distance approximately (time × commanded speed, no odometry) | measure how far it actually went |

A walk's direction is `forward` (the default), `backward`, `left` or `right`; its speed is
always positive. Every `ask` stands alone: say the direction and the number, with its
unit, every time.

**First-run checks** (in the gantry, output on):

| check | expected |
|---|---|
| `run "stand 10s"` | steps in place; no visible creep forward |
| `run "walk 3s@0.2"` | walks forward, stops, keeps balancing |
| `run "walk backward 2s@0.2"` | steps backward (the gantry lets it move about 0.3 m) |
| `run "walk left 3s@0.15"`, then `run "walk right 3s@0.15"` | steps sideways, legs do not cross |
| `run "turn left 90"` | turns. The rope twists, so judge the angle only untethered |
| Ctrl-C during `run "walk 10s@0.2"` | zero velocity at once; the report gives the time walked |
| `ask "Walk backward 1 meter at 0.2 meters per second."` | the printed plan says backward, 0.20 m/s, 5 s, before you press `y` |

## 9. Stopping

| what | effect |
|---|---|
| **Ctrl-C** in the `ask` / `run` terminal | zero velocity at once, and a report of how far the step got; the robot keeps balancing |
| the `ask` process dies | no command for 500 ms → the bridge commands zero velocity |
| the bridge sees a fault (stale sensor data, the policy stops answering, a non-finite target, control rate under 45 Hz for 3 s) | **DEGRADED**: the motors go to damping. Ctrl-C the stack and start it again |
| anything unexpected | **the emergency stop** |

Finish: Ctrl-C the stack with the robot hanging and still, then
`bash ~/ldrq-v2/llm/ollama_ctl.sh stop`.

## 10. Vision (optional; stage E, not yet run with real frames)

The vision node and the steps `turn_to` / `walk_to` are in the archive with `perception:
off`. On the robot, the engine, its parity and the coexistence with the policy have
passed; `turn_to` and `walk_to` have not run there yet. Do this part **with the output
off** until they have.

1. **Camera stream, after every boot.** In the Unitree app, turn **video_hub off** and
   **Stereo patch PC1 on**. For depth (`walk_to` needs it): `sudo unitree_depth_start`.
2. **Probe** (moves nothing): `cd ~/ldrq-v2 && python3 perception/r1_perception_cli.py probe`.
3. **Engine** (about 7 minutes to build; FP16 is the robot's engine):
   ```bash
   B=$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner
   PM=~/ldrq-v2/perception/models
   ( cd $PM && sha256sum -c SHA256SUMS && sha256sum -c yolox_fixture.sha256 )
   $B/r1_build_engine --onnx $PM/yolox_nano.onnx --plan $PM/yolox_fp16.plan --fp16
   python3 ~/ldrq-v2/perception/r1_perception_cli.py engine-check --engine $PM/yolox_fp16.plan --gate detections
   ```
   It must end in `PARITY PASS`.
4. **Run the node** (terminal 3, with the stack up):
   `cd ~/ldrq-v2 && python3 perception/r1_perception_cli.py run --engine $PM/yolox_fp16.plan 2>&1 | tee $O/v2_perception.log`.
   Its stats line should show about 12.5 Hz.
5. **Turn the steps on:** in `~/ldrq-v2/mission_ctl/config/mission.yaml`, set
   `perception: on`. Then, output off: `python3 $M run "turn_to the chair"`, and
   `python3 $M run "walk_to the chair stop_at 1.5"` with a chair 3 m ahead. The protocol and
   what to record are in the repository's [docs/stageE_perception.md](../docs/stageE_perception.md).

## 11. Troubleshooting

| symptom | cause and fix |
|---|---|
| `install_bundle.sh` fails at step 5 with `unitree/...: No such file` | unitree_sdk2 is not installed under `/usr/local` |
| `PARITY FAILED` at step 7 | do not run that engine. Check JetPack 5.1.1 / TensorRT 8.5.2 |
| the bridge dies at start: `InvalidParameterTypeException` | a numeric launch argument written as an integer; use `1.3`, `55.0` |
| the bridge prints v1's range (`vx [0.00, 1.00]  vy [0.00, 0.00]  wz [-0.50, 0.50]`) | the v2 bundle is not installed in this tree: step 4 |
| `ask` refuses backward or sideways walks | the same: the command layer reads the installed bundle's range |
| commands have no effect, or `ask` waits for the bridge | `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` differ between terminals, or `env.sh` was not sourced |
| `ask` refuses: "another publisher on ~/cmd_vel" | another `ask` / `run` is still running; only one may command the robot |
| the model server will not start: port in use | another tree's server is running: stop it with that tree's `ollama_ctl.sh stop` |
| `ollama_ctl.sh warm`: the model did not answer | `bash ~/ldrq-v2/llm/ollama_ctl.sh restart` (the runtime occasionally starts without serving) |
| sway, tremor, grinding joints | developer mode is off: Unitree's controller is commanding the motors too |
| the vision node gets 0 frames | the app settings were lost at reboot (video_hub off, Stereo patch PC1 on), or another program holds port 5002 |
| `ros2 topic list` crashes (`bad_alloc`) | a known clash between ROS foxy's and the Unitree SDK's DDS libraries on this robot. Nothing here needs the `ros2` CLI except `ros2 launch` |

## 12. Roll back to v1

```bash
bash ~/ldrq-v2/policy_pack/install_bundle.sh ~/ldrq-v2/bundles/2026-08-19_11-03-32_week04_nohead
```
The bridge's command range returns to v1's (forward only, turns to 0.5 rad/s), and the
command layer refuses backward and sideways walks again. Install the v2 bundle again to
return.

## 13. What to bring back

```bash
cd ~ && tar czf v2_logs_$(date +%m%d).tar.gz orin_commissioning/
```
And a note of each check: what was asked, what the robot did, anything unexpected and
when. Video that will be published must show no faces and have its audio removed.

## Remove

```bash
bash ~/ldrq-v2/llm/ollama_ctl.sh stop
rm -rf -- ~/ldrq-v2
```
Nothing was installed outside that directory. With option A, the links point into the
release tree, which is left as it was.
