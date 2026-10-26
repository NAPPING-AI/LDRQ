# v1 on the robot: from the quick-start archive to an English command

For the operator of a **Unitree R1 EDU** (onboard Jetson Orin NX). About an hour the first
time, two people. 中文版: [ROBOT_GUIDE_zh.md](ROBOT_GUIDE_zh.md).

v1 walks forward (up to 1.0 m/s) and turns on the spot (up to 0.5 rad/s). It does not walk
backward or sideways; that is v2 (branch `v2`).

> **Safety first.** This is research code for a robot that can fall and hurt people.
> Keep the emergency stop in your hand whenever the motors are on. Hang the robot in a
> gantry, with a little slack in the rope, for every first run. Read §6 before anything
> moves.

## What you need

| | |
|---|---|
| robot | Unitree R1 **EDU**, with the onboard Orin NX 16 GB |
| robot system | JetPack 5.1.1 (TensorRT 8.5.2, CUDA 11.4, Ubuntu 20.04, Python 3.8), ROS 2 foxy in `/opt/ros/foxy`, **unitree_sdk2** installed under `/usr/local` (the bridge is compiled against it; it is not in the archive) |
| disk / memory | about 5 GB free disk; about 2.5 GB free memory while the language model is loaded |
| a PC | to copy files to the robot over SSH; the robot itself needs no internet |
| people and gear | an operator with the e-stop, a second person; a gantry |

## 1. On the PC: copy the archive to the robot

From a clone of this branch (`git clone -b main https://github.com/NAPPING-AI/LDRQ.git`):

```bash
cd LDRQ/quickstart
sha256sum -c ldrq-v1-quickstart.tar.gz.sha256        # OK
scp ldrq-v1-quickstart.tar.gz unitree@<robot-ip>:~/
```
Copy it as `.tar.gz`. A zip loses the files' execute permissions.

## 2. On the robot: extract and self-test

```bash
cd ~ && tar xzf ldrq-v1-quickstart.tar.gz             # creates ~/ldrq-v1/
bash ~/ldrq-v1/selftest.sh
```
Every line must be `ok`, except two `warn` lines about the language model until step 3 is
done, and `warn` lines about the system if this is not the robot. `selftest.sh` moves
nothing and needs no stack. It checks:
- every file against `FILES.sha256`;
- the command layer's and the English front end's tests;
- the eval sets against the shipped configuration;
- the policy bundle, and that bad bundles are refused;
- the robot system: JetPack 5, ROS 2 foxy, unitree_sdk2, TensorRT 8.

## 3. The language model (needed for `ask`, not for `run`)

The model server (Ollama for JetPack 5, about 0.5 GB) and the model (qwen3:1.7b, about
1.3 GB) are too large for git. Take one of these:

**A. The robot already has release v1.1.0** (`~/r1-robot-v1.1.0`): link its runtime and
model; they are only read.
```bash
ln -s ~/r1-robot-v1.1.0/llm/ollama ~/ldrq-v1/llm/ollama
ln -s ~/r1-robot-v1.1.0/llm/models ~/ldrq-v1/llm/models
```

**B. Otherwise**, download from the v1.1.0 release
(<https://github.com/kedianwen/LDRQ/releases/tag/v1.1.0>) on the PC:
`release_v1.1.0.tar.gz`, `qwen3-1.7b_r1-robot-v1.1.0.tar.gz` and `SHA256SUMS`. Check
them with `sha256sum -c SHA256SUMS`, copy both to the robot, then:
```bash
mkdir -p ~/llm_tmp && cd ~/llm_tmp
tar xzf ~/release_v1.1.0.tar.gz r1-robot-v1.1.0/llm/ollama
tar xzf ~/qwen3-1.7b_r1-robot-v1.1.0.tar.gz
mv r1-robot-v1.1.0/llm/ollama r1-robot-v1.1.0/llm/models ~/ldrq-v1/llm/
```

**C. No model.** Skip this step and step 5. `run` takes the same plans as scripts
(`run "walk 3s@0.2; turn left 90"`); `ask` will not work.

Run `bash ~/ldrq-v1/selftest.sh` again: the two `warn` lines about the model become `ok`.

## 4. Build and install the policy (about 2 minutes)

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/ldrq-v1/deploy/env.sh
bash ~/ldrq-v1/policy_pack/install_bundle.sh ~/ldrq-v1/bundles/2026-08-19_11-03-32_week04_nohead
```
`install_bundle.sh` runs eight checked steps, and any failure stops it:
1. Validate the bundle.
2. Show what will change.
3. Stage the files.
4. Generate the joint map and the configs.
5. `colcon build` the bridge and the policy node (about 30 s).
6. Build the TensorRT engine **on this robot** (about 25 s).
7. **Check the engine against the bundle's reference** (the parity gate). Expect `max_abs`
   around 1e-05 against a 1e-3 limit; the lab robot measured 1.144e-05.
8. Write `deploy/artifacts/installed.json`.

The last lines print the engine's path and fingerprint. The fingerprint changes from build
to build (TensorRT picks kernels by timing them); parity is what proves the numbers.

## 5. Start the language model (skip with option C)

```bash
bash ~/ldrq-v1/llm/ollama_ctl.sh start
bash ~/ldrq-v1/llm/ollama_ctl.sh warm
# expect: "health: OK -- qwen3:1.7b answered in X s, and the plan is right"
bash ~/ldrq-v1/llm/ollama_ctl.sh status     # the model shows "on GPU" (~1.5 GB)
```
In v1 the model runs on the Orin's GPU (`llm_num_gpu: -1` in `mission_ctl/config/mission.yaml`).
The server listens on `127.0.0.1:11434` only and runs as your user, with no sudo and no
systemd. If another server holds the port (an older tree's), stop that one first, for
example `bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh stop`. Leave `ollama_ctl.sh watchdog`
off while the robot moves.

## 6. Every power-on, before the robot moves

1. **Put the robot in developer mode** with its handheld remote. Otherwise Unitree's own
   motion controller keeps commanding the motors at the same time as this stack. It looks
   like a bad policy: sway, tremor, grinding joints.
2. **Lock the clocks** (sudo, lasts until reboot):
   `bash ~/ldrq-v1/deploy/tools/w08_preflight.sh --lock-clocks`. It must report 0 FAIL.
3. Hang the robot in the gantry with a little slack in the rope; clear 1 m around it.

In **every** terminal, first:
```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1      # before env.sh
source ~/ldrq-v1/deploy/env.sh
M=~/ldrq-v1/mission_ctl/r1_mission_cli.py; O=~/orin_commissioning; mkdir -p $O
```
Terminals with different `ROS_DOMAIN_ID` settings cannot see each other: the stack seems to
ignore every command.

## 7. Start the stack (terminal 1)

**First with the output off.** The motors receive nothing; everything else runs:
```bash
ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0 2>&1 | tee $O/v1_stack_off.log
```
Wait for `RUNNING | obs 50.0 Hz ... cmd 500.0 Hz` (about 5 s), with no `DEGRADED`. Then
Ctrl-C and start it again with `enable_output:=true` (log `$O/v1_stack_on.log`). The robot
takes the policy's standing pose and steps in place.

- **Write numeric arguments with a decimal point** (`1.3`, `1.0`, `55.0`). An integer such
  as `kp_scale:=1` stops the bridge at start-up.
- `iface` is the interface on the robot's `192.168.123.x` network: `eth10` on the lab
  robot. Check with `ip -br addr`.
- `kp_scale` multiplies the joint stiffness. The real robot walked stably from 1.10 to
  1.50; the trained value, 1.0, sits on the edge of that, so use **1.3**.

## 8. Command it (terminal 2)

Scripts, no model needed:
```bash
python3 $M capability                         # what the robot can do, from the installed bundle
python3 $M run "walk 3s@0.2; turn left 90"     # shows the plan, asks y/N
```
English (`--log` keeps a record of each request and its plan):
```bash
python3 $M ask "Walk forward for 3 seconds at 0.2 meters per second." --log $O/ask_log.jsonl
python3 $M ask "Turn left 90 degrees." --log $O/ask_log.jsonl
python3 $M ask "Stand for 3 seconds, then walk forward 1 meter at 0.3 meters per second, then turn right 90 degrees." --log $O/ask_log.jsonl
```
What `ask` does, in order:
1. The model transcribes the sentence into steps (about 1–4 s).
2. Code checks every step against the policy's training range.
3. The plan is printed with its speeds, times and angles.
4. **You press `y` or `N`.** Read the plan before you press `y`.
5. The robot executes it.
6. A short report says what was done.

| it can | it refuses, and says why |
|---|---|
| walk forward at 0 to 1.0 m/s (default 0.4) | walk backward; step sideways |
| turn on the spot, up to 0.5 rad/s, to a given angle, closed on the IMU heading | turn while walking |
| stand still | a step under 2 s; more than 20 steps, 120 s or 30 m in one plan |
| walk a distance approximately (time × commanded speed, no odometry) | measure how far it actually went |

- **Every `ask` stands alone.** "Again", "back" or "the other way" are refused: always
  say the direction and the number, with its unit.
- `--dry-run` simulates the plan and prints the command trace without moving anything.

**First-run checks** (in the gantry, output on):

| check | expected |
|---|---|
| `run "stand 5s"` | steps in place, then the report |
| `run "walk 3s@0.2"` | walks, then stops and keeps balancing |
| `run "turn left 90"` | turns. In the gantry the rope twists, so judge the angle only untethered |
| Ctrl-C during `run "walk 10s@0.2"` | zero velocity at once; the report gives the time walked |
| `ask "Walk forward for 3 seconds at 0.2 meters per second."` | the printed plan says 3 s at 0.20 m/s before you press `y` |

## 9. Stopping

| what | effect |
|---|---|
| **Ctrl-C** in the `ask` / `run` terminal | zero velocity at once, and a report of how far the step got; the robot keeps balancing |
| the `ask` process dies | no command for 500 ms → the bridge commands zero velocity |
| the bridge sees a fault (stale sensor data, the policy stops answering, a non-finite target, control rate under 45 Hz for 3 s) | **DEGRADED**: the motors go to damping. Ctrl-C the stack and start it again |
| anything unexpected | **the emergency stop** |

Finish: Ctrl-C the stack with the robot hanging and still, then
`bash ~/ldrq-v1/llm/ollama_ctl.sh stop`.

## 10. Troubleshooting

| symptom | cause and fix |
|---|---|
| `install_bundle.sh` fails at step 5 with `unitree/...: No such file` | unitree_sdk2 is not installed under `/usr/local` |
| `PARITY FAILED` at step 7 | do not run that engine. Check JetPack 5.1.1 / TensorRT 8.5.2 |
| the bridge dies at start: `InvalidParameterTypeException` | a numeric launch argument written as an integer; use `1.3`, `55.0` |
| commands have no effect, or `ask` waits for the bridge | `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` differ between terminals, or `env.sh` was not sourced |
| `ask` refuses: "another publisher on ~/cmd_vel" | another `ask` / `run` is still running; only one may command the robot |
| `ollama_ctl.sh warm`: the model did not answer | `bash ~/ldrq-v1/llm/ollama_ctl.sh restart` (the runtime occasionally starts without serving) |
| sway, tremor, grinding joints | developer mode is off: Unitree's controller is commanding the motors too |
| `ros2 topic list` crashes (`bad_alloc`) | a known clash between ROS foxy's and the Unitree SDK's DDS libraries on this robot. Nothing here needs the `ros2` CLI except `ros2 launch` |

## 11. What to bring back

```bash
cd ~ && tar czf v1_logs_$(date +%m%d).tar.gz orin_commissioning/
```
And a note of each check: what was asked, what the robot did, anything unexpected and
when. Video that will be published must show no faces and have its audio removed.

## Remove

```bash
bash ~/ldrq-v1/llm/ollama_ctl.sh stop
rm -rf -- ~/ldrq-v1
```
Nothing was installed outside that directory. With option A, the links point into
`~/r1-robot-v1.1.0`, which is left as it was.
