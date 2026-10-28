# V3: a dance, and a proximity reflex

V3 backs the robot off about 0.33 m when a person comes within 0.6 m or an obstacle
within 0.45 m, then holds it stepping in place until the operator confirms a new plan.
The walking policy is still v2 ([models/v2_drift_ft](../models/v2_drift_ft/README.md)).

Since 2026-10-23 the robot also dances, on request ("dance for 20 seconds"): a second
policy, trained to track a short routine on the spot, runs next to v2. When a person
comes within 0.9 m of the dancing robot, v2 takes over on that same control step,
steps back and stands. See [The dance](#the-dance-two-policies).

## Status after the first robot session (2026-10-23)

**The head camera's depth is not good enough close up.** The robot's own head shell hides
part of the right stereo camera's view. The stereo module fills the pixels it cannot match
with near values. As a result, every frame of an empty room had "obstacles" 0.05–0.35 m
ahead, and a person about 2 m away read 0.6 m. The depth is also hole-filled and smoothed
(a flat floor does not come out flat), and it is not aligned with the left image. So:

- Perception runs with `proximity_mode: person_box`: people only, and their distance
  comes from the height of their detection box (`distance = person_box_k / box height
  fraction`), with no depth. A box touching both the top and the bottom of the image is
  someone right in front. `perception/tools/person_box_cal.py` sets `person_box_k` with
  no tape, from a helper at arm's length (about 0.65 m) and one step further (about 1.3 m).
- mission_ctl has `reflex_obstacles: off`. Its capability statement says the reflex acts
  on people and not yet on objects.
- The obstacle path (`proximity_mode: depth`) stays. It needs a depth source that can be
  trusted: calibration and unfilled depth from the module's maker, or depth computed from
  the left and right images.

Replaying the robot's own frames through the perception node and the bridge on a dev box:
depth mode retreats on a phantom obstacle at 0.20 m, as the robot did; person_box mode
does not retreat.

## Robot sessions (2026-10-24)

Installed fresh from `~/r1-robot-v1.1.0` with [tools/install_v3.sh](tools/install_v3.sh)
(the source tree's fingerprint unchanged). Both engines built on the Orin and passed parity
against their fixtures: v2 max_abs 2.3e-05, the dance policy 4.6e-05. `person_box_k`
calibrated to 0.465 without tape.

**Output off, robot hanging.** The two engines take at most 2.7 ms per control step together
at p99 (budget 4 ms). A 10 s dance ended on a bar boundary 13.2 s after it started. A person
walking up to the dancing robot handed it to v2 on the same control step, which stepped back
(triggered at 0.84 and 0.59 m). With someone 0.83 m away, a new dance was refused.

**Output on.** The bridge stayed RUNNING throughout (no DEGRADED, no CRC failures). The
robot danced and handed back to v2 at the end of a bar (10 s and 20 s requests), stopped
at the next bar on Ctrl-C, and stepped back from the dance when a person came within
0.79–0.88 m. A dance asked for with someone too close was refused.

Fixed after these sessions:

- mission_ctl's reply had no sentence for a dance step. It crashed (`IndexError`) when a
  dance was interrupted or refused. The record now carries `danced_s`, and the reply says
  how long the robot danced, or that it did not.
- The dry run's simulated bridge stepped the conductor once per executor step. At the
  CLI's 10 Hz a bar lasted five times too long, so a dance never ended in time. It now
  steps the conductor at the bridge's 50 Hz in between.

Seen once with output off, not explained yet: `DEGRADED: joint_target stale` (the
runner's targets more than 60 ms late) shortly after the stack started, while the camera
stream was off. Restarting the stack cleared it, and it did not recur.

**Depth from the two images** (recordings taken apart on a dev box,
[perception/tools/stereo_eval.py](perception/tools/stereo_eval.py)). Not used by the robot
yet:
- The left and right streams share one RTP clock: the offset is the same in every
  recording, so frames pair by timestamp.
- The floor fit gives the camera's height consistently at 1.10 m. That was the height the
  robot hung at.
- A box at arm's length reads 0.63 m. A person one step further reads 1.15 m.
- Three things are still to fix: a person at arm's length reads low; there are mismatches
  on the carpet close to the robot; and finding the head shell by "pixels that never
  change" does not work, because room light changes the shell's brightness as the robot
  turns. The shell's edge is fixed in every frame and is the better cue.

## The head (2026-10-28)

The policy does not drive the two head joints; the bridge held them at the URDF default
(0, 0). On the robot that turned the head about 43 degrees to the right every time driving
started, together with the first steps in place. The head's encoder reading with the head
facing straight ahead is not 0, and it is not the same from one session to the next:
`head_yaw_joint` read +0.7519 rad, and later the same day +2.109 (the bridge's own log).
So no fixed offset in bridge.yaml is right.

The bridge now holds the head where it is when driving starts, and logs that pose
(`head held where it is: pitch ... yaw ... rad (encoder)`). The head's observation is its
offset from that pose (0 before driving and while degraded), so the policy sees it at 0,
as in training. Start with the head facing ahead.

`watch_obs.py` therefore shows the head joints near 0, not their encoder readings; the
bridge's log line has those. `watch_obs.py --check-default` shows the head joints as a
mismatch. That is expected.

## The reflex

The reflex runs in the bridge, once per control step, between mission_ctl's velocity
command and the policy's observation. Neither the language model nor mission_ctl is in
its loop. The reference implementation is [reflex/reflex.py](reflex/reflex.py).

| State | Command written into the observation |
|---|---|
| OFF | mission_ctl's command, untouched (reflex disabled, or the bridge not RUNNING) |
| ARMED | mission_ctl's command, with forward speed clamped to 0 inside the stop distance, or while the camera cannot vouch for the path (stale frames, no depth, a hole in the middle of the depth image) |
| RETREAT | (-0.3, 0, 0) m/s for 1.5 s |
| STAND | (0, 0, 0): the policy steps in place, as for the `stand` step. Held until mission_ctl sends a reset (the operator confirmed a new plan) and the way is clear |

- **Retreat triggers.** A person nearer than 0.6 m, or an obstacle nearer than 0.45 m,
  on two consecutive frames. A person box tall enough to be near counts as near
  without depth.
- **In STAND.** A second retreat needs a new approach: the hazard must first move
  beyond the trigger + 0.2 m. Someone who stays close gets a robot that stays standing.
- **At most two retreats in 10 s.** After that the robot stands, latched until a reset.
- **Only the forward part of a command is blocked.** Backward, sideways and turning
  pass.
- **The camera looks forward only.** The 0.33 m the robot backs off is into space it
  cannot see.

### Stop distance

Walking toward something must stop early enough that the robot comes to rest outside
the retreat trigger. The stop distance therefore grows with speed, with a margin on
every term ([reflex/stop_distance.py](reflex/stop_distance.py)):

    stop(v) = max(floor, trigger + 0.10 m + 1.5 * v * latency + 1.3 * brake(v))

- floor: the fixed distances used before V3 (person 1.2 m, obstacle 0.6 m);
- latency: 0.25 s until it is measured on the robot;
- brake(v): the worst braking distance in simulation (below);
- v: the largest forward command of the last second.

| Forward command | Worst braking | Person stop | Obstacle stop |
|---|---|---|---|
| 0 | 0.07 m | 1.20 m | 0.64 m |
| 0.3 m/s | 0.16 m | 1.20 m | 0.87 m |
| 0.5 m/s | 0.23 m | 1.20 m | 1.03 m |
| 1.0 m/s | 0.47 m | 1.69 m | 1.54 m |

### Simulation (2026-10-22)

[sim/eval_retreat.py](sim/eval_retreat.py) switches the v2 policy's command from a
starting motion to the retreat. The switch time is spread over the gait cycle. Runs used
the training task (its randomization on, pushes off), seeds 1-5, 256 envs each.

| Starting motion | Falls | Backed off from the trigger point, mean / least |
|---|---|---|
| standing | 0 | 0.33 / 0.24 m |
| turning on the spot (±0.5 rad/s), stepping sideways (±0.2 m/s) | 0 | 0.33 / 0.22 m |
| forward 0.3 m/s | 0 | 0.25 / 0.13 m |
| forward 0.5 m/s | 0 | 0.21 / 0.09 m |
| forward 1.0 m/s | 0 | none: on average it ended 0.01 m beyond the trigger point, at worst 0.45 m |

- **Retreat length.** A 1.0 s retreat backed off only 0.19 m from standing, so the
  retreat is 1.5 s.
- **No ramp.** A 0.2 s ramp delayed the reversal from 0.09 s to 0.20 s and doubled the
  overshoot, with no gain in stability.
- **Gait phase.** Where in the gait the switch came changed the distance by under 1 cm.
- **Braking.** The braking distances in the stop-distance table come from the same runs
  with a zero retreat speed.
- **Forward walks.** These should not meet the reflex at speed: the stop distance comes
  first.

### In the bridge and in mission_ctl

- **Bridge** (`ros2_ws/src/r1_hw_bridge`): the state machine is
  `include/r1_hw_bridge/reflex_fsm.hpp`, with no ROS types. It is a line-for-line port
  of `reflex/reflex.py`, and `test/test_reflex_fsm.cpp` replays every tick of
  `reflex/golden.csv` against it and compares exactly. With `reflex_enable: true`
  (default false, in `config/bridge.yaml` and as a launch argument of
  `r1_stack.launch.py`), the bridge:
  - reads `/r1_perception/proximity`;
  - replaces the observation's velocity command once per control step;
  - publishes `~/reflex` (JSON: state, reason, commands in and out, distances, frame age);
  - takes `~/reflex_reset`;
  - with `reflex_log`, writes one CSV line per step;
  - logs, every 5 s, how old the frames are when they arrive: the latency the stop
    distance has to assume.

  It refuses to start with the reflex on under a command range that has no backward
  walking (v1's).
- **mission_ctl** (`mission_ctl/r1_mission/reflex_link.py`, the executor, the node):
  - **The reflex takes over.** A retreat, or the robot held standing, ends the plan. The
    reply names the cause: "stepped back because of an obstacle 0.40 m ahead -- standing
    still until a new instruction".
  - **The bridge cuts a forward walk.** The plan ends after 0.3 s, unless a retreat
    follows. The walk's recorded time stops where the walk stopped.
  - **Reset before a new plan.** When the operator confirms a new plan, mission_ctl sends
    the reset. While the bridge refuses it (still too close), nothing starts and the
    operator is told why.
  - **The camera guard.** Where it applies (`perception_guard`), it uses the same
    speed-dependent stop distance, computed independently of the bridge.
  - **Dry runs.** A dry run with `--world` simulates the bridge's reflex with the
    reference. World objects can move (`vx`, `vy`, `move_t`, `stop_t`), as in
    `tests/worlds/approach.json`, where a person walks up to the robot.
  - **Configuration.** `reflex: on` in `mission.yaml`.
  - **The language model.** The prompt is unchanged, and so is its id. The capability
    statement, which the model does not see, gains a sentence on the reflex.

### On a dev box

The bridge builds without the Unitree SDK against a fake robot (a robot standing at
its default pose; `-DR1_FAKE_ROBOT_IO=ON`, never on the robot), so the real node runs
on ROS 2 topics:

    colcon build --cmake-args -DR1_FAKE_ROBOT_IO=ON      # a workspace with r1_hw_bridge
    r1_hw_bridge_node --ros-args --params-file v3/ros2_ws/src/r1_hw_bridge/config/bridge.yaml \
        -p reflex_enable:=true -r joint_target:=/r1_policy_node/joint_target \
        -r action:=/r1_policy_node/action

- [tools/reflex_devbox_e2e.py](tools/reflex_devbox_e2e.py) plays the policy, the
  camera and mission_ctl around it, and checks what the observation carries: walking,
  blocked, the retreat, standing, and walking again after a reset.
- [tools/reflex_devbox_world.py](tools/reflex_devbox_world.py) plays only the policy
  and the camera, so `mission_ctl/r1_mission_cli.py run ...` runs live against the
  bridge.
- `mission_ctl/tests/reflex_messages.jsonl` is a recording of the bridge's `~/reflex`
  messages, which mission_ctl's tests parse.

### On the robot

V3 installs into a new tree, `~/r1-robot-v3`, made from the one in use. The tree in
use is only read, so going back to v2 is sourcing its `env.sh`.

- `tools/make_robot_package.sh <n>` (dev box) packs `r1_v3_<n>_<date>.tar.gz`. It holds:
  - the V3 code laid out as the robot tree has it;
  - a hash of every file in the package;
  - the hashes of what release v2.0.0 has in the places V3 replaces.
- `install_v3.sh`, run from the unpacked package, does the rest:
  - copies the tree, linking the Ollama runtime and model store instead of copying them;
  - lays V3 over the copy, provided `joint_map.hpp` is identical, and shows the
    configuration diffs;
  - rebuilds the workspace;
  - checks that CycloneDDS is loaded from `/usr/local/lib`, and runs `v3_tests.sh` in the
    new tree;
  - gates the policy engine with `r1_parity_check`;
  - compares a fingerprint of the source tree before and after the install.

  `--rehearsal` runs the same steps on a dev box against the fake robot.

The camera's height and pitch place the floor that the walking corridor leaves out.
With the height known (1.2 m on the robot), `perception/tools/fit_floor.py` gets the
pitch and the focal length from the floor alone, with no tape and no calibration file. A
camera tilted a few degrees more than configured turns an empty floor into an obstacle
about a metre ahead.

## The dance (two policies)

**The routine** ([dance/routine.py](dance/routine.py)) is a 10 s loop of four bars, both
feet on the floor throughout: the right arm out and waving (3 s), the left (3 s), both
arms up with two knee dips (2 s), standing still (2 s). Every bar begins and ends in the
default standing pose at rest; a bar boundary is where the robot may be handed back to
v2. [dance/check_routine.py](dance/check_routine.py) checks it against the URDF before
any training: joint limits, self-collision, joint speed and acceleration, the soles not
moving, and that no point of the arms enters the image region the person-distance
estimate uses (an arm in front of the lens cuts a person's box short, and a short box
reads farther, the unsafe direction). The order of the arm's joints matters for that last
check: the arm swings out and straightens, turns about its own axis while straight, and
only then bends the forearm up.

**The policy** is trained with BeyondMimic's motion tracking
([third_party/whole_body_tracking](third_party/whole_body_tracking/UPSTREAM.md), MIT,
vendored unchanged; Isaac Lab 2.1.0, as this project uses). The R1 task is
[dance/r1_tracking.py](dance/r1_tracking.py): v2's robot, gains, action term and default
pose (the bridge sends one set of gains for both policies), the pelvis as the anchor
(the IMU is there), the observation without state estimation (137 floats, no history),
v2's randomization added to upstream's. Upstream randomizes the default pose by writing
the action offsets with articulation joint ids, which works only when the action drives
every joint in articulation order; R1's drives 24 of 26 in another, and the R1 task
replaces that term.

**The conductor** ([reflex/conductor.py](reflex/conductor.py), the reference;
[conductor.hpp](ros2_ws/src/r1_hw_bridge/include/r1_hw_bridge/conductor.hpp) in the
bridge, matched tick for tick on [reflex/golden_dance.csv](reflex/golden_dance.csv))
decides each control step which policy drives the robot:

- a dance starts on a request on `/r1_hw_bridge/dance` (`std_msgs/Bool`, true) when the
  reflex is armed, a fresh camera frame shows nobody and nothing within 1.0 m, v2 has
  stood still for 1 s, and the dance policy has fresh output;
- the request repeats as a heartbeat; false, a lost heartbeat (0.5 s), 63 s of dancing
  or stale camera frames end the dance at the next bar boundary;
- while dancing, the reflex triggers at 0.9 m (the arms reach about 0.5 m); a retreat, the
  bridge leaving RUNNING, or the dance policy's output going stale hands the robot to v2
  at once;
- v2 runs the whole time on (0, 0, 0), so its history is warm when it takes over; after a
  dance ends, another one needs a fresh request (the heartbeat must lapse first).

**The runner** ([ros2_ws/src/r1_policy_runner](ros2_ws/src/r1_policy_runner), a fork)
runs both engines on every frame when `dance_engine_path` is set. With dancing on, the
bridge appends the IMU quaternion, the dance tick and the active policy to the frame, and
the runner builds the dance policy's observation
([dance_obs.hpp](ros2_ws/src/r1_policy_runner/include/r1_policy_runner/dance_obs.hpp);
the reference is [dance/dance_obs.py](dance/dance_obs.py), which matches Isaac's own
observation of the same state to 1.5e-7). The observation's last-action slot carries the
action that was applied, from whichever policy that was.

**mission_ctl** has a `dance` step (5 to 60 s, 20 s if not said), refused without the
reflex. `stop dancing` takes the fast stop path, which never reaches the model. The
language model's prompt (`1b4e65a44dba`) adds the step and two examples. On the frozen
dance set ([eval/nl_test6.jsonl](mission_ctl/eval/nl_test6.jsonl), 30 cases, CPU) the
first run had one unsafe answer: "dance while walking forward 2 meters" written as a
dance and then a walk. A deterministic rule now refuses an instruction that asks for
two moving steps at once (while, at the same time, simultaneously, meanwhile); it was
then scored on a set frozen for it ([eval/nl_test7.jsonl](mission_ctl/eval/nl_test7.jsonl)).
The earlier sets score as they did with the previous prompt.

| set | correct | unsafe |
|---|---|---|
| nl_test6 (dance), first run | 28/30 | 1 |
| nl_test6, with the rule | 28/30 | 0 |
| nl_test7 (two actions at once) | 9/10 | 0 |

**Handover in simulation** ([sim/eval_handover.py](sim/eval_handover.py)) runs both
policies the way the runner does and hands over as the conductor does, at random points
of the dance, with v2 alone as the baseline.
The policy is [models/dance_wave](models/dance_wave/README.md). With motor strength at the
robot's kp_scale 1.3 and the dance task's other randomization, about 2000 runs each:

| scenario | falls | stepped back (mean, min) | drift over the 10 s stand (mean) |
|---|---|---|---|
| normal: start from v2's gait at a random time, end at a bar boundary | 0 / 2017 | - | - |
| emergency: v2 takes over at a random point of the dance, -0.3 m/s for 1.5 s | 0 / 2044 | 0.43 m, 0.34 m | 0.24 m |
| baseline: v2 alone, the same retreat from standing | 1 / 2048 | 0.39 m, 0.33 m | 0.24 m |

The drift is v2's own at this kp (the baseline drifts the same), so the handover is judged
against the baseline, not against a fixed number. About 1.5 % of envs go down in the first
second after the evaluation writes them into the standing pose, before any dance, v2 alone
included; they are counted apart. The dance policy's observation as the runner builds it
matches Isaac's own to 2.4e-7. At the trained kp (1.0), about 1000 runs each: one fall per
scenario, v2 alone included, all 1.5-1.6 s after the reset (just past the settling window,
which the softer gains settle more slowly); the emergency retreat was 0.36 m against v2's
0.33 m, and the drift the same as v2's.

## Isolation

All V3 code lives under `v3/`, and nothing outside this directory changes, so v1 and v2
(release v2.0.0) stay reproducible as they are. The three packages V3 changes are
forks, copied from commit `28fb7c1` (release v2.0.0):

| Fork | Upstream |
|---|---|
| `v3/perception/` | `perception/` |
| `v3/mission_ctl/` | `mission_ctl/` |
| `v3/ros2_ws/src/r1_hw_bridge/` | `deploy/ros2_ws/src/r1_hw_bridge/` (same package name; built in its own workspace) |
| `v3/ros2_ws/src/r1_policy_runner/` | `deploy/ros2_ws/src/r1_policy_runner/` (the dance engine; unchanged without it) |

`v3/third_party/whole_body_tracking/` is upstream's code, unchanged. Everything else is
shared and used as it is: `llm/`, `policy_pack/`, the v2 artifacts. A fix made upstream after `28fb7c1` reaches V3 only when it is copied
into the fork by hand.

Changes in the forks so far, each needed because V3 runs v2:

- `ros2_ws/src/r1_hw_bridge/config/bridge.yaml`: v2's command ranges, as the v2
  bundle's installer writes them.
- `mission_ctl/r1_mission/node.py`: finds that `bridge.yaml` and the shared `llm/`.
- `mission_ctl/eval/run_nl_eval.py`: the eval sets frozen against v1 (stages C and E)
  are judged against v1's envelope, so their scores stay comparable.
- `mission_ctl/tests/test_nl.py`: the same for the consistency test, and the CLI's
  refusal test uses an instruction v2 refuses (strafing is allowed under v2).
- `mission_ctl/config/mission.yaml`: `obj_stop_min_m` 0.9, above the obstacle stop
  distance at `walk_to`'s speed.

V3 features in the forks (the bridge and mission_ctl are described above):

- `perception/`: a second topic, `/r1_perception/proximity` (`std_msgs/Float64MultiArray`,
  layout `PROX_LAYOUT` in `r1_perception/pipeline.py`), one per frame and published
  before the JSON detections: the nearest person's range, a person box tall enough to
  be near, the 30th-nearest return in the walking corridor, how much of the depth
  image has returns (whole and center), and the capture time in `CLOCK_MONOTONIC`
  seconds. Float64 because a float32 of a day's uptime resolves only about 8 ms. The
  JSON message carries the same facts as `proximity`.

## Contents

| Path | What |
|---|---|
| [sim/eval_retreat.py](sim/eval_retreat.py) | How v2 answers the retreat command, from several starting motions and gait phases |
| [sim/summarize_retreat.py](sim/summarize_retreat.py) | Combines runs over seeds into one table: falls summed, worst case over all seeds |
| [reflex/stop_distance.py](reflex/stop_distance.py) | Speed-dependent stop distance, with margins on latency and braking |
| [reflex/reflex.py](reflex/reflex.py) | The reflex state machine (OFF / ARMED / RETREAT / STAND): the reference the bridge's C++ must match |
| [reflex/cases.py](reflex/cases.py) | Scripted cases with hand-written checkpoints; writes [reflex/golden.csv](reflex/golden.csv), every tick of every case |
| [reflex/test_reflex.py](reflex/test_reflex.py) | Stop distance, the cases, golden.csv still reproduced, perception's message read by the reflex |
| [ros2_ws/src/r1_hw_bridge/include/r1_hw_bridge/reflex_fsm.hpp](ros2_ws/src/r1_hw_bridge/include/r1_hw_bridge/reflex_fsm.hpp) | The bridge's state machine (C++17, no ROS) |
| [ros2_ws/src/r1_hw_bridge/test/test_reflex_fsm.cpp](ros2_ws/src/r1_hw_bridge/test/test_reflex_fsm.cpp) | The C++ against golden.csv, every tick, exactly |
| [ros2_ws/src/r1_hw_bridge/test/fake_robot_io.cpp](ros2_ws/src/r1_hw_bridge/test/fake_robot_io.cpp) | A robot standing still, for running the bridge on a dev box |
| [mission_ctl/r1_mission/reflex_link.py](mission_ctl/r1_mission/reflex_link.py) | mission_ctl's side: the bridge's messages, the reset, the dry run's simulated bridge |
| [mission_ctl/tests/test_reflex.py](mission_ctl/tests/test_reflex.py) | Executor, guard, reset and dry runs with the reflex |
| [tools/](tools/) | Dev-box drivers for the real bridge node (above); the robot package (below) |
| [perception/tools/fit_floor.py](perception/tools/fit_floor.py) | Camera pitch and focal length from the floor in captured depth frames, the height known (inverse depth is linear in the pixel on a plane). Needs pinhole depth of a flat floor; the head module's filled depth does not give one |
| [perception/tools/near_check.py](perception/tools/near_check.py) | What is within 0.35 m of the camera in captured frames, where in the image, and whether it stays there (the robot's own body, a stand, filled-in depth) |
| [perception/tools/person_box_cal.py](perception/tools/person_box_cal.py) | `person_box_k` from a helper at arm's length and one step further: no tape |
| [perception/tools/prox_watch.py](perception/tools/prox_watch.py) | Watch `/r1_perception/proximity` for a few seconds: what the reflex would act on; `--trace` writes every frame |
| [dance/routine.py](dance/routine.py), [dance/check_routine.py](dance/check_routine.py), [dance/urdf_fk.py](dance/urdf_fk.py) | The routine, its checks, and the URDF kinematics they use (numpy) |
| [dance/csv_to_npz_r1.py](dance/csv_to_npz_r1.py) | The routine into BeyondMimic's motion file, with a replay video (Isaac) |
| [dance/r1_tracking.py](dance/r1_tracking.py), [dance/train_dance.py](dance/train_dance.py), [dance/export_dance.py](dance/export_dance.py) | The dance task, its training entry point, and the export for the runner |
| [dance/dance_obs.py](dance/dance_obs.py) | The dance policy's observation as the runner builds it; the reference table's file format |
| [reflex/conductor.py](reflex/conductor.py), [reflex/dance_cases.py](reflex/dance_cases.py) | Which policy drives: the reference, its cases and golden file |
| [sim/eval_handover.py](sim/eval_handover.py) | Handovers between the two policies in simulation |
| [tools/dance_devbox_camera.py](tools/dance_devbox_camera.py) | Plays the camera (a person at scripted distances) for a dance run on a dev box |
| [tools/update_v3.sh](tools/update_v3.sh), [tools/make_update_package.sh](tools/make_update_package.sh) | Update an installed V3 tree in place (backup, overlay, build, engines with parity) |
| [perception/tools/stereo_capture.py](perception/tools/stereo_capture.py), [stereo_probe.py](perception/tools/stereo_probe.py), [stereo_unpack.py](perception/tools/stereo_unpack.py) | Record the head cameras' RTP packets on the robot, and take them apart on a dev box: the start of depth computed from the two images |
| [perception/r1_perception/stereo.py](perception/r1_perception/stereo.py), [perception/tools/stereo_eval.py](perception/tools/stereo_eval.py) | Depth from the two head images (rectify, SGBM, floor fit, corridor) and its checks on recordings |
| [tools/install_v3.sh](tools/install_v3.sh), [tools/make_robot_package.sh](tools/make_robot_package.sh) | A fresh V3 tree on the robot from the v2 one (read only), with the dance engine built and parity-checked there |
| [release/](release/README.md) | The V3 robot release: `make_release_v3.sh` (dev box), `install.sh` and `selftest.sh` (robot), its README (English, 中文), release notes and third-party notices |
| [run_tests.sh](run_tests.sh) | Every V3 check that needs no GPU, ROS or robot (`PYTHON=... v3/run_tests.sh`) |

Status: complete (2026-10-26). The dance and the person reflex ran on the robot, hanging, with output off and on (2026-10-24), and those runs closed the project's engineering. Not done: untethered runs, and obstacles (depth from the two images was started and not finished).
