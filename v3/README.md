# V3: proximity reflex (in progress)

V3 backs the robot off about 0.33 m when a person comes within 0.6 m or an obstacle
within 0.45 m, then holds it stepping in place until the operator confirms a new plan.
The walking policy is still v2 ([models/v2_drift_ft](../models/v2_drift_ft/README.md));
V3 is a change to the system around it.

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

## Isolation

All V3 code lives under `v3/`, and nothing outside this directory changes, so v1 and v2
(release v2.0.0) stay reproducible as they are. The three packages V3 changes are
forks, copied from commit `28fb7c1` (release v2.0.0):

| Fork | Upstream |
|---|---|
| `v3/perception/` | `perception/` |
| `v3/mission_ctl/` | `mission_ctl/` |
| `v3/ros2_ws/src/r1_hw_bridge/` | `deploy/ros2_ws/src/r1_hw_bridge/` (same package name; built in its own workspace) |

Everything else is shared and used as it is: the policy runner, `llm/`, `policy_pack/`,
the v2 artifacts. A fix made upstream after `28fb7c1` reaches V3 only when it is copied
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

V3 features in the forks:

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
| [run_tests.sh](run_tests.sh) | Every V3 check that needs no GPU, ROS or robot (`PYTHON=... v3/run_tests.sh`) |

Status: in development; not on the robot.
