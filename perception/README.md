# perception/ · object detection for the mission layer

`r1_perception` turns the R1's head camera into a stream of detections that the mission layer can act on:
- class, confidence and box;
- **bearing** (left positive, the same sign as the IMU yaw) and elevation;
- **median depth inside the box**, and the horizontal **range** derived from it;
- per-frame safety facts: a person too close, something in the walking corridor.

It runs YOLOX-Nano (COCO, Apache-2.0) through TensorRT on the robot's Orin NX at about 12.5 Hz. It is used only by `mission_ctl`'s `turn_to` and `walk_to`, and never enters the 50 Hz control loop.

Status and results: [docs/stageE_perception.md](../docs/stageE_perception.md).

> **The camera is the head's stereo module.** Unitree's R1 "video stream service" page
> describes it: the left camera's RGB (544×448) arrives as RTP/H264 on UDP port 5002,
> and depth (544×448, uint16 millimetres) in the v4l2loopback `/dev/video-dep`. Both
> need, after every boot, the APP's video_hub OFF and "Stereo patch PC1" ON, plus
> `sudo unitree_depth_start`. `source: stereo` reads them. The intrinsics in
> `config/perception.yaml` (marked PROBE) are placeholders until the module's
> `cam_param.txt` is read, and depth-to-RGB registration is assumed until checked on an
> overlay (`capture --vis`).

## Files

| file | role |
|---|---|
| `r1_perception_cli.py` | entry point: `probe`, `run`, `engine-check`, `compare`, `capture`, `look` |
| `config/perception.yaml` | source, lens, mount, thresholds, safety distances, scheduling |
| `r1_perception/yolox.py` | letterbox, grid decode, class-aware NMS (numpy) |
| `r1_perception/camera_model.py` | pinhole intrinsics (or fisheye ⇄ rectified pinhole), bearing / elevation / range in the body frame |
| `r1_perception/depth.py` | median depth in a box; the corridor check that ignores the floor |
| `r1_perception/tracker.py` | N consecutive frames per target (`confirm_frames`) |
| `r1_perception/pipeline.py` | one frame → one message; the operator's annotated picture |
| `r1_perception/runtime.py` | TensorRT (ctypes CUDA, TensorRT 8.5 and 10) and onnxruntime back ends |
| `r1_perception/sources.py` | the stereo module (left RGB over UDP + depth), ROS image topics, V4L2, or a replay folder |
| `r1_perception/node.py` | the ROS 2 node and its phase-locked trigger |
| `tools/make_fixture.py` | the parity fixture (R1FX, the walking policy's format), on the dev box |
| `tools/calibrate_fisheye.py` | checkerboard → `cam_*` lines for `perception.yaml` |
| `tools/eval_detect.py` | precision, person recall and threshold sweep on labeled robot images |
| `tools/eval_tag.py` | bearing and range error against AprilTag ground truth |
| `models/fetch_yolox.sh` | downloads the model and a sample image, pinned by `SHA256SUMS` |
| `tests/test_perception.py` | numpy-only tests (in `run_tests.sh`) |

## How to use

**Dev box: get the model and build a fixture.**

```bash
bash perception/models/fetch_yolox.sh               # yolox_nano.onnx + dog.jpg, sha256-checked
python3 perception/tools/make_fixture.py --onnx perception/models/yolox_nano.onnx \
    --images <robot frames>/*.jpg --out perception/models/yolox_fixture.bin
```

**Robot: probe, build, gate** (inside `deploy/env.sh`, like the walking policy).

```bash
python3 perception/r1_perception_cli.py probe       # camera, topics, Python pieces -> ~/orin_commissioning/camera_probe
B=$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner
$B/r1_build_engine --onnx perception/models/yolox_nano.onnx --plan perception/models/yolox_fp32.plan
$B/r1_build_engine --onnx perception/models/yolox_nano.onnx --plan perception/models/yolox_fp16.plan --fp16
$B/r1_parity_check --plan perception/models/yolox_fp32.plan --fixture perception/models/yolox_fixture.bin --tol 1e-3
python3 perception/r1_perception_cli.py engine-check --engine perception/models/yolox_fp32.plan
python3 perception/r1_perception_cli.py engine-check --engine perception/models/yolox_fp16.plan --gate detections
```

On the robot (10-14) FP16 passed its gate and FP32 missed the raw gate by 0.4e-3 with identical detections; FP16 is the robot's engine and what `run` picks when `--engine` is not given.

**Robot: run it, look at it.**

```bash
python3 perception/r1_perception_cli.py run          # with the stack up: phase-locked to the policy
python3 perception/r1_perception_cli.py look         # what it sees now
python3 perception/r1_perception_cli.py capture --vis          # frames for the evaluations
python3 perception/r1_perception_cli.py compare --frames ~/r1_captures/<dir>   # FP32 vs FP16
```

`run` takes `--engine` (a `.plan`, or an `.onnx` for onnxruntime) and `--set key=value` to override `perception.yaml`, for example `--set source=replay --set replay_dir=~/r1_captures/x --set rectify=0`.

## Design

- **Scheduling, not speed.** The walking policy publishes `/r1_policy_node/action` right after each TensorRT inference. The node runs on every fourth one, which gives 12.5 Hz, and YOLO then has about 19 ms of otherwise idle GPU before the next policy step. The CPU half (frame, rectify, letterbox) runs on the step before, so the GPU starts as soon as the slot opens; on the robot, doing it inside the slot let FP32 run into the policy's next step while the model was busy on the CPU. In stage C, a model sharing that GPU pushed the policy's inference from 0.45 ms to about 5 ms. This is the mechanism stage E relies on for p99 ≤ 2 ms. If no action arrives for a second (no stack, or the dev box), a timer takes over and the stats line says `trigger timer`.
- **The same parity gate as the policy.**
  - `r1_build_engine` and `r1_parity_check` already handle any single static-shape FP32 input and output, so YOLOX goes through them unchanged.
  - `engine-check` puts the Python runtime, which the node actually uses, through the same fixture.
  - FP32 is gated on raw outputs (max_abs ≤ 1e-3). FP16 is gated on detections: FP16 moves background anchors' raw values by about 1 without changing a single box, so a raw tolerance would only tell you that FP16 is not FP32.
- **Two thresholds.**
  - `safety_conf` (low, single frame): what is published, and what the person check uses. A missed person costs more than a needless stop.
  - The mission layer's `obj_target_conf` (higher, `confirm_frames` in a row): what `turn_to` and `walk_to` may act on.
  - `tools/eval_detect.py` calibrates the two separately.
- **The floor is never an obstacle.** The corridor check converts depth pixels to body-frame points and counts only those 0.10–1.9 m above the floor inside the robot's width. It ignores the floor at any camera pitch, without a tuned image region.
- **Unregistered depth is dropped, not used.** If the driver does not align depth to RGB (`depth_aligned: 0`), a box's median would come from whatever lies behind the object. The node reports no depth, and `walk_to` refuses.
- **No new packages on the robot.** It uses `std_msgs`/`sensor_msgs` and JSON on `std_msgs/String`; no `cv_bridge`, no `pycuda`. CUDA memory goes through `ctypes` on `libcudart`.

## Known pitfalls

| pitfall | what handles it |
|---|---|
| A 150° lens rectified to one pinhole image stretches its edges without bound | the rectified view covers `rect_hfov_deg` (100°); the outer band is given up |
| OpenCV's fisheye calibration fails to start on a wide lens (`InitExtrinsics: norm_u1 > 0`) | `calibrate_fisheye.py` starts from the nominal field of view (`--hfov-guess-deg`) and drops views it names as ill-conditioned |
| OpenCV 5 renamed the fisheye flags (`cv2.CALIB_*`); the robot's 4.5 has `cv2.fisheye.CALIB_*` | the tool looks up whichever exists |
| A small AprilTag is not ground truth: its range comes from its apparent size, and 1 px on a 12 px tag is 8 % | `eval_tag.py` skips tags under 30 px; use a 0.25 m tag at ≤ 3 m |
| Labeling by correcting the model's own boxes makes its misses invisible | `eval_detect.py` says so; label from scratch |
| `rt.infer` on TensorRT 8.5 (the robot) is a different API from 10 (the dev box) | `runtime.py` handles both; only 10 has run, so `engine-check` must pass on the robot before `run` |

## History

Written in stage E (2026-10-14), on the dev box. It ran end to end there under ROS 2 humble:
- the TensorRT engine built by `r1_build_engine`, gated by `r1_parity_check`;
- a replay source as the camera;
- `mission_ctl` grounding `turn_to` live, with the snapshot.

On the robot, nothing has run yet: [docs/stageE_perception.md](../docs/stageE_perception.md) lists what is measured and what is pending.
