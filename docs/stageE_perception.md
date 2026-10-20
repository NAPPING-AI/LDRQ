# Stage E: seeing objects (YOLOX-Nano) and walking to them

Stage E adds vision to the command layer: "Face the chair", "Walk to the woman by the window and stop 1.5 m from her". A YOLOX-Nano detector runs on the robot's Orin NX next to the 50 Hz walking policy, and its detections feed two new primitives, `turn_to` and `walk_to`.

**Status (2026-10-14): built and tested on the dev box; two robot sessions done (engines, parity, coexistence, output off); the coexistence gate passes.** No camera is connected to the Orin NX itself: the head's stereo module streams to it over the network (Unitree's R1 "video stream service" page, 2026-10-16), and the node now reads that stream (`source: stereo`); nothing that needs real frames has run on the robot yet. Every robot measurement still missing is marked *pending*, together with the command that produces it.

| item | where it stands |
|---|---|
| `r1_perception` node: rectify → YOLOX-Nano (TensorRT) → bearing, depth, range, safety facts at 10–15 Hz | **done on the dev box** (ROS 2 humble, replayed frames): 12.4 Hz |
| engine built on the robot, parity gate | **robot: FP16 passes** (detections gate, C++ build, Python runtime on TensorRT 8.5); FP32 misses the raw gate by a hair (1.37e-3 vs 1e-3), detections identical |
| FP32 vs FP16, latency and detections | **robot** (clocks locked): FP16 3.7 ms vs FP32 5.8 ms, 18/18 detections matched. **FP16 is the robot's engine** |
| `turn_to`, `walk_to`; object table; grounding; the operator's picture | **done**; tested in simulation and live against the node |
| English: the two new steps | **evaluated on the dev box** with the model on the CPU, as on the robot; robot rerun pending |
| safety: a person near or an obstacle stops the walk; a quiet camera stops it | **done**; tested in simulation |
| N-frame confirmation; thresholds calibrated on robot images | confirmation **done**; calibration **pending** (~100 images) |
| `coexist.py`: policy p99 ≤ 2 ms, obs 50 Hz, no DEGRADED with YOLO running | **robot: PASS** with FP16 in every phase, `yolo+ask` included (p99 ≤ 592 µs; idle 489). A scheduling flaw found in the first session (FP32 `yolo+ask` p99 5.2 ms) is fixed and confirmed on the robot (593 µs) |
| AprilTag bearing / depth error | tool **built, checked on synthetic data**; **robot pending** |
| 10 × `turn_to`, 10 × `walk_to` on the robot | **pending**; the same protocol in simulation is below |
| `./run_tests.sh`: no-robot tests | **done**: three new suites and a `walk_to` dry run |

**The camera is the head's stereo module** (Unitree's R1 "video stream service" page; the facts below are the page's, not yet measured):
- **Streams.** It pushes UDP RTP/H264 to the Orin NX: a 1280×720 image on port 5001, and the left and right cameras at 544×448 on ports 5002 and 5003.
- **Depth.** 544×448, uint16 millimetres, arrives in the v4l2loopback `/dev/video-dep` once `sudo unitree_depth_start` runs.
- **Setup after every boot.** In the APP, turn video_hub off and "Stereo patch PC1" on, then start depth.
- **Calibration.** It is downloadable as `cam_param.txt` from the module's web page.

Stage E uses the **left camera with depth** (`source: stereo`). A gst-launch child decodes the H264, with the Jetson hardware decoder when present. Depth is paired with a frame only if it is younger than 0.3 s. Two things are assumed until session 3 checks them:
- depth is registered to the left image (same size);
- the intrinsics (placeholders until `cam_param.txt`).

Without fresh depth, `walk_to` refuses and `turn_to` still works.

## The design in one page

```
camera (150 deg) ─ rectify to a 100 deg pinhole view ─ YOLOX-Nano (TensorRT, GPU) ─ NMS
   ─ per detection: bearing (left +), elevation, median depth in the box, range
   ─ tracker: hits in a row ─ per-frame guard: person near? obstacle in the corridor?
   ─ /r1_perception/detections (JSON, ~12.5 Hz)
                                  │
ask "Walk to the red chair" ─ model copies "the red chair" ─ table: chair ─ plan
   ─ grounding (standing): exactly one confirmed chair, confident, with depth
   ─ picture with the target in red ─ operator [y/N]
   ─ executor: look ─ turn by the bearing (IMU-closed) ─ walk ≤ 1 m (guarded) ─ look ─ ...
```

Decisions, and why:

- **The language model moved to the CPU** (user, 2026-10-14). In stage C, four CPU threads held policy inference at 542 µs p99 against 5 ms with the model on the GPU ([stageC_nl_eval.md](stageC_nl_eval.md)). The GPU now belongs to the policy and the detector. The cost is speed: about 1–2 s per instruction at 21 tok/s.
- **Scheduling, not a faster model, protects the policy.** The policy publishes `/r1_policy_node/action` right after each inference. The detector runs on every fourth one, which gives 12.5 Hz, so its GPU work falls in the ~19 ms the policy leaves idle. The frame preparation, which is CPU only, runs one step earlier so that it cannot push the GPU work late (a robot finding, below). This is what stage E relies on for p99 ≤ 2 ms. If the robot measurement still misses, the fallback is to move YOLO to one of the Orin NX's DLA cores. That is a third optimization, outside the two this stage allows, so it needs a decision first.
- **The same parity gate as the walking policy.** `r1_build_engine` and `r1_parity_check` handle any single static-shape FP32 input and output, so YOLOX goes through them unchanged, with a fixture in the same R1FX format. The node runs TensorRT from Python, so that path has its own gate: `engine-check` runs the same fixture.
- **Two thresholds.** One is for safety (low, single frame: a person must not be missed). The other is for targets (higher, three frames in a row: the robot must not walk to the wrong thing). One number cannot serve both, so they are calibrated separately.
- **The model still only transcribes.** It copies the words that name the object. Every word must have been said; a table maps the head noun to a COCO class or refuses; the detector decides whether the object is there and unique; the operator confirms on a picture.
- **One object step per plan, first among the moving steps.** The target is confirmed on a picture taken before the plan starts. "Walk 2 m, then go to the chair" would act on a chair nobody looked at, so it is refused.
- **Nothing is steered while walking.** The robot does not turn while walking (stage B). `walk_to` measures standing, turns by the bearing on the IMU, walks at most 1 m open loop, stops, measures again, and when it arrives within 0.2 m of the stop distance (the shortest walk it can make) it faces the target once more. Distances are re-measured every metre, so open-loop error does not accumulate.
- **The camera guards every walk inside them.** A person within 1.2 m, a return inside the walking corridor within 0.6 m (computed in 3-D, so the floor never counts), or frames older than 0.5 s stop the plan at once, standing. The person check acts on single frames at the safety threshold, by design.

Code: [perception/](../perception/README.md) (the node, the tools) and [mission_ctl/](../mission_ctl/README.md) (`objects.py`, `percept.py`, the executor's object phases).

## Measured so far (dev box)

The dev box is an RTX 2080 Ti with TensorRT 10.7. Latencies here are the dev box's and **say nothing about the Orin**. What carries over is the method and the numerical agreement.

### Model, engine, parity

YOLOX-Nano is Megvii's release 0.1.1rc0 ONNX: opset 11, static `images [1,3,416,416]` → `output [1,3549,85]`, sha256 `c789161e…0b7d` (`perception/models/SHA256SUMS`). The grid decode is not in the graph: it has Sigmoid but no Exp. Decoding YOLOX's own sample image gives dog 0.86, bicycle 0.80, car 0.80, which matches the reference demo.

| engine (built by `r1_build_engine`) | raw max_abs vs ONNX | detections vs ONNX | compact p50 / p99 | paced 12.5 Hz p50 / p99 |
|---|---|---|---|---|
| FP32 (TF32 cleared) | **3.8e-04** (gate 1e-3: PASS, C++ tool and Python runtime alike) | identical | 1.73 / 2.44 ms | 1.85 / 2.19 ms |
| FP16 | 1.29 (raw gate fails, as expected) | **13/13 matched, IoU ≥ 0.994, confidence within 0.006** | 1.49 / 1.94 ms | 1.65 / 1.93 ms |

The FP16 result is why `engine-check` has two gates:
- **FP32:** gated on raw outputs.
- **FP16:** gated on detections. FP16 moves the raw values of background anchors by about 1 without changing one box. A raw tolerance would only tell you that FP16 is not FP32.

The FP32 result comes from a 4-image fixture (the sample image, mirrored and cropped). On the robot, the fixture is built from the robot's own frames.

The node ran under ROS 2 humble on replayed frames at 12.4 Hz:
- YOLO alone: p50 2.06 ms;
- the whole frame (rectify, letterbox, infer, decode, depth, guard): p50 3.7 ms;
- trigger: `timer`, because there is no policy on the dev box.

`mission_ctl` grounded `turn_to the dog` against it live, saved the operator's picture, and refused `walk_to the dog` because this source has no depth.

### English: the prompt with two new steps

The prompt gained the `turn_to`/`walk_to` steps and two examples. One iteration followed on the dev sets only, which taught it that "turn around to face it" is `turn_to`. A fix in the deterministic normalizer also stops "turn around to face the dog" from being rewritten as 180°. The resulting prompt is `2bb25e8e2ecc`.

A new held-out set of 20 object instructions (`nl_test4.jsonl`, sha256 `d5150a83…`) was written and frozen before any model saw it. All scoring was on Ollama 0.34.4 with qwen3:1.7b **on 4 CPU threads**, the robot's placement.

| set | correct (as authored) | unsafe | stage C, same cases |
|---|---|---|---|
| test, test2, test3 (stage C's 80) | **74/80** (77/80 counted by outcome) | 1 | 77/80, 1 unsafe |
| test4 (objects, new) | **20/20** (19 exact transcriptions) | 0 | — |
| dev / dev objects (design only) | 17/20, 8/8 | 0 | — |

The three cases that changed on the old sets still refuse, but now for a different reason:
- **T38 "Walk to the door", U19 "Go to the kitchen" and V16 "Look at me":** the model now writes these as object steps, and the table refuses them (`unknown_object`). The authored reasons were `unsupported`/`no_amount`, so they are scored wrong. Nothing moves in either version.
- **T38 in stage C** passed only because the model invented "walk 1 m" and that number was dropped.
- **The frozen sets are not edited after scoring**, so both numbers are reported.
- **The remaining misses are stage C's own:** V12 ("run 5 m" walked as a walk: the known unsafe case), V13 and U17.

On the CPU the model decodes at 18–19 tok/s on the dev box, p50 1.4–2.0 s per instruction.

### The trial protocol, simulated

`mission_ctl/eval/sim_object_trials.py` runs the robot's acceptance protocol in a simulated room: 10 × `turn_to` and 10 × `walk_to`, targets 8–45° off and 1.8–4.5 m away, stop distances 0.8/1.0/1.5 m. The robot follows commands exactly, and the camera adds Gaussian noise (bearing sd 1.5°, range sd 4 cm). This measures the control logic alone; the robot adds the real camera, depth and gait.

| | success | final heading error (truth) | stop distance error (truth) |
|---|---|---|---|
| `turn_to` × 10 | 10/10 | mean 2.8°, max 5.3° | — |
| `walk_to` × 10 | 10/10 | mean 3.4°, max 4.8° | mean +0.3 cm, \|err\| mean 3.6 cm, max 6.3 cm |

`walk_to` took 2–4 walks and 15–32 s.

The no-robot tests (`mission_ctl/tests/test_objects.py`) also cover:
- a person stepping into the path, which stops the walk at once;
- something in the corridor below the target threshold, which stops it as an obstacle;
- the camera going quiet mid-walk, which stops it within 0.5 s;
- two chairs, which is refused as ambiguous;
- a chair outside the view, a target beyond the trusted depth range, or no depth: each refused before anything moves.

### The evaluation tools, checked on synthetic data

| tool | check | result |
|---|---|---|
| camera model vs OpenCV's fisheye | 200 random points | 2e-13 px |
| `calibrate_fisheye.py` | 30 rendered checkerboard views, known lens | fx 488.8 (true 489), principal point within 0.6 px, RMS 0.44 px; ray error ≤ 0.86° within the 100° view |
| `eval_tag.py` | 11 rendered 0.25 m tags, planted errors +0.8° and +5 cm | +0.73° and +1.6 cm: within the tag's own ~3 cm range uncertainty at this resolution |
| `eval_detect.py` | 10 labeled frames, planted hits and misses | every count as planted; thresholds suggested accordingly |

Two lessons from building these tools:
- **OpenCV's fisheye calibration does not start on a wide lens without a starting guess.** The tool gives it the nominal field of view.
- **A small AprilTag is not ground truth.** Its range comes from its apparent size, so tags under 30 px are skipped. Stop distances are also measured with a tape measure.

## On the robot: pending

The full plan, in Chinese, is in the project's stage plan (private). Every command below exists and has run on the dev box.

1. **Probe the camera** (moves nothing): `python3 perception/r1_perception_cli.py probe`. It records devices, ROS image topics with encodings, rates and a frame each, depth statistics, and whether `tensorrt`, `cv2` and `libcudart` are present. Fill the PROBE lines in `perception/config/perception.yaml`.

   **Robot, 10-14:** the probe hung and its output was lost. It found one V4L2 device, a `v4l2loopback` named `Depth_Display` (`/dev/video102`, linked as `/dev/video-dep`), with output capabilities only. OpenCV's `read()` on it never returns when nothing writes to it, and Ctrl-C also killed `tee`, so the buffered lines died with the process. The probe now runs each device read in a child process killed after 8 s, skips output-only devices, line-buffers, and saves `probe.json` after each part. The loopback's name suggests that a vendor process turns a depth camera's stream into a display; the camera itself is still unknown.

   **Robot, second probe (fixed probe, 10-14):** it completed. The Orin NX has **no camera**:
   - `lsusb` shows only the two root hubs, so no USB device is attached;
   - there are no ROS image topics;
   - no camera process is running;
   - the only V4L2 device is the `Depth_Display` loopback (`video_nr=102`, `exclusive_caps=Y`), created at boot, with nothing writing to it.

   The head camera must be wired to another computer in the robot, and its frames must reach this one over the network. How they do is the next question. Until then, `turn_to` and `walk_to` can run only on replayed frames.
2. **Calibrate:** 20–30 checkerboard views with `capture`, then `tools/calibrate_fisheye.py` on the dev box. Check that straight lines come out straight.
3. **Engine and parity:**
   - build the fixture from robot frames (dev box);
   - `r1_build_engine` FP32 and FP16, then `r1_parity_check` (FP32);
   - `engine-check` (FP32 raw gate, FP16 detection gate);
   - `compare` on ~100 frames with the clocks locked.

   Robot, 10-14, clocks locked, fixture and frames still YOLOX's sample images (not robot frames):

   | engine | build | parity | p50 / p99 paced | detections |
   |---|---|---|---|---|
   | FP32 (TF32 cleared) | 120 s | `r1_parity_check` and `engine-check`: max_abs **1.37e-3, FAIL** (gate 1e-3) | 5.83 / 5.99 ms | 13/13 vs ONNX, IoU 1.000 |
   | FP16 | 402 s | `engine-check` detections gate: **PASS** | 3.74 / 3.87 ms | 13/13 vs ONNX, IoU ≥ 0.992; vs FP32 18/18 on 5 frames |

   The FP32 miss is one box-offset output (channel 1 of a stride-8 anchor, reference 1.209) off by 1.4e-3 grid cells, about 0.01 px; every detection is identical. The gate was not widened to let it through. FP32 was dropped instead: FP16 passes its own gate and is 1.6× faster, and that time is what the policy shares. On the dev box FP32 passed at 3.8e-4, so TensorRT 8.5 chooses different FP32 kernels.
4. **Coexistence** (clocks locked, stack up, output off, then on):

   ```bash
   python3 llm/coexist.py --stack-log ... --phases idle,yolo,ask,yolo+ask,idle
   ```

   Gate, in the yolo phases: policy inference p99 ≤ 2000 µs, obs ≥ 45 Hz with no slow window, no DEGRADED, detector 10–15 Hz.

   Robot, 10-14, output off, the model on the CPU (4 threads), replayed 1280×720 frames (GPU load does not depend on the picture). Idle: policy p99 486 µs.

   | engine | phase | policy p50 / p99 / max | obs Hz min | DEGRADED | detector Hz / YOLO p99 | verdict |
   |---|---|---|---|---|---|---|
   | FP32 | yolo | 468 / 1650 / 3396 µs | 50.0 | 0 | 12.4–12.6 / 7.0 ms | PASS (barely) |
   | FP32 | yolo+ask | 533 / **5244** / 5350 µs | 50.0 | 0 | 12.4–12.6 / 7.7 ms | **FAIL** |
   | FP16 | yolo | 465 / **553** / 1727 µs | 50.0 | 0 | 12.4–12.6 / 4.0 ms | **PASS** |
   | FP16 | yolo+ask | *pending* | | | | |

   Second session, 10-14, with the fix below:

   | engine | phase | policy p50 / p99 / max | obs Hz min | DEGRADED | detector Hz / YOLO p99 | verdict |
   |---|---|---|---|---|---|---|
   | FP16 | yolo | 475 / 552 / 625 µs | 50.0 | 0 | 12.4–12.6 / 3.9 ms | **PASS** |
   | FP16 | ask | 521 / 583 / 767 µs | 50.0 | 0 | — | PASS |
   | FP16 | yolo+ask | 532 / **592** / 762 µs | 50.0 | 0 | 12.4–12.6 / 4.6 ms | **PASS** |
   | FP32 | yolo+ask | 530 / **593** / 1380 µs | 50.0 | 0 | 12.4–12.6 / 7.0 ms | **PASS** (was 5244 µs) |

   Every load phase also meets the plan 3.6 targets (lag p95 ≤ idle + 1 ms). With FP16 the `yolo+ask` phase costs the policy about 100 µs at p99, about what the model alone costs.

   **Why `yolo+ask` failed:** the node did its CPU work (frame, rectify, letterbox: about 6 ms idle, 9 ms with the model busy on the CPU) inside its slot, after the policy's action arrived, and only then started the GPU. With FP32's 7 ms on top, the tail reached the policy's next step, and the policy waited for YOLO's whole inference (p99 5.2 ms ≈ one YOLO run). **Fix:** the node now does the CPU half on the policy step before its slot (`Pipeline.prepare`), so when the slot opens the GPU starts at once (`Pipeline.finish`). The frame is 20 ms older. Checked on the dev box, then on the robot: FP32 `yolo+ask` went from 5244 to 593 µs.
5. **Detection on ~100 robot images:** `capture` around the lab (people, chairs, tables; some empty scenes). Label from scratch (YOLO txt), then run `tools/eval_detect.py`. It reports precision, person recall, and the two thresholds to set.
6. **AprilTag:** a 0.25 m tag36h11 on the target, `capture`, then `tools/eval_tag.py`. It gives bearing error and range error (bias, mean |err|, p95).
7. **Trials:** 10 × `turn_to`, 10 × `walk_to` (gantry first, then untethered). Record success, final heading error (tag) and stop distance error (tape).

   | | success | final heading error | stop distance error |
   |---|---|---|---|
   | `turn_to` × 10 | *pending* | *pending* | — |
   | `walk_to` × 10 | *pending* | *pending* | *pending* |
8. **English on the robot:** the held-out sets again with `--options '{"num_gpu": 0, "num_thread": 4}'`.

## Limits and risks

- **Camera unknown:**
  - no depth, or depth not registered to RGB, means no `walk_to`;
  - a 150° lens leaves few pixels per degree at the edges, so small, far objects are detected poorly;
  - the 100° rectified view is the working field, and objects outside it cannot be targets.
- **Python TensorRT on JetPack 5.1.1:** works (TensorRT 8.5.2.2, `engine-check` and the node both ran on the robot).
- **Scheduling may not be enough:** memory bandwidth and clock effects can still slow the policy with YOLO idle-gapped. The DLA is the fallback and needs a decision. On the robot, YOLO alone moved the FP16 policy's p99 from 486 to 553 µs.
- **CPU contention:** the model (4 threads), the detector's preprocessing and the bridge's 500 Hz loop share 8 cores. The `yolo+ask` phase measures exactly this.
- **Distances stay open loop between measurements.** The stop error depends on the depth camera's accuracy at 1–1.5 m and on the last walk's overshoot, bounded by the 0.2 m arrival tolerance.
- **The English change has a cost:** 3 refusals on the stage C sets now carry a different reason. The frozen sets were not re-authored.
