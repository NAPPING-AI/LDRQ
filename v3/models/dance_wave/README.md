# The V3 dance policy: `2026-10-23_16-51-44_wave`

Dances [the routine](../../dance/routine.py) on the spot (10 s loop: the right arm waves,
the left, both arms up with two knee dips, standing still), both feet on the floor. Runs
next to v2 in the V3 runner fork; the bridge's conductor decides which one drives the robot
([v3/README.md](../../README.md#the-dance-two-policies)). Trained with BeyondMimic's motion
tracking on the task [dance/r1_tracking.py](../../dance/r1_tracking.py), 6000 iterations,
4096 environments.

| file | what it is |
|---|---|
| `model_5999.pt` | rsl_rl checkpoint, last iteration |
| `policy.pt` | TorchScript actor with its observation normalizer: 137-dim observation -> 24-dim action |
| `policy.onnx` | the same as ONNX, **the artifact that ships to the robot** (the engine is built on the Orin) |
| `dance_ref.bin` | the reference the runner plays: one loop, 500 frames at 50 Hz |
| `interface.json` | observation layout, action joints / scales / default pose (v2's), export checks |
| `parity_fixture.bin` | 512 observation vectors and this policy's reference outputs, for `r1_parity_check` |

`SHA256SUMS` lists them. The action term (joints, order, scales, default pose) is v2's
exactly -- checked at export -- because the bridge drives both policies with one set of
gains. Dev-box TensorRT 10.7, fp32: parity 5.3e-5 against the fixture.

Rebuild: `v3/dance/routine.py` -> `v3/dance/csv_to_npz_r1.py` -> `v3/dance/train_dance.py`
-> `v3/dance/export_dance.py` -> `deploy/tools/make_fixture.py --in-dim 137`.
