# The v2 policy: `2026-10-16_17-57-14_v2_drift_ft`

Walks backward and sideways, turns faster, and steps in place without creeping forward
on a zero command. **The policy of release v2.0.0.** On the robot (2026-10-20) it walked
backward and sideways and held its place on a zero command. The v1 policy,
[`week04_nohead`](../week04_nohead/), stays in the release for rollback. The write-up is
[docs/v2_policy.md](../../docs/v2_policy.md); the task, and why each change was made:
[tasks/r1_flat/flat_v2_env_cfg.py](../../tasks/r1_flat/flat_v2_env_cfg.py) and
[tasks/r1_flat/README.md](../../tasks/r1_flat/README.md#v2).

| file | what it is |
|---|---|
| `model_4799.pt` | rsl_rl checkpoint, last iteration of stage 2 |
| `policy.pt` | TorchScript actor: 425-dim observation → 24-dim action (same interface as v1) |
| `policy.onnx` | the same actor as ONNX, **the artifact that ships to the robot** |
| `parity_fixture.bin` | 512 observation vectors and this policy's reference outputs |
| `interface/` | the observation/action contract and per-joint gains, as `policy_pack/make_bundle.py --interface` takes them; identical to v1's except for the run id and task name |

`SHA256SUMS` lists all six.

## Command envelope (training's final command ranges)

| | v1 | v2 |
|---|---|---|
| vx | 0 → 1.0 m/s | −0.6 → 1.0 m/s |
| vy | 0 | −0.5 → 0.5 m/s |
| wz | ±0.5 rad/s | ±1.57 rad/s |

## Measured in simulation

`scripts/eval_command_grid.py`, 64 envs, nominal conditions, 10 s per command:

| command | v1 | v2 |
|---|---|---|
| 0 (zero command) | +0.046 m/s, 0.45 m per 10 s | +0.014 m/s, 0.14 m per 10 s; stepping (air 0.47/0.47) |
| zero, over 256 randomization draws | — | +0.007 m/s mean (p10/p90 −0.006/+0.023) |
| vx 0.1 / 0.5 / 1.0 | 0.137 / 0.502 / 0.982 | 0.107 / 0.490 / 1.017 |
| vx −0.3 / −0.6 | −0.269 / falls | −0.278 / −0.588 |
| vy ±0.5 | not tracked | ±0.478 |
| wz 1.0 / 1.57 | 0.834 / 1.032 | 0.929 / 1.407 |
| falls on a zero command | 0 | 0 (5 seeds × 256 envs) |

Gait: alternating (phase 0.50), symmetric (swing ratio 0.99), 1.55 Hz.

## Numerical checks

| check | max_abs |
|---|---|
| ONNX Runtime against PyTorch, fixture's 512 samples | 1.5e-05 |
| TensorRT 10.7 FP32 engine on the dev box, `r1_parity_check` | 2.3e-05 (gate 1e-3) |

The robot's engine is built on the Orin by `policy_pack/install_bundle.sh` and gated
against the same fixture.

## How it was trained

Two stages, seed 42; training is deterministic (re-running stage 2 from the same stage-1
checkpoint gave the same weights bit for bit). The commands are in the docstring of
`flat_v2_env_cfg.py`. Stage 1 is run `2026-10-16_15-51-33_v2` (4000 iterations); stage 2
resumes it for 800 iterations with the zero-command drift penalty.
