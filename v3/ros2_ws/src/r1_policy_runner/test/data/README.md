# Test data for test_dance_obs.cpp

Generated, not hand-written; regenerate both together after the dance changes:

```bash
deploy/.venv/bin/python v3/dance/dance_obs.py --npz outputs/v3/dance/routine_wave.npz \
    --out v3/ros2_ws/src/r1_policy_runner/test/data/dance_ref.bin \
    --fixture v3/ros2_ws/src/r1_policy_runner/test/data/dance_obs_fixture.bin
```

- `dance_ref.bin`: one 10 s loop of the dance reference (v3/dance/routine.py, through
  csv_to_npz_r1.py), the table the runner plays.
- `dance_obs_fixture.bin`: bridge frames and the observations the Python reference
  (v3/dance/dance_obs.py) builds from them. That Python was checked against Isaac's own
  observation of the same state (v3/sim/eval_handover.py: max difference 1.1e-7).
