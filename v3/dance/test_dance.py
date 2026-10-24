#!/usr/bin/env python3
"""V3 dance without Isaac: the routine's static checks (check_routine.py, at 5 checked
frames per second), its timing as the conductor counts it, and the runner's dance
observation -- the Python reference must still produce the fixture the C++ test reads.
numpy only.

  python3 v3/dance/test_dance.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_routine as CR  # noqa: E402
import dance_obs as DO  # noqa: E402
import routine as RT  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok    " + name)
    else:
        FAIL += 1
        print("  FAIL  {}  <- {}".format(name, detail))


print("== the routine ==")
check("a 10 s loop of bars 3 + 3 + 2 + 2 s", RT.LOOP_S == 10.0 and [b[1] for b in RT.BARS] == [3.0, 3.0, 2.0, 2.0])
q0 = RT.joints_at(0.0)
check("it starts in the default pose", all(abs(q0[j] - RT.DEFAULT[j]) < 1e-12 for j in RT.ART_JOINTS))
if CR.urdf_fk.URDF.exists():
    for ok, label, detail in CR.run(fps=5.0, verbose=False):
        check("check_routine: " + label, ok, detail)
else:   # an installed robot tree has no assets/: the checks ran on the dev box
    print("  skip  check_routine: no {} here".format(CR.urdf_fk.URDF))

print("== the runner's dance observation ==")
RUNNER = next(d for d in (HERE.parent / "ros2_ws" / "src" / "r1_policy_runner",
                          HERE.parent / "deploy" / "ros2_ws" / "src" / "r1_policy_runner") if d.is_dir())
data = RUNNER / "test" / "data"
ref = DO.Reference.read(data / "dance_ref.bin")
check("the reference: 500 frames at 50 Hz, frame 0 = the default pose",
      ref.frames == 500 and ref.fps == 50.0
      and np.abs(ref.joint_pos[0] - [RT.DEFAULT[j] for j in RT.ART_JOINTS]).max() < 1e-6)
def fixture_floats(path):
    raw = pathlib.Path(path).read_bytes()
    return np.frombuffer(raw, dtype="<f4", offset=len(DO.FIXTURE_MAGIC) + 4)


with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp) / "fixture.bin"
    DO.fixture(ref, out)
    # numerically, not byte for byte: numpy versions differ in the last bit of a trig function
    a, b = fixture_floats(out), fixture_floats(data / "dance_obs_fixture.bin")
    check("dance_obs.py still gives the observations in the C++ test's fixture (to 1e-6)",
          a.shape == b.shape and float(np.abs(a - b).max()) <= 1e-6,
          "max diff {}".format(float(np.abs(a - b).max()) if a.shape == b.shape else "shape"))
q = DO.quat_unit([0.9, 0.1, -0.2, 0.3])
check("the orientation term is the identity's [1,0,0,1,0,0] for the robot's own orientation",
      np.allclose(DO.anchor_ori_b(q, q), [1, 0, 0, 1, 0, 0]))

print("--------------------------------------------------")
print("pass {}  fail {}".format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
