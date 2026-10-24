"""The dance policy's observation as the robot builds it: the reference for the runner.

In training (r1_tracking.py, BeyondMimic without state estimation) the policy sees, in order:

  command          52  the reference's joint positions (26) and velocities (26), absolute,
                       articulation order, at the current dance tick
  motion_anchor_ori_b 6  the reference pelvis orientation in the robot pelvis frame: the
                       rotation matrix of inv(q_robot) * q_ref, its first two columns,
                       row by row
  base_ang_vel      3  pelvis angular velocity, pelvis frame (the IMU gyro)
  joint_pos        26  relative to the default pose
  joint_vel        26  absolute
  actions          24  the last action applied
                  ---
                  137  no history, no scaling (the ONNX carries the normalizer)

On the robot the bridge's observation frame already has gyro, joint_pos (relative),
joint_vel and the applied last action (85 floats, v2's layout), and V3 appends the IMU
quaternion (w, x, y, z), the dance tick and the active policy (6 more). What only the
runner can add is the reference: `Reference` is the table it plays (one 10 s loop, 50 Hz),
written by `export` to the binary file the C++ reads.

Yaw: in training the robot starts facing the way the motion faces (yaw 0, +-0.2 rad of
noise), so the reference's heading is the robot's. On the robot the IMU's heading is
arbitrary: the reference is turned to the robot's heading -- every step while it is not
dancing (so a standby inference sees what the first dancing step will see), and frozen at
the heading of the first dancing step from then on. Only yaw: pitch and roll are the
reference's own.

numpy only (Python 3.8): runs on the robot's test box and in run_tests.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import struct
import sys

import numpy as np

V2_FRAME = 85                 # v2's frame: ang_vel 3, gravity 3, cmd 3, jpos 26, jvel 26, act 24
EXTRA = 6                     # V3: quat w x y z, dance tick, policy (0 walk, 1 dance)
FRAME = V2_FRAME + EXTRA
N_JOINTS, N_ACT = 26, 24
OBS_DIM = 2 * N_JOINTS + 6 + 3 + N_JOINTS + N_JOINTS + N_ACT     # 137
_ANG, _JPOS, _JVEL, _ACT = slice(0, 3), slice(9, 35), slice(35, 61), slice(61, 85)
_QUAT, _TICK, _POLICY = slice(85, 89), 89, 90

REF_MAGIC = b"R1DANCEREF1\n"
_REF_HEAD = struct.Struct("<IIIf")        # frames, joints, bodies (1: the pelvis), fps


# ------------------------------------------------------------------ quaternions (w, x, y, z)
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_unit(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def anchor_ori_b(q_robot, q_ref):
    """BeyondMimic's motion_anchor_ori_b: inv(q_robot) * q_ref as a matrix, first two
    columns, flattened row by row (isaaclab subtract_frame_transforms + matrix_from_quat
    [..., :2].reshape)."""
    rel = quat_mul(quat_conj(quat_unit(q_robot)), quat_unit(q_ref))
    return matrix(rel)[:, :2].reshape(-1)


# ------------------------------------------------------------------ the reference
class Reference(object):
    """One loop of the dance at the control rate: joint_pos, joint_vel (frames x 26),
    pelvis quaternion (frames x 4, w x y z)."""

    def __init__(self, joint_pos, joint_vel, pelvis_quat, fps=50.0):
        self.joint_pos = np.asarray(joint_pos, dtype=np.float32)
        self.joint_vel = np.asarray(joint_vel, dtype=np.float32)
        self.pelvis_quat = np.asarray(pelvis_quat, dtype=np.float32)
        self.fps = float(fps)
        self.frames = self.joint_pos.shape[0]

    @classmethod
    def from_npz(cls, path, loop_frames=500, pelvis="pelvis_link"):
        """The first `loop_frames` of a csv_to_npz_r1.py motion (its first loop)."""
        d = np.load(str(path))
        bodies = [str(b) for b in d["body_names"]]
        i = bodies.index(pelvis)
        n = loop_frames
        return cls(d["joint_pos"][:n], d["joint_vel"][:n], d["body_quat_w"][:n, i], float(d["fps"][0]))

    def write(self, path):
        with open(str(path), "wb") as f:
            f.write(REF_MAGIC)
            f.write(_REF_HEAD.pack(self.frames, self.joint_pos.shape[1], 1, self.fps))
            for a in (self.joint_pos, self.joint_vel, self.pelvis_quat):
                f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())

    @classmethod
    def read(cls, path):
        raw = pathlib.Path(str(path)).read_bytes()
        if not raw.startswith(REF_MAGIC):
            raise ValueError("{} is not a dance reference".format(path))
        off = len(REF_MAGIC)
        n, j, _, fps = _REF_HEAD.unpack_from(raw, off)
        off += _REF_HEAD.size
        arr = np.frombuffer(raw, dtype="<f4", offset=off)
        jp = arr[:n * j].reshape(n, j)
        jv = arr[n * j:2 * n * j].reshape(n, j)
        pq = arr[2 * n * j:2 * n * j + 4 * n].reshape(n, 4)
        return cls(jp, jv, pq, fps)

    def at(self, tick):
        k = int(tick) % self.frames
        return self.joint_pos[k], self.joint_vel[k], self.pelvis_quat[k]


class DanceObs(object):
    """The runner's dance observation, one call per bridge frame (stateful: the yaw)."""

    def __init__(self, reference):
        self.ref = reference
        self.yaw0 = None          # the heading the dance is played at, frozen while dancing

    def __call__(self, frame):
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape[0] != FRAME:
            raise ValueError("frame has {} floats, expected {}".format(frame.shape[0], FRAME))
        q_robot = quat_unit(frame[_QUAT])
        tick = int(round(frame[_TICK]))
        dancing = frame[_POLICY] > 0.5 and tick >= 0
        if not dancing or self.yaw0 is None or tick == 0:
            self.yaw0 = yaw_of(q_robot)
        if not dancing:
            tick = 0                  # standby: the pose the dance starts from
        jp, jv, q_ref = self.ref.at(tick)
        q_ref = quat_mul(yaw_quat(self.yaw0), quat_unit(q_ref))
        obs = np.concatenate([jp, jv, anchor_ori_b(q_robot, q_ref), frame[_ANG], frame[_JPOS],
                              frame[_JVEL], frame[_ACT]])
        return obs.astype(np.float32)


FIXTURE_MAGIC = b"R1DANCEOBS1\n"


def fixture(ref, path, seed=0):
    """Frames and the observations DanceObs builds from them, for the C++ test: standby
    with the heading wandering, a dance through a loop's end, a stop, standby, a second
    dance. Written as magic, count (u32), then per step 91 + 137 float32."""
    rng = np.random.RandomState(seed)
    plan = [(0, -1)] * 12 + [(1, k) for k in range(0, 40)] + [(1, k) for k in range(480, 530)] \
        + [(0, -1)] * 10 + [(1, k) for k in range(0, 20)]
    builder = DanceObs(ref)
    frames, obs = [], []
    yaw = 0.3
    for policy, tick in plan:
        yaw += rng.uniform(-0.02, 0.02)
        tilt = rng.normal(0.0, 0.03, 2)
        q = quat_unit(quat_mul(yaw_quat(yaw), quat_unit([1.0, tilt[0], tilt[1], 0.0])))
        f = np.zeros(FRAME, dtype=np.float32)
        f[:V2_FRAME] = rng.normal(0.0, 0.5, V2_FRAME)
        f[_QUAT] = q
        f[_TICK] = tick
        f[_POLICY] = policy
        frames.append(f)
        obs.append(builder(f))
    with open(str(path), "wb") as fh:
        fh.write(FIXTURE_MAGIC)
        fh.write(struct.pack("<I", len(frames)))
        for f, o in zip(frames, obs):
            fh.write(f.astype("<f4").tobytes())
            fh.write(o.astype("<f4").tobytes())
    return len(frames)


def sha256(path):
    return hashlib.sha256(pathlib.Path(str(path)).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Write the runner's dance reference from a motion npz.")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loop-frames", type=int, default=500)
    ap.add_argument("--fixture", default="", help="also write the C++ test's fixture here")
    args = ap.parse_args()
    ref = Reference.from_npz(args.npz, args.loop_frames)
    ref.write(args.out)
    back = Reference.read(args.out)
    same = all(np.array_equal(a, b) for a, b in ((ref.joint_pos, back.joint_pos),
                                                  (ref.joint_vel, back.joint_vel),
                                                  (ref.pelvis_quat, back.pelvis_quat)))
    print("{}: {} frames x {} joints at {:g} fps, read back {}; sha256 {}".format(
        args.out, ref.frames, ref.joint_pos.shape[1], ref.fps, "identical" if same else "DIFFERENT",
        sha256(args.out)))
    if args.fixture:
        n = fixture(back, args.fixture)
        print("{}: {} steps".format(args.fixture, n))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
