"""Turn the dance CSV (routine.py) into BeyondMimic's motion npz, for R1, offline.

R1's copy of the upstream converter (v3/third_party/whole_body_tracking/scripts/
csv_to_npz.py, MIT). The conversion itself is unchanged: the motion is resampled to the
policy rate, written into the simulated robot one frame at a time without stepping the
physics, and the robot's own body poses and velocities are logged. What differs:

  * the robot is R1 (assets/r1/r1.py) and the CSV's joint columns are routine.ART_JOINTS;
  * the npz is written to --out instead of being uploaded to a WandB registry;
  * --video films the replay from the front-left, so the dance can be looked at before
    anything is trained (V3 plan 01, T1).

Run from the project root:
    ~/IsaacLab/isaaclab.sh -p v3/dance/csv_to_npz_r1.py --headless \\
        --csv outputs/v3/dance/routine_wave.csv --out outputs/v3/dance/routine_wave.npz \\
        [--video outputs/v3/dance/routine_wave.mp4]

Isaac Sim may swallow stdout when it closes, so the summary is also written next to --out.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

parser = argparse.ArgumentParser(description="Dance CSV -> BeyondMimic motion npz (R1).")
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--input_fps", type=float, default=50.0)
parser.add_argument("--output_fps", type=float, default=50.0)
parser.add_argument("--out", type=str, required=True, help="npz to write")
parser.add_argument("--video", type=str, default="", help="mp4 of the replay (needs cameras)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp  # noqa: E402

from assets.r1.r1 import R1_CFG, require_usd  # noqa: E402
import routine  # noqa: E402

require_usd()


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)))
    robot: ArticulationCfg = R1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class Motion:
    """The CSV, resampled to output_fps: positions lerped, the root rotation slerped."""

    def __init__(self, path, input_fps, output_fps, device):
        raw = torch.from_numpy(np.loadtxt(path, delimiter=",")).to(torch.float32).to(device)
        self.pos_in = raw[:, :3]
        self.rot_in = raw[:, 3:7][:, [3, 0, 1, 2]]          # xyzw -> wxyz
        self.dof_in = raw[:, 7:]
        if self.dof_in.shape[1] != len(routine.ART_JOINTS):
            raise SystemExit("{} has {} joint columns, routine.ART_JOINTS has {}".format(
                path, self.dof_in.shape[1], len(routine.ART_JOINTS)))
        n_in = raw.shape[0]
        self.duration = (n_in - 1) / input_fps
        self.dt = 1.0 / output_fps
        times = torch.arange(0, self.duration, self.dt, device=device, dtype=torch.float32)
        phase = times / self.duration * (n_in - 1)
        i0 = phase.floor().long()
        i1 = torch.clamp(i0 + 1, max=n_in - 1)
        w = (phase - i0).unsqueeze(1)
        self.pos = self.pos_in[i0] * (1 - w) + self.pos_in[i1] * w
        self.dof = self.dof_in[i0] * (1 - w) + self.dof_in[i1] * w
        self.rot = torch.stack([quat_slerp(self.rot_in[a], self.rot_in[b], float(c))
                                for a, b, c in zip(i0, i1, w[:, 0])])
        self.lin_vel = torch.gradient(self.pos, spacing=self.dt, dim=0)[0]
        self.dof_vel = torch.gradient(self.dof, spacing=self.dt, dim=0)[0]
        q_rel = quat_mul(self.rot[2:], quat_conjugate(self.rot[:-2]))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.dt)
        self.ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        self.frames = times.shape[0]


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / args_cli.output_fps)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0)
    if args_cli.video:
        scene_cfg.camera = CameraCfg(
            prim_path="/World/replay_cam", update_period=0.0, height=540, width=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)))
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    joint_ids = robot.find_joints(routine.ART_JOINTS, preserve_order=True)[0]
    motion = Motion(args_cli.csv, args_cli.input_fps, args_cli.output_fps, sim.device)
    cam = scene["camera"] if args_cli.video else None
    if cam is not None:
        eye = torch.tensor([[2.2, 1.2, 1.25]], device=sim.device)
        target = torch.tensor([[0.0, 0.0, 0.85]], device=sim.device)
        cam.set_world_poses_from_view(eye, target)

    log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                           "body_ang_vel_w")}
    images = []
    for i in range(motion.frames):
        root = robot.data.default_root_state.clone()
        root[:, :3] = motion.pos[i]
        root[:, :2] += scene.env_origins[:, :2]
        root[:, 3:7] = motion.rot[i]
        root[:, 7:10] = motion.lin_vel[i]
        root[:, 10:] = motion.ang_vel[i]
        robot.write_root_state_to_sim(root)
        jp = robot.data.default_joint_pos.clone()
        jv = robot.data.default_joint_vel.clone()
        jp[:, joint_ids] = motion.dof[i]
        jv[:, joint_ids] = motion.dof_vel[i]
        robot.write_joint_state_to_sim(jp, jv)
        sim.render()                                  # kinematic replay: no physics step
        scene.update(sim.get_physics_dt())
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())
        if cam is not None:
            images.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().copy())

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {k: np.stack(v) for k, v in log.items()}
    np.savez(str(out), fps=np.array([args_cli.output_fps]), joint_names=np.array(robot.joint_names),
             body_names=np.array(robot.body_names), **data)
    feet = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    drift = float(np.abs(data["body_pos_w"][:, feet] - data["body_pos_w"][:1, feet]).max())
    lines = ["{}: {} frames at {:g} fps ({:.2f} s), {} joints, {} bodies".format(
                 out, motion.frames, args_cli.output_fps, motion.frames / args_cli.output_fps,
                 data["joint_pos"].shape[1], data["body_pos_w"].shape[1]),
             "joint order matches routine.ART_JOINTS: {}".format(list(robot.joint_names) == routine.ART_JOINTS),
             "feet move at most {:.4f} m over the motion".format(drift)]
    if images:
        import imageio.v2 as imageio
        video = Path(args_cli.video)
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(video), images, fps=int(round(args_cli.output_fps)), macro_block_size=8)
        lines.append("video: {} ({} frames)".format(video, len(images)))
    text = "\n".join(lines)
    print(text, flush=True)
    out.with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
