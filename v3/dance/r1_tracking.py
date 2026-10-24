"""R1 dance task: BeyondMimic's motion tracking (v3/third_party/whole_body_tracking) on R1.

Registered as Isaac-Dance-R1-v3 (train) and Isaac-Dance-R1-Play-v3 (evaluate). Everything
is upstream's TrackingEnvCfg -- rewards, terminations, adaptive start sampling, pushes --
except what R1 and V3's deployment need, set in R1DanceEnvCfg.__post_init__ below:

  robot      assets/r1/r1.py, the same actuators, gains and default pose v2 was trained with:
             the bridge sends ONE set of PD gains, so both policies must share it
  actions    v2's action term exactly (tasks/r1_flat: the 24 joints without the head, the
             per-group scales, 0-1 control steps of delay) instead of upstream's all-joint term
  anchor     pelvis_link: the IMU is in the pelvis, so on the robot the anchor's orientation
             is the IMU's quaternion with no kinematics in between
  policy obs upstream's "without state estimation" variant: no anchor position error, no
             base linear velocity -- the robot measures neither
  DR         upstream's (friction, default-pose offsets, CoM, pushes) plus v2's mass and
             motor-strength randomization, stiffness widened to cover the kp_scale 1.3 the
             robot runs at. The default-pose offsets use randomize_action_joint_default_pos
             below: upstream's version indexes the action offsets by articulation joint id,
             which only works when the action drives every joint in articulation order
             (G1). R1's drives 24 of 26 in another order, and upstream's crashes with a CUDA
             index assert.

The motion file comes from the environment variable R1_DANCE_MOTION (an npz written by
v3/dance/csv_to_npz_r1.py) or the train/eval scripts' --motion argument.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_UPSTREAM = _ROOT / "v3" / "third_party" / "whole_body_tracking" / "source" / "whole_body_tracking"
for p in (str(_ROOT), str(_UPSTREAM)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import whole_body_tracking.tasks.tracking.mdp as wbt_mdp  # noqa: E402
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg  # noqa: E402
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg  # noqa: E402

from assets.r1.r1 import R1_CFG  # noqa: E402
from tasks.r1_flat.flat_env_cfg import ActionsCfg as V2ActionsCfg  # noqa: E402

ANCHOR = "pelvis_link"
# The bodies whose poses are tracked (upstream's G1 list, mapped to R1's links).
BODIES = [
    "pelvis_link",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "waist_yaw_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_link",
]
END_EFFECTORS = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link"]
TORSO = "waist_yaw_link"

# Motor strength: v2 trained with 0.85-1.15 and runs on the robot at kp_scale 1.3
# (stage A: the real robot walks 1.10-1.50). The dance has to hold up over that range.
STIFFNESS_SCALE = (0.85, 1.6)
DAMPING_SCALE = (0.85, 1.15)


def randomize_action_joint_default_pos(env, env_ids, offset_range=(-0.01, 0.01), action_name="joint_pos"):
    """Shift the default pose of the action's joints (encoder calibration error).

    Keeps the action term's offsets in step with the new defaults, mapping through the
    term's own joint ids. Also stores the nominal default as upstream does, for export."""
    robot = env.scene["robot"]
    term = env.action_manager.get_term(action_name)
    ids = torch.as_tensor(term._joint_ids if not isinstance(term._joint_ids, slice)
                          else range(robot.num_joints), device=robot.device, dtype=torch.long)
    robot.data.default_joint_pos_nominal = torch.clone(robot.data.default_joint_pos[0])
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=robot.device)
    rows = torch.as_tensor(env_ids, device=robot.device, dtype=torch.long)[:, None]
    noise = torch.empty((rows.shape[0], ids.shape[0]), device=robot.device).uniform_(*offset_range)
    robot.data.default_joint_pos[rows, ids[None, :]] += noise
    term._offset[rows[:, 0]] = robot.data.default_joint_pos[rows, ids[None, :]]


@configclass
class R1DanceEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = R1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos = V2ActionsCfg().joint_pos

        motion = self.commands.motion
        motion.motion_file = os.environ.get("R1_DANCE_MOTION", "")
        motion.anchor_body_name = ANCHOR
        motion.body_names = list(BODIES)

        # without state estimation (upstream's G1FlatWoStateEstimationEnvCfg)
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.events.add_joint_default_pos = EventTerm(
            func=randomize_action_joint_default_pos, mode="startup", params={"offset_range": (-0.01, 0.01)})
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=TORSO)
        self.events.add_base_mass = EventTerm(
            func=wbt_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": (0.9, 1.1), "operation": "scale"},
        )
        self.events.actuator_gains = EventTerm(
            func=wbt_mdp.randomize_actuator_gains,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": STIFFNESS_SCALE,
                    "damping_distribution_params": DAMPING_SCALE, "operation": "scale"},
        )

        self.terminations.ee_body_pos.params["body_names"] = list(END_EFFECTORS)
        keep = "".join("(?!{}$)".format(b) for b in END_EFFECTORS)
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["^{}.+$".format(keep)])


@configclass
class R1DanceEnvCfg_PLAY(R1DanceEnvCfg):
    """Evaluation: fewer envs, no observation noise, no pushes. DR stays on."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class R1DancePPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "r1_dance_v3"


gym.register(
    id="Isaac-Dance-R1-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:R1DanceEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}:R1DancePPORunnerCfg"},
)
gym.register(
    id="Isaac-Dance-R1-Play-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:R1DanceEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{__name__}:R1DancePPORunnerCfg"},
)
