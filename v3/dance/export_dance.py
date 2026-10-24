# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained dance policy for the robot: what the V3 runner loads.

Writes into --out (e.g. v3/models/dance_<run>/):
  policy.pt / policy.onnx   the actor with its observation normalizer folded in, as v2's
                            were exported (scripts/play_r1.py): the runner feeds raw values
  dance_ref.bin             the reference the runner plays (v3/dance/dance_obs.py)
  interface.json            what the runner and the bridge must agree on: the observation
                            layout (137), the action joints, scales and default pose (they
                            must be v2's, the bridge sends one set of gains), the run and
                            checkpoint, sha256 of every file
It also checks, on observations from the task itself, that the TorchScript and the
inference policy the checkpoint loads into give the same actions.

Run from the project root (env_isaaclab):
    ~/IsaacLab/isaaclab.sh -p v3/dance/export_dance.py --headless \\
        --checkpoint logs/rsl_rl/r1_dance_v3/<run>/model_<n>.pt \\
        --motion outputs/v3/dance/routine_wave.npz --out v3/models/dance_<run>
Then the parity fixture: deploy/tools/make_fixture.py --policy <out>/policy.pt --in-dim 137.
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

parser = argparse.ArgumentParser(description="Export the V3 dance policy for the runner.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--motion", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import dance_obs as DO  # noqa: E402
import r1_tracking  # noqa: E402
import routine  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg("Isaac-Dance-R1-Play-v3", device=args_cli.device, num_envs=16)
    env_cfg.commands.motion.motion_file = str(Path(args_cli.motion).resolve())
    env = RslRlVecEnvWrapper(gym.make("Isaac-Dance-R1-Play-v3", cfg=env_cfg))
    uw = env.unwrapped
    runner = OnPolicyRunner(env, r1_tracking.R1DancePPORunnerCfg().to_dict(), log_dir=None, device=uw.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=uw.device)
    nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    export_policy_as_jit(nn, runner.obs_normalizer, path=str(out), filename="policy.pt")
    export_policy_as_onnx(nn, normalizer=runner.obs_normalizer, path=str(out), filename="policy.onnx")

    # the TorchScript against the checkpoint's own inference policy, on real observations
    jit = torch.jit.load(str(out / "policy.pt"), map_location=uw.device).eval()
    obs, _ = env.get_observations()
    worst = 0.0
    with torch.inference_mode():
        for _ in range(100):
            a = policy(obs)
            worst = max(worst, float((jit(obs) - a).abs().max()))
            obs, _, _, _ = env.step(a)

    ref = DO.Reference.from_npz(args_cli.motion)
    ref.write(out / "dance_ref.bin")

    obs_terms = uw.observation_manager.active_terms["policy"]
    obs_dims = [int(d[0]) for d in uw.observation_manager.group_obs_term_dim["policy"]]
    term = uw.action_manager.get_term("joint_pos")
    robot = uw.scene["robot"]
    ids = list(term._joint_ids) if not isinstance(term._joint_ids, slice) else list(range(robot.num_joints))
    names = [robot.joint_names[i] for i in ids]
    scale = term._scale[0].tolist() if torch.is_tensor(term._scale) else [float(term._scale)] * len(ids)
    # the nominal pose: every env's default carries the +-0.01 rad calibration randomization
    nominal = getattr(robot.data, "default_joint_pos_nominal", robot.data.default_joint_pos[0])
    default = nominal[ids].tolist()
    info = {
        "what": "V3 dance policy (BeyondMimic motion tracking, without state estimation)",
        "task": "Isaac-Dance-R1-v3",
        "checkpoint": str(Path(args_cli.checkpoint).resolve().relative_to(_ROOT)),
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "control": {"sim_dt": env_cfg.sim.dt, "decimation": env_cfg.decimation,
                    "control_dt": env_cfg.sim.dt * env_cfg.decimation},
        "observation": {"dim": int(sum(obs_dims)), "history_length": 1,
                        "terms": [{"name": n, "width": w} for n, w in zip(obs_terms, obs_dims)],
                        "normalizer": "folded into policy.pt / policy.onnx -- feed raw values",
                        "built_by": "the runner, v3/ros2_ws/src/r1_policy_runner dance_obs.hpp "
                                    "(reference v3/dance/dance_obs.py)"},
        "action": {"dim": len(ids), "joint_names": names, "scale": scale, "default_joint_pos": default,
                   "gains": "assets/r1/r1.py, as v2 (the bridge's joint_map.hpp sends one set)"},
        "reference": {"file": "dance_ref.bin", "frames": ref.frames, "fps": ref.fps,
                      "bars_s": [b[1] for b in routine.BARS]},
        "jit_vs_checkpoint_max_abs": worst,
        "sha256": {f: sha256(out / f) for f in ("policy.pt", "policy.onnx", "dance_ref.bin")},
    }
    (out / "interface.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: info[k] for k in ("checkpoint", "observation", "jit_vs_checkpoint_max_abs")},
                     indent=1), flush=True)
    v2 = json.loads((_ROOT / "models" / "v2_drift_ft" / "interface" / "policy_interface.json").read_text())
    same = (names == v2["action"]["joint_names"]
            and all(abs(a - b) < 1e-6 for a, b in zip(scale, v2["action"]["scale"]))
            and all(abs(a - b) < 1e-6 for a, b in zip(default, v2["action"]["default_pos"])))
    print("action joints, scales and default pose are v2's: {}".format("yes" if same else "NO"), flush=True)
    env.close()
    if not same:
        raise SystemExit("the dance policy's action term differs from v2's: the runner and the "
                         "bridge cannot drive both")


if __name__ == "__main__":
    main()
    simulation_app.close()
