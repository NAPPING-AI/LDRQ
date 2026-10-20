import gymnasium as gym

from . import flat_env_cfg

gym.register(
    id="Isaac-Velocity-Flat-R1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:R1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:R1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-R1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:R1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:R1FlatPPORunnerCfg",
    },
)

# v2: backwards, sideways, faster turns, no drift on a zero command (flat_v2_env_cfg.py).
gym.register(
    id="Isaac-Velocity-Flat-R1-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_v2_env_cfg:R1FlatV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:R1FlatV2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-R1-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_v2_env_cfg:R1FlatV2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:R1FlatV2PPORunnerCfg",
    },
)
