
from gymnasium.envs.registration import register

register(
    id="Urchin",
    entry_point="tactip_mujoco_gym.envs.Urchin.main",
)
