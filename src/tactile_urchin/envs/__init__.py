
from gymnasium.envs.registration import register

register(
    id="Urchin",
    entry_point="tactile_urchin.envs.Urchin:Plane",
)
