from .base import TactileUrchin
import numpy as np
from gymnasium import spaces

class Plane(TactileUrchin):
    def __init__(self, xml_subpath=("assets", "icosabot.xml"),
                 obs_dim=230,
                 action_dim=12,
                 **kwargs):
        super().__init__(
            xml_subpath=xml_subpath,
            obs_dim=obs_dim,
            action_dim=action_dim,
            **kwargs,   # frame_skip, threshold, refractory, etc.
        )

    def _reward(self):
        # replace with your task reward
        return 0.0

    def _done(self):
        return False

P_ZERO       = 0.2    # fraction of commands that are "stand still"
CMD_HOLD_S   = 4.0    # resample the command every N seconds inside an episode
EMA_TAU_S    = 0.5    # velocity smoothing time-constant
class UrchinRoll(TactileUrchin):
    """Follow a random velocity command (world-frame direction + speed)."""
    P_ZERO       = 0.2    # fraction of commands that are "stand still"
    CMD_HOLD_S   = 4.0    # resample the command every N seconds inside an episode
    EMA_TAU_S    = 0.5    # velocity smoothing time-constant
    def __init__(self, xml_subpath=("assets", "icosabot.xml"), max_speed=0.2, **kw):   # <- your XML path
        super().__init__(xml_subpath, **kw)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.trace.size + 30 + 3,), np.float32)
        self.max_speed = max_speed
        self.cmd = np.zeros(2)
        self.hi = 0.025
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)  # SB3-friendly
        self.frame_skip = 5
        self.dt = 0.002  # physics timestep
        self.policy_dt = self.dt * self.frame_skip  # 0.05 s
        self._v_ema = np.zeros(2, dtype=np.float64)
    def _sample_cmd(self):
        if self.np_random.random() < P_ZERO:
            self.cmd = np.zeros(2)
        else:
            ang = self.np_random.uniform(0, 2 * np.pi)
            spd = self.np_random.uniform(0.2, 1.0) * self.max_speed
            self.cmd = spd * np.array([np.cos(ang), np.sin(ang)])
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        ang = self.np_random.uniform(0, 2 * np.pi)
        spd = 0.0 if self.np_random.random() < 0.1 else self.np_random.uniform(0.2, 1.0) * self.max_speed
        self.cmd = spd * np.array([np.cos(ang), np.sin(ang)])
        return self._get_obs(), {}
 
    def step(self, action):
        self._a = (np.asarray(action) + 1) / 2             # [-1,1] -> [0,1]
        return super().step(self._a * self.hi)             # -> [0, 0.025] m
 
    def _get_obs(self):
        obs = super()._get_obs()
        R = self.data.xmat[1].reshape(3, 3)                # root body orientation
        cmd_body = R.T @ np.array([*self.cmd, 0.0]) / self.max_speed   # command in body frame
        return np.append(obs, cmd_body).astype(np.float32)
 
    
    def _reward(self):
        alpha = 1.0 - np.exp(-self.dt / EMA_TAU_S)
        self._v_ema += alpha * (self.data.qvel[:2] - self._v_ema)
        v = self._v_ema

        sigma = 0.3 * self.max_speed
        err = v - self.cmd
        track = np.exp(-(err @ err) / sigma**2)

        spd_cmd = np.linalg.norm(self.cmd)
        if spd_cmd > 1e-6:
            c_hat = self.cmd / spd_cmd
            along = v @ c_hat
            lateral = v - along * c_hat
            progress = np.clip(along / spd_cmd, -1.0, 1.0)         # gradient even when v = 0
            lat_pen = np.clip((lateral @ lateral) / sigma**2, 0, 1)
            shaped = 0.5 * progress - 0.25 * lat_pen
            act_w = 0.02
        else:
            shaped = 0.0
            act_w = 0.1                                             # extra cost for fidgeting at rest

        return float(track + shaped - act_w * np.mean(self._a ** 2))
 