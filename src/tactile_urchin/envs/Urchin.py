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
STAGES = [
    dict(angles=[0],                    speeds=(0.6, 0.6), p_zero=0.0),
    dict(angles=[0],                    speeds=(0.3, 1.0), p_zero=0.0),
    dict(angles=[0, 180],               speeds=(0.3, 1.0), p_zero=0.0),
    dict(angles=[0, 90, 180, 270],      speeds=(0.3, 1.0), p_zero=0.0),
    dict(angles=list(range(0, 360, 45)), speeds=(0.3, 1.0), p_zero=0.0),
    dict(angles=None,                   speeds=(0.2, 1.0), p_zero=0.0),   # None = any angle
    dict(angles=None,                   speeds=(0.2, 1.0), p_zero=0.2),
]
REPLAY = 0.25     # fraction of commands drawn from EARLIER stages

class UrchinRoll(TactileUrchin):
    CMD_HOLD_S = 4.0
    EMA_TAU_S  = 0.5

    def __init__(self, xml_subpath=("assets", "icosabot.xml"), max_speed=0.2, **kw):
        super().__init__(xml_subpath, **kw)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.trace.size + 30 + 3 + 3,), np.float32)
        self.max_speed = max_speed
        self.cmd = np.zeros(2)
        self.hi = 0.025
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)
        self.frame_skip = 5
        self.dt = 0.002
        self.policy_dt = self.dt * self.frame_skip
        self._v_ema = np.zeros(2)
        self._a = np.zeros(self.action_dim)
        self._t_cmd = 0.0
        self.stage, self._cmd_stage = 0, 0
        self._seg_prog, self._seg_n = 0.0, 0

    # ---- curriculum ----
    def set_stage(self, k):
        self.stage = int(np.clip(k, 0, len(STAGES) - 1))

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._sample_cmd()
        self._t_cmd, self._seg_prog, self._seg_n = 0.0, 0.0, 0
        self._v_ema[:] = 0.0
        self._a = np.zeros(self.action_dim)
        return self._get_obs(), {}

    def step(self, action):
        self._a = (np.asarray(action) + 1) / 2
        obs, r, term, trunc, info = super().step(self._a * self.hi)
        self._t_cmd += self.policy_dt
        if self._t_cmd >= self.CMD_HOLD_S:                       # end of a command segment
            if self._seg_n > 0:
                info["seg_progress"] = self._seg_prog / self._seg_n
                info["cmd_stage"] = self._cmd_stage
            self._sample_cmd()
            self._t_cmd, self._seg_prog, self._seg_n = 0.0, 0.0, 0
            obs = np.asarray(obs, np.float32).copy()
            obs[-6:-3] = self._cmd_body()                        # show the new command immediately
        return obs, r, term, trunc, info

    def _get_obs(self):
        obs = super()._get_obs()
        R = self.data.xmat[1].reshape(3, 3)
        grav_body = R.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([obs, self._cmd_body(), grav_body]).astype(np.float32)

    def _sample_cmd(self):
        k = self.stage
        if k > 0 and self.np_random.random() < REPLAY:
            k = int(self.np_random.integers(0, k))
        self._cmd_stage = k
        cfg = STAGES[k]
        if self.np_random.random() < cfg["p_zero"]:
            self.cmd = np.zeros(2)
            return
        ang = (np.radians(self.np_random.choice(cfg["angles"])) if cfg["angles"] is not None
               else self.np_random.uniform(0, 2 * np.pi))
        self.cmd = np.array([np.cos(ang), np.sin(ang)])          # unit direction, no speed

    def _cmd_body(self):
        R = self.data.xmat[1].reshape(3, 3)
        return R.T @ np.array([*self.cmd, 0.0])                  # unit length (or zero), no scaling

    def _reward(self):
        alpha = 1.0 - np.exp(-self.policy_dt / self.EMA_TAU_S)
        self._v_ema += alpha * (self.data.qvel[:2] - self._v_ema)
        v = self._v_ema
        ref = self.max_speed                  # now just a scale: "this fast along the direction = full reward"
        sigma = 0.3 * ref

        if np.linalg.norm(self.cmd) > 1e-6:
            along = v @ self.cmd
            lateral = v - along * self.cmd
            progress = np.clip(along / ref, -1.0, 1.0)           # more speed = more reward, capped at ref
            lat_pen = np.clip((lateral @ lateral) / sigma**2, 0, 1)
            reward = progress - 0.5 * lat_pen
            act_w = 0.02
            self._seg_prog += progress
            self._seg_n += 1
        else:
            reward = np.exp(-(v @ v) / sigma**2)                 # stand still = good
            act_w = 0.1

        return float(reward - act_w * np.mean(self._a ** 2))