"""
Urchin environments.

    Plane      - stub task (unchanged)
    UrchinRoll - roll in a commanded ground-plane direction, joystick-style

SPARSE ACTUATION
----------------
The policy does NOT output 12 spine commands. It outputs 5 gait numbers in [-1, 1]:

    a[0:3]  nudge to the push point on the sphere (body frame)
    a[3]    reach: angular radius of the patch of spines allowed to fire
    a[4]    stroke amplitude

SparsePush turns those into 12 spine extensions. At most `max_active` spines are out
at once and the rest are fully retracted. With a = 0 you get the hand-built prior:
push on the ground behind the direction of travel and below the body.

Because the mapping lives in the env, PPO explores in the 5-number gait space (where
it makes sense), and the "never all spines at once" rule also holds when you drive
the robot with a joystick.

ASSUMPTION: actuator i, slide joint i and body hub{i:02d} all belong to the same spine.
"""
import numpy as np
import mujoco
from gymnasium import spaces

from .base import TactileUrchin


# =============================================================== geometry ===
def spine_directions(model, n=12):
    """Unit vector of each spine in the core's frame (hubXX bodies are children of 'core')."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"hub{i:02d}") for i in range(n)]
    if min(ids) < 0:
        raise ValueError("Expected bodies hub00..hub11 in the model.")
    p = np.array([model.body_pos[i] for i in ids], dtype=float)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


class SparsePush:
    """5 gait numbers -> 12 spine extensions in [0, 1], at most `max_active` non-zero."""
    LATENT = 5

    def __init__(self, spines, elev_deg=50.0, nudge=0.5,
                 min_reach_deg=40.0, max_reach_deg=75.0, edge_deg=8.0,
                 min_amp=0.5, max_active=3):
        self.S = np.asarray(spines, float)
        self.elev = np.radians(elev_deg)
        self.nudge = nudge
        self.lo, self.hi = np.radians(min_reach_deg), np.radians(max_reach_deg)
        self.edge = np.radians(edge_deg)
        self.min_amp = min_amp
        self.max_active = max_active

    def __call__(self, a, cmd_body, down_body):
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        n = np.linalg.norm(cmd_body)
        if n < 1e-6:                                    # "hold" command: everything retracted
            return np.zeros(len(self.S))
        d = cmd_body / n

        # push point: behind the motion and below, plus the learned nudge
        p = -np.cos(self.elev) * d + np.sin(self.elev) * down_body + self.nudge * a[:3]
        p /= np.linalg.norm(p) + 1e-9

        reach = self.lo + 0.5 * (a[3] + 1.0) * (self.hi - self.lo)
        amp = self.min_amp + (1.0 - self.min_amp) * 0.5 * (a[4] + 1.0)

        cos = self.S @ p
        edge_w = np.cos(reach - self.edge) - np.cos(reach)
        w = np.clip((cos - np.cos(reach)) / edge_w, 0.0, 1.0)   # soft angular cut-off
        if self.max_active < len(w):
            w[np.argsort(cos)[:-self.max_active]] = 0.0         # hard top-k: sparse
        if w.max() == 0.0:
            w[np.argmax(cos)] = 1.0                             # always push with something
        return amp * w


# ================================================================== tasks ===
class Plane(TactileUrchin):
    def __init__(self, xml_subpath=("assets", "icosabot.xml"),
                 obs_dim=230, action_dim=12, **kwargs):
        super().__init__(xml_subpath=xml_subpath, obs_dim=obs_dim,
                         action_dim=action_dim, **kwargs)

    def _reward(self):
        return 0.0

    def _done(self):
        return False


ALL8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
STAGES = [
    [(1, 0)],                                    # 0: one direction (random yaw makes it varied in body frame)
    [(1, 0), (-1, 0)],                           # 1: + opposite
    [(1, 0), (-1, 0), (0, 1), (0, -1)],          # 2: four axes
    ALL8,                                        # 3: + diagonals
    ALL8 + [(0, 0)] * 2,                         # 4: + hold still (~20% of commands)
]
REPLAY = 0.25     # fraction of commands drawn from EARLIER stages


class UrchinRoll(TactileUrchin):
    """Command (cx, cy): roll the centre along that world direction.
       Zero command: stay put (all spines retracted)."""

    CMD_HOLD_S = 3.0     # seconds per command segment during training
    LAT_W      = 0.1     # penalty for being off the commanded line
    LAT_SCALE  = 0.1     # metres off the line that counts as "fully off"
    LATV_W     = 0.2     # penalty for smoothed sideways velocity
    HOLD_SIGMA = 0.05    # metres of drift at which the hold reward falls to ~37%
    ACT_W      = 0.02    # effort penalty on spine extension
    EMA_TAU    = 0.3     # seconds, smoothing for the sideways-velocity penalty
    SETTLE_S   = 0.3     # seconds of physics after reset so the robot lands
    OBS_DIM    = 33

    def __init__(self, xml_subpath=("assets", "icosabot.xml"), max_speed=0.05,
                 max_steps=1200, frame_skip=5, random_yaw=True, auto_cmd=True,
                 push_kwargs=None, **kw):
        super().__init__(xml_subpath, obs_dim=self.OBS_DIM, action_dim=12,
                         frame_skip=frame_skip, **kw)
        m = self.model
        self.max_steps = int(max_steps)          # base class truncates on this (was hard-coded 200)
        self.max_speed = float(max_speed)        # m/s that counts as "full progress" - measure it with check_prior()
        self.random_yaw, self.auto_cmd = random_yaw, auto_cmd
        self.policy_dt = m.opt.timestep * self.frame_skip
        self.seg_steps = max(1, round(self.CMD_HOLD_S / self.policy_dt))

        hi = m.actuator_ctrlrange[:self.action_dim, 1].copy()
        self.hi = np.where(hi > 0, hi, 0.025)    # full stroke per actuator (m)

        self.core_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "core")
        jt = m.jnt_type
        free = np.flatnonzero(jt == int(mujoco.mjtJoint.mjJNT_FREE))
        slides = np.flatnonzero(jt == int(mujoco.mjtJoint.mjJNT_SLIDE))[:self.action_dim]
        self.qp, self.qv = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
        self.slide_qadr = m.jnt_qposadr[slides]

        self.push = SparsePush(spine_directions(m, self.action_dim), **(push_kwargs or {}))
        self.action_space = spaces.Box(-1.0, 1.0, (SparsePush.LATENT,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.OBS_DIM,), np.float32)

        self.cmd, self.cmd_dir, self.perp = np.zeros(2), np.zeros(2), np.zeros(2)
        self.stage, self._cmd_stage = 0, 0
        self._latent = np.zeros(SparsePush.LATENT)
        self._ext = np.zeros(self.action_dim)
        self._p0, self._last, self._v_ema = np.zeros(2), np.zeros(2), np.zeros(2)
        self._s_prev = self._seg_score = self._path = 0.0
        self._seg_t = 0
        self._rinfo = {}

    # ---- helpers ----------------------------------------------------------
    def _R(self):
        return self.data.xmat[self.core_id].reshape(3, 3)

    def root_xy(self):
        return self.data.qpos[self.qp:self.qp + 2].copy()

    def _frames(self):
        R = self._R()
        return R.T @ np.array([*self.cmd_dir, 0.0]), R.T @ np.array([0.0, 0.0, -1.0])

    # ---- commands / curriculum -------------------------------------------
    def set_stage(self, k):
        self.stage = int(np.clip(k, 0, len(STAGES) - 1))

    def set_cmd(self, cx, cy):
        self.cmd = np.array([cx, cy], float)
        n = np.linalg.norm(self.cmd)
        self.cmd_dir = self.cmd / n if n > 1e-6 else np.zeros(2)
        self.perp = np.array([-self.cmd_dir[1], self.cmd_dir[0]])

    def command(self, cx, cy, deadzone=0.2):
        """Joystick entry point: change direction now and restart progress tracking."""
        if np.hypot(cx, cy) < deadzone:
            cx = cy = 0.0
        self.set_cmd(cx, cy)
        self._start_segment()

    def _sample_cmd(self):
        k = self.stage
        if k > 0 and self.np_random.random() < REPLAY:
            k = int(self.np_random.integers(0, k))
        self._cmd_stage = k
        choices = STAGES[k]
        self.set_cmd(*choices[int(self.np_random.integers(len(choices)))])

    def _start_segment(self):
        self._p0 = self.root_xy()
        self._last = self._p0.copy()
        self._s_prev = self._seg_score = self._path = 0.0
        self._v_ema = self.data.qvel[self.qv:self.qv + 2].copy()
        self._seg_t = 0

    def _resync_events(self):
        """Tactile event state after the settle steps (sensors unused for now, kept consistent)."""
        self._count = 0
        self._ev_start = 0
        self._last_t[:] = -np.inf
        self.values = self.data.sensordata[self.adr].copy()
        self._ref = self._tf(self.values)
        self.trace[:] = 0

    # ---- gym API ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.random_yaw:                                  # random heading, same face down
            th = self.np_random.uniform(-np.pi, np.pi)
            qz = np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])
            q = np.empty(4)
            mujoco.mju_mulQuat(q, qz, self.data.qpos[self.qp + 3:self.qp + 7].copy())
            self.data.qpos[self.qp + 3:self.qp + 7] = q
        for _ in range(int(round(self.SETTLE_S / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._resync_events()

        self._latent[:] = 0.0
        self._ext[:] = 0.0
        cmd = (options or {}).get("cmd")
        if cmd is not None:
            self.set_cmd(*cmd)
        else:
            self._sample_cmd()
        self._start_segment()
        return self._get_obs(), {}

    def step(self, action):
        self._latent = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
        self._ext = self.push(self._latent, *self._frames())          # sparse, in [0, 1]
        obs, r, term, trunc, info = super().step(self._ext * self.hi)
        info.update(self._rinfo)
        info["n_active"] = int((self._ext > 0).sum())

        self._seg_t += 1
        if self.auto_cmd and self._seg_t >= self.seg_steps:            # end of a command segment
            info["seg_progress"] = self._seg_score                     # for the curriculum callback
            info["cmd_stage"] = self._cmd_stage
            if np.any(self.cmd_dir) and self._path > 0.01:
                info["seg_straightness"] = float(np.clip(self._s_prev / self._path, -1, 1))
            self._sample_cmd()
            self._start_segment()
            obs = self._get_obs()                                      # show the new command now
        return obs, float(r), term, trunc, info

    def _get_obs(self):
        R = self._R()
        cmd_b = R.T @ np.array([*self.cmd_dir, 0.0])
        down_b = R.T @ np.array([0.0, 0.0, -1.0])
        v = self.data.qvel[self.qv:self.qv + 3]           # world-frame linear velocity
        w = self.data.qvel[self.qv + 3:self.qv + 6]       # angular velocity, body frame
        ext = self.data.qpos[self.slide_qadr] / self.hi
        rel = self.root_xy() - self._p0
        s, lat = rel @ self.cmd_dir, rel @ self.perp
        obs = np.concatenate([
            cmd_b, down_b,                                            # 6  command + gravity, body frame
            (R.T @ v) / self.max_speed,                               # 3  velocity, body frame
            w / 10.0,                                                 # 3  spin, body frame
            ext,                                                      # 12 measured spine extensions
            [s / 0.3, lat / 0.3],                                     # 2  progress / signed drift
            [v[:2] @ self.cmd_dir / self.max_speed,
             v[:2] @ self.perp / self.max_speed],                     # 2  velocity along / across command
            self._latent,                                             # 5  previous gait action
        ])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _reward(self):
        pos = self.root_xy()
        rel = pos - self._p0
        v = self.data.qvel[self.qv:self.qv + 2]
        act_pen = self.ACT_W * float(np.mean(self._ext ** 2))

        if np.any(self.cmd_dir):
            s, lat = rel @ self.cmd_dir, rel @ self.perp
            # progress telescopes: summed over a segment it is exactly the distance covered
            progress = np.clip((s - self._s_prev) / (self.max_speed * self.policy_dt), -3.0, 3.0)
            self._s_prev = s
            self._path += float(np.linalg.norm(pos - self._last))
            self._last = pos

            alpha = 1.0 - np.exp(-self.policy_dt / self.EMA_TAU)
            self._v_ema += alpha * (v - self._v_ema)
            lat_pen = self.LAT_W * min(abs(lat) / self.LAT_SCALE, 1.0)
            latv_pen = self.LATV_W * min(abs(self._v_ema @ self.perp) / self.max_speed, 1.0)

            self._seg_score = float(np.clip(s / (self.max_speed * self.CMD_HOLD_S), -1.0, 1.0))
            self._rinfo = {"rew/progress": float(progress), "rew/lat": -lat_pen,
                           "rew/lat_vel": -latv_pen, "rew/effort": -act_pen}
            return float(progress - lat_pen - latv_pen - act_pen)

        d = float(np.linalg.norm(rel))
        speed = float(np.linalg.norm(v))
        hold = np.exp(-(d / self.HOLD_SIGMA) ** 2) - 0.5 * min(speed / self.max_speed, 1.0)
        self._seg_score = float(1.0 - np.clip(d / 0.1, 0.0, 1.0))
        self._rinfo = {"rew/hold": float(hold), "rew/effort": -act_pen}
        return float(hold - act_pen)