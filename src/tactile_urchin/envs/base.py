import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mujoco import MjModel, MjData
from mujoco import mj_step, mj_resetData, mj_forward
from importlib.resources import files
from mujoco import renderer
from dm_control import mujoco

EVENT_DTYPE = np.dtype([
    ("t", "f8"),         # sim time
    ("pad", "u1"),       # pad index (0-19)
    ("taxel", "u1"),     # taxel index within pad (0-9)
    ("polarity", "i1"),  # +1 pressure increased, -1 decreased
    ("value", "f4"),     # raw touch reading (N) when event fired
])
 

 
class TactileUrchin(gym.Env):
    """
    Shared MuJoCo + Gym logic for all tactile environments.
    """

    def __init__(self, xml_subpath, obs_dim, action_dim, threshold=0.02, refractory=0.0,
                 log_domain=False, eps=1e-3, capacity=65536):
        super().__init__()

        self.xml_path = files("tactip_urchin").joinpath(*xml_subpath)
        self.physics = mujoco.Physics.from_xml_path(str(self.xml_path))
        self.model = MjModel.from_xml_path(str(self.xml_path))
        self.data = MjData(self.model)

        self.action_space = spaces.Box(
            low=-0.02, high=0.02, shape=(action_dim,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.renderer = renderer.Renderer(self.model,height=128, width=128)
        self.step_count = 0
        self.max_steps = 200
        self.num_actuators = self.model.nu
        self.threshold = float(threshold)   # N (or log units if log_domain)
        self.refractory = float(refractory)  # min seconds between events per taxel
        self.log_domain = log_domain
        self.eps = eps
 
        # --- discover sensors by name -------------------------------------
        found = []
        for sid in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            m = name and self._NAME_RE.fullmatch(name)
            if m:
                assert self.model.sensor_dim[sid] == 1
                found.append((int(m.group(1)), int(m.group(2)), self.model.sensor_adr[sid]))
        found.sort()
        self.pad_ids = np.array([f[0] for f in found], dtype=np.uint8)
        self.taxel_ids = np.array([f[1] for f in found], dtype=np.uint8)
        self.adr = np.array([f[2] for f in found], dtype=int)
        self.n_pads = int(self.pad_ids.max()) + 1
        self.n_taxels = int(self.taxel_ids.max()) + 1
        self.n = len(found)
 
        # --- state ---------------------------------------------------------
        self.values = np.zeros(self.n)        # latest raw readings
        self._ref = np.zeros(self.n)
        self._last_t = np.full(self.n, -np.inf)
        self._buf = np.empty(capacity, dtype=EVENT_DTYPE)
        self._count = 0
 
    # ---------------------------------------------------------------------
    def _tf(self, x):
        return np.log(x + self.eps) if self.log_domain else x
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self._count = 0
        self._last_t[:] = -np.inf
        if self.data is not None:
            self.values = self.data.sensordata[self.adr].copy()
        self._ref = self._tf(self.values)
        self.step_count = 0
        return self._get_obs(), {}
    def update(self, data):
        """Poll sensors; returns the array of new events (possibly empty)."""
        t = data.time
        self.values = data.sensordata[self.adr]
        x = self._tf(self.values)
        diff = x - self._ref
        fire = np.abs(diff) >= self.threshold
        if self.refractory > 0:
            fire &= (t - self._last_t) >= self.refractory
        idx = np.flatnonzero(fire)
        if idx.size == 0:
            return self._buf[:0]
 
        self._grow(idx.size)
        out = self._buf[self._count:self._count + idx.size]
        out["t"] = t
        out["pad"] = self.pad_ids[idx]
        out["taxel"] = self.taxel_ids[idx]
        out["polarity"] = np.sign(diff[idx]).astype(np.int8)
        out["value"] = self.values[idx]
        self._count += idx.size
 
        self._ref[idx] = x[idx]
        self._last_t[idx] = t
        return out
 
    def _grow(self, k):
        if self._count + k > len(self._buf):
            new = np.empty(max(2 * len(self._buf), self._count + k), dtype=EVENT_DTYPE)
            new[:self._count] = self._buf[:self._count]
            self._buf = new
 
    # --- accessors -------------------------------------------------------
    @property
    def events(self):
        return self._buf[:self._count]
 
    def window(self, t0, t1):
        ev = self.events
        lo, hi = np.searchsorted(ev["t"], [t0, t1])
        return ev[lo:hi]
 
    def frame(self, t0, t1, signed=True):
        """(n_pads, n_taxels) event-count image over [t0, t1)."""
        ev = self.window(t0, t1)
        img = np.zeros((self.n_pads, self.n_taxels))
        w = ev["polarity"].astype(float) if signed else 1.0
        np.add.at(img, (ev["pad"], ev["taxel"]), w)
        return img
 
    def pressure_map(self):
        """Latest raw readings as (n_pads, n_taxels)."""
        img = np.zeros((self.n_pads, self.n_taxels))
        img[self.pad_ids, self.taxel_ids] = self.values
        return img
    def step(self, action):
        mj_forward(self.model, self.data)
        self.data.ctrl[:-1] = action[:-1]
        mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward()
        terminated = self._done()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)

        # events fired during the last control step -> signed count image
        ev = self.tactile.events[self._ev_start:]
        frame = np.zeros_like(self.trace)
        np.add.at(frame, (ev["pad"], ev["taxel"]), ev["polarity"])

        # decaying trace gives the policy a short memory of recent contact
        self.trace = self.trace_decay * self.trace + frame

        acc = self.data.sensor("imu_acc").data
        gyro = self.data.sensor("imu_gyro").data
        slide_pos = self.data.qpos[7:19]    # 7 = freejoint qpos size
        slide_vel = self.data.qvel[6:18]    # 6 = freejoint qvel size

        return np.concatenate([
            self.trace.ravel(), acc, gyro, slide_pos, slide_vel
        ]).astype(np.float32)

    def _reward(self):
        raise NotImplementedError

    def _done(self):
        return False
   