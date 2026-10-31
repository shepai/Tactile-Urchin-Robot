import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
import mujoco 

def spine_directions(model, n=12):
    """Unit vector of each spine in the core's frame (hub bodies are children of 'core')."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"hub{i:02d}") for i in range(n)]
    p = np.array([model.body_pos[i] for i in ids])
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def taxel_directions(model, data, core="core"):
    """Unit vector from the core centre to every taxel site (pad??_t??), in the core's frame."""
    mujoco.mj_forward(model, data)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, core)
    R, c = data.xmat[cid].reshape(3, 3), data.xpos[cid]
    ids = [i for i in range(model.nsite)
           if (nm := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)) and nm.startswith("pad")]
    d = np.array([R.T @ (data.site_xpos[i] - c) for i in ids])
    return d / np.linalg.norm(d, axis=1, keepdims=True)



class TaxelFlowPolicy(ActorCriticPolicy):
    """obs -> push point on the sphere -> nearest taxel = anchor (hard pick)
       -> only spines within `reach` of the anchor extend, at high amplitude
       -> all other spines fully retracted.

    Expects the LAST 6 obs entries to be: command in body frame (3), down-direction in body frame (3)."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 spine_directions=None, anchor_directions=None,
                 hidden_dim=50, log_std_init=-1.5,        # small noise -> no random flicker
                 elev_deg=50.0,                            # prior: push point 50 deg below horizontal, behind motion
                 sel_sharp=60.0,                           # sharpness of the nearest-taxel choice
                 min_reach_deg=35.0, max_reach_deg=70.0,   # spines within this angle of the anchor fire
                 edge_deg=8.0,                             # width of the cut-off edge
                 min_amp=0.5,                              # pushes are at least 80% of full stroke
                 max_active=3,
                 max_ext=1.0,
                 **kwargs):
        if spine_directions is None or anchor_directions is None:
            raise ValueError("Pass spine_directions=... and anchor_directions=... via policy_kwargs.")
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        unit = lambda x: torch.tensor(
    np.asarray(x, np.float64) / np.linalg.norm(x, axis=1, keepdims=True),
    dtype=torch.float32)
        self.register_buffer("S", unit(spine_directions))      # [N,3]
        self.register_buffer("A", unit(anchor_directions))     # [K,3]
        self.n, self.K = self.S.shape[0], self.A.shape[0]
        self.elev = math.radians(elev_deg)
        self.sel_sharp = sel_sharp
        self.reach_lo, self.reach_hi = math.radians(min_reach_deg), math.radians(max_reach_deg)
        self.edge = math.radians(edge_deg)
        self.min_amp = min_amp
        self.max_active, self.max_ext = max_active, max_ext
        obs_dim = int(np.prod(observation_space.shape))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.vtrunk = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
                            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, 5)                   # push-point offset (3), reach, amplitude
        nn.init.normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)
        self.value_net = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((self.n,), float(log_std_init)))
        self.latent_dim_pi = hidden_dim

        # layers above were created after SB3 built its optimizer -> rebuild it
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    # ---- core ----
    def _prep(self, obs):
        obs = obs.float()
        return obs.unsqueeze(0) if obs.dim() == 1 else obs.flatten(1)

    def _action_mean(self, o, feat):
        h = self.head(feat)
        cmd, down = o[:, -6:-3], o[:, -3:]                     # body-frame command, body-frame "down"
        spd = cmd.norm(dim=-1, keepdim=True)
        d = cmd / (spd + 1e-6)

        # push point: behind the motion and below, plus a learned correction
        p0 = -math.cos(self.elev) * d + math.sin(self.elev) * down
        p = F.normalize(p0 + 0.5 * torch.tanh(h[:, :3]), dim=-1)

        # pick the nearest taxel: hard in the forward pass, soft gradient backward
        p = F.normalize(p0 + 0.5 * torch.tanh(h[:, :3]), dim=-1)
        cosang = p @ self.S.T                                   # [B,N] spine closeness to the push point

        reach = self.reach_lo + torch.sigmoid(h[:, 3:4]) * (self.reach_hi - self.reach_lo)
        amp = self.min_amp + (1 - self.min_amp) * torch.sigmoid(h[:, 4:5])
        edge_w = torch.cos(reach - self.edge) - torch.cos(reach)
        within = ((cosang - torch.cos(reach)) / edge_w).clamp(0, 1)      # smooth cut-off, no acos

        top = cosang.topk(self.max_active + 1, dim=-1).values
        thr = 0.5 * (top[:, -2:-1] + top[:, -1:])                         # midway between k-th and (k+1)-th
        near = torch.sigmoid((cosang - thr) / 0.02)                       # smooth top-k

        ext = self.max_ext * amp * within * near
        ext = ext * (spd > 0.05).float()
        return 2 * ext - 1
    def _dist(self, mean):
        d = DiagGaussianDistribution(self.n)
        d.proba_distribution(mean, self.log_std.expand_as(mean))
        return d

    # ---- SB3 API ----
    def forward(self, obs, deterministic=False):
        o = self._prep(obs); f = self.trunk(o)
        mean = self._action_mean(o, f); dist = self._dist(mean)
        a = (mean if deterministic else dist.get_actions(deterministic=False)).clamp(-1, 1)
        return a, self.value_net(self.vtrunk(o)), dist.log_prob(a)

    def get_distribution(self, obs):
        o = self._prep(obs)
        return self._dist(self._action_mean(o, self.trunk(o)))

    def evaluate_actions(self, obs, actions):
        o = self._prep(obs); f = self.trunk(o)
        dist = self._dist(self._action_mean(o, f))
        return self.value_net(f), dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        return self.value_net(self.trunk(self._prep(obs)))

    def _predict(self, observation, deterministic=False):
        o = self._prep(observation)
        mean = self._action_mean(o, self.trunk(o))
        return mean if deterministic else self._dist(mean).get_actions(deterministic=False)