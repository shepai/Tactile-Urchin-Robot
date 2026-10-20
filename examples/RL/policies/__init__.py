import numpy as np, torch, torch.nn as nn, mujoco
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


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
    """obs -> MLP -> {taxel logits, sharpness, amplitude, base}
       -> softmax picks a taxel -> spines near it extend (smooth falloff)
       -> 12-D action mean in [-1, 1] -> Gaussian."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 spine_directions=None, anchor_directions=None,
                 hidden_dim=256, log_std_init=-1.0,
                 min_kappa=2.0, max_kappa=30.0, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        unit = lambda x: torch.tensor(np.asarray(x, np.float32) /
                                      np.linalg.norm(x, axis=1, keepdims=True))
        self.register_buffer("S", unit(spine_directions))     # [N,3] spines
        self.register_buffer("A", unit(anchor_directions))    # [K,3] taxels
        self.n, self.K = self.S.shape[0], self.A.shape[0]
        self.min_kappa, self.max_kappa = min_kappa, max_kappa

        obs_dim = int(np.prod(observation_space.shape))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, self.K + 3)         # K logits + kappa + amp + base
        nn.init.normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)
        self.value_net = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((self.n,), float(log_std_init)))
        self.latent_dim_pi = hidden_dim

        # SB3 built its optimizer inside super().__init__, before the layers above existed.
        # Rebuild it or none of them get trained.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    # ---- features / mean / dist ----
    def _feat(self, obs):
        obs = obs.float()
        obs = obs.unsqueeze(0) if obs.dim() == 1 else obs.flatten(1)
        return self.trunk(obs)

    def extract_features(self, obs, features_extractor=None):
        return self._feat(obs)

    def _action_mean(self, feat):
        h = self.head(feat)
        logits = h[:, :self.K]
        kappa = self.min_kappa + torch.sigmoid(h[:, self.K]) * (self.max_kappa - self.min_kappa)
        amp   = torch.sigmoid(h[:, self.K + 1])[:, None]
        base  = 0.3 * torch.sigmoid(h[:, self.K + 2])[:, None]

        w = torch.softmax(logits, dim=-1)                                  # [B,K] which taxel
        cos = self.S @ self.A.T                                            # [N,K] spine-taxel closeness
        bump = torch.exp(kappa[:, None, None] * (cos[None] - 1.0))         # [B,N,K]
        flow = (bump * w[:, None, :]).sum(-1)                              # [B,N] 1 near the chosen taxel
        return (2.0 * (base + amp * flow) - 1.0).clamp(-1.0, 1.0)

    def _dist(self, mean):
        d = DiagGaussianDistribution(self.n)
        d.proba_distribution(mean, self.log_std.expand_as(mean))
        return d

    # ---- SB3 API ----
    def forward(self, obs, deterministic=False):
        f = self._feat(obs); mean = self._action_mean(f); dist = self._dist(mean)
        a = mean if deterministic else dist.get_actions(deterministic=False)
        a = a.clamp(-1.0, 1.0)
        return a, self.value_net(f), dist.log_prob(a)

    def get_distribution(self, obs):
        return self._dist(self._action_mean(self._feat(obs)))

    def evaluate_actions(self, obs, actions):
        f = self._feat(obs); dist = self._dist(self._action_mean(f))
        return self.value_net(f), dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        return self.value_net(self._feat(obs))

    def _predict(self, observation, deterministic=False):
        mean = self._action_mean(self._feat(observation))
        return mean if deterministic else self._dist(mean).get_actions(deterministic=False)