import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


class SphericalCPGPolicy(ActorCriticPolicy):
    """
    CPG-based ActorCriticPolicy for Stable-Baselines3.

    Outputs a 12-D action in [-1, 1] (the environment's UrchinRoll.step
    then maps this to [0, 0.025] m).

    Design
    ------
    observation -> MLP -> {direction, freq, amp, k, bias, phase[, fourier]}
              -> spherical CPG -> 12-D action mean in [-1, 1]
              -> DiagGaussian(action_mean, exp(log_std)) -> sample
              -> clamp to [-1, 1]

    Phase is predicted from the observation (not carried as mutable state),
    so evaluate_actions() is deterministic w.r.t. the stored (obs, action)
    pairs -- required for PPO's ratio to be meaningful.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        # CPG configuration
        n_spines: int = 12,
        spine_directions: Optional[np.ndarray] = None,
        # Ranges (in env's action space; -1..1)
        min_frequency: float = 0.5,
        max_frequency: float = 5.0,
        min_amplitude: float = 0.05,
        max_amplitude: float = 1.0,
        min_wave_number: float = 0.0,
        max_wave_number: float = 8.0,
        fourier_harmonics: int = 0,
        hidden_dim: int = 256,
        **kwargs,
    ):
        if not isinstance(action_space, spaces.Box):
            raise ValueError("SphericalCPGPolicy requires a Box action space.")
        if action_space.shape != (n_spines,):
            raise ValueError(
                f"SphericalCPGPolicy expects action space shape ({n_spines},), "
                f"got {action_space.shape}"
            )

        super().__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )

        # --- Bookkeeping ------------------------------------------------
        self.n_spines = n_spines
        self.actuator_min = float(action_space.low.min())
        self.actuator_max = float(action_space.high.max())

        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.min_wave_number = min_wave_number
        self.max_wave_number = max_wave_number
        self.fourier_harmonics = fourier_harmonics

        # --- Spine geometry --------------------------------------------
        if spine_directions is None:
            angles = np.linspace(0.0, 2.0 * np.pi, n_spines, endpoint=False)
            spine_directions = np.stack(
                [np.cos(angles), np.sin(angles), np.zeros(n_spines)], axis=1
            )
        spine_directions = np.asarray(spine_directions, dtype=np.float32)
        if spine_directions.shape != (n_spines, 3):
            raise ValueError(
                f"spine_directions must have shape ({n_spines}, 3), "
                f"got {spine_directions.shape}"
            )
        norms = np.linalg.norm(spine_directions, axis=1, keepdims=True)
        spine_directions = spine_directions / np.maximum(norms, 1e-8)
        self.register_buffer(
            "spine_directions",
            torch.tensor(spine_directions, dtype=torch.float32),
        )

        # --- Feature MLP ------------------------------------------------
        obs_dim = int(np.prod(observation_space.shape))
        self.cpg_features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

        # --- CPG head ---------------------------------------------------
        fourier_dim = 1 + 2 * fourier_harmonics
        self.cpg_output_dim = 3 + 1 + 1 + 1 + 1 + 1 + fourier_dim
        self.cpg_head = nn.Linear(hidden_dim, self.cpg_output_dim)
        # Start close to zero-mean policy:
        nn.init.zeros_(self.cpg_head.weight)
        nn.init.zeros_(self.cpg_head.bias)

        # --- Value net (overrides SB3's default MlpExtractor-based one) -
        self.value_net = nn.Linear(hidden_dim, 1)

        # --- Action noise ----------------------------------------------
        # SB3's DiagGaussianDistribution wants a log_std parameter of
        # shape (action_dim,). This is what PPO trains (via the entropy
        # bonus / log-prob gradients).
        self.log_std = nn.Parameter(torch.zeros(n_spines))

        # SB3 expects this attribute to exist; not actually used by our
        # forward, but harmless.
        self.latent_dim_pi = hidden_dim

    # ================================================================
    # Feature extraction
    # ================================================================
    def _get_cpg_features(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.cpg_features(obs)

    def extract_features(self, obs, features_extractor=None):
        # Called by SB3 during _build() to size the default nets; our
        # forward() bypasses those nets anyway.
        return self._get_cpg_features(obs)

    # ================================================================
    # CPG
    # ================================================================
    def _generate_cpg_action(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.cpg_head(features)
        idx = 0

        # Direction (unit vector in 3-D)
        direction_raw = raw[:, idx:idx + 3]; idx += 3
        direction = torch.tanh(direction_raw)
        direction = direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8)

        # Frequency
        f = torch.sigmoid(raw[:, idx:idx + 1]); idx += 1
        frequency = self.min_frequency + f * (self.max_frequency - self.min_frequency)

        # Amplitude
        a = torch.sigmoid(raw[:, idx:idx + 1]); idx += 1
        amplitude = self.min_amplitude + a * (self.max_amplitude - self.min_amplitude)

        # Wave number
        k = torch.sigmoid(raw[:, idx:idx + 1]); idx += 1
        wave_number = self.min_wave_number + k * (self.max_wave_number - self.min_wave_number)

        # Bias — mapped to [-1, 1] in the env's action space.
        b = torch.sigmoid(raw[:, idx:idx + 1]); idx += 1
        bias = self.actuator_min + b * (self.actuator_max - self.actuator_min)

        # Phase in [0, 2π)
        phase = torch.sigmoid(raw[:, idx:idx + 1]) * 2.0 * math.pi; idx += 1

        # Fourier coefficients (optional)
        if self.fourier_harmonics > 0:
            fourier = torch.tanh(raw[:, idx:idx + 1 + 2 * self.fourier_harmonics])
        else:
            fourier = None

        # Spatial phase
        # direction: [B, 1, 3]; spine_directions: [1, N, 3]
        proj = torch.sum(
            direction.unsqueeze(1) * self.spine_directions.unsqueeze(0),
            dim=-1,
        )                                    # [B, N]
        spatial_phase = phase + wave_number * proj  # broadcast [B,1]+[B,N]

        # Waveform
        if self.fourier_harmonics == 0:
            waveform = torch.sin(spatial_phase)
        else:
            waveform = fourier[:, 0:1].expand(-1, self.n_spines)
            c = 1
            for h in range(1, self.fourier_harmonics + 1):
                s = fourier[:, c:c + 1]; cc = fourier[:, c + 1:c + 2]; c += 2
                waveform = waveform + s * torch.sin(h * spatial_phase)
                waveform = waveform + cc * torch.cos(h * spatial_phase)

        # Bias + amplitude*waveform
        action = bias + amplitude * waveform

        # Clamp to action space [-1, 1]
        return torch.clamp(action, self.actuator_min, self.actuator_max)

    # ================================================================
    # Distribution
    # ================================================================
    def _make_dist(self, action_mean: torch.Tensor) -> DiagGaussianDistribution:
        # NOTE: no SquashedDiagGaussian -- we clamp after sampling,
        #       so the log-prob of the Gaussian is the correct
        #       (unmodified) one. The env also clamps internally.
        dist = DiagGaussianDistribution(self.n_spines)
        dist.proba_distribution(
            action_mean,
            self.log_std.expand_as(action_mean),
        )
        return dist

        # ================================================================
    # SB3 API
    # ================================================================
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self._get_cpg_features(obs)
        action_mean = self._generate_cpg_action(features)
        dist = self._make_dist(action_mean)

        if deterministic:
            actions = action_mean
        else:
            actions = dist.get_actions(deterministic=False)

        actions = torch.clamp(actions, self.actuator_min, self.actuator_max)
        log_prob = dist.log_prob(actions)
        values = self.value_net(features)
        return actions, values, log_prob

    def get_distribution(self, obs):
        features = self._get_cpg_features(obs)
        action_mean = self._generate_cpg_action(features)
        return self._make_dist(action_mean)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self._get_cpg_features(obs)
        action_mean = self._generate_cpg_action(features)
        dist = self._make_dist(action_mean)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(features)
        return values, log_prob, entropy

    # --- ADD THESE TWO OVERRIDES -----------------------------------
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self._get_cpg_features(obs)
        return self.value_net(features)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        features = self._get_cpg_features(observation)
        action_mean = self._generate_cpg_action(features)
        if deterministic:
            return action_mean
        return self._make_dist(action_mean).get_actions(deterministic=False)