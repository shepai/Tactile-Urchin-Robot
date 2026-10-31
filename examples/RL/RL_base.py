import os, time
from collections import defaultdict, deque

import numpy as np
import gymnasium as gym
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from recorder import *                 # TrainingVideoCallback, write_video
import tactile_urchin.envs             # noqa: F401  - importing registers the env IDs


# ================================================================ callbacks ===
class CurriculumCallback(BaseCallback):
    """Advances the env's stage when recent progress on the CURRENT stage clears `threshold`.
       seg_progress = distance along the command / (max_speed * CMD_HOLD_S), so 0.3 means
       'sustained 30% of max_speed in the right direction'."""

    def __init__(self, n_stages=5, threshold=0.3, window=100, verbose=1):
        super().__init__(verbose)
        self.n_stages, self.threshold = n_stages, threshold
        self.buf = deque(maxlen=window)
        self.stage = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            if "seg_progress" in info and info.get("cmd_stage") == self.stage:
                self.buf.append(info["seg_progress"])
        self.logger.record("curriculum/stage", self.stage)
        if self.buf:
            self.logger.record("curriculum/progress", float(np.mean(self.buf)))
            self.logger.record("curriculum/segments", len(self.buf))
        if (len(self.buf) == self.buf.maxlen and np.mean(self.buf) >= self.threshold
                and self.stage < self.n_stages - 1):
            self.stage += 1
            self.training_env.env_method("set_stage", self.stage)
            self.buf.clear()
            if self.verbose:
                print(f"[curriculum] -> stage {self.stage} at {self.num_timesteps} steps")
        return True


class InfoStatsCallback(BaseCallback):
    """Logs the mean of every reward component (info['rew/...']) and a few diagnostics,
       so you can see which term is driving the return."""

    KEYS = ("n_active", "seg_straightness")

    def __init__(self):
        super().__init__()
        self.sums, self.counts = defaultdict(float), defaultdict(int)

    def _on_step(self):
        for info in self.locals["infos"]:
            for k, v in info.items():
                if k.startswith("rew/") or k in self.KEYS:
                    self.sums[k] += float(v)
                    self.counts[k] += 1
        return True

    def _on_rollout_end(self):
        for k in self.sums:
            name = k if k.startswith("rew/") else f"diag/{k}"
            self.logger.record(name, self.sums[k] / max(self.counts[k], 1))
        self.sums.clear()
        self.counts.clear()


# ================================================================ training ===
def run(env_name, model_name, log_name, total_timesteps, n_envs=8, record_video=True,
        video_every_percent=10, video_seconds=7, episode_steps=1200, max_speed=0.05,
        curriculum_threshold=0.3, seed=0):
    os.makedirs("models", exist_ok=True)
    log_dir = os.path.join("logs", log_name)
    os.makedirs(log_dir, exist_ok=True)

    env_kwargs = dict(max_steps=episode_steps, max_speed=max_speed)
    env = make_vec_env(env_name, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)

    # Plain MLP over the 5 gait numbers: sparsity is enforced by the env's SparsePush,
    # and PPO's value function / log-probs are the standard, tested ones.
    model = PPO("MlpPolicy", env,
                n_steps=1024, batch_size=1024, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, target_kl=0.03,
                device="cpu", verbose=1, seed=seed,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[128, 128]),
                                   log_std_init=-1.0))
    model.set_logger(configure(log_dir, ["stdout", "csv"]))

    callbacks = [CurriculumCallback(threshold=curriculum_threshold), InfoStatsCallback()]
    if record_video:
        callbacks.append(TrainingVideoCallback(
            env_name=env_name,
            total_timesteps=total_timesteps,
            video_dir=os.path.join(log_dir, "videos"),
            every_percent=video_every_percent,
            video_seconds=video_seconds,
        ))

    try:
        model.learn(total_timesteps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("Interrupted - saving the current model.")

    model.save(os.path.join(log_dir, model_name))
    model.save(os.path.join("models", model_name))


# ========================================================== sanity checks ===
def check_prior(env_name, seconds=3.0, **env_kwargs):
    """Runs the hand-built gait (action = 0) in several directions, no learning.
       If this can't roll the robot roughly the right way, fix the prior before training.
       The printed speed is what you should use for max_speed."""
    env = gym.make(env_name, **env_kwargs).unwrapped
    env.auto_cmd = False
    n = int(round(seconds / env.policy_dt))
    speeds = []
    for cmd in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]:
        env.reset(seed=0)
        env.command(*cmd)
        p0 = env.root_xy()
        active = []
        for _ in range(n):
            _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
            active.append(info["n_active"])
        d = env.root_xy() - p0
        along, lat = d @ env.cmd_dir, d @ env.perp
        speeds.append(along / seconds)
        print(f"cmd {cmd}: along {along*100:6.1f} cm   sideways {lat*100:6.1f} cm   "
              f"speed {along/seconds*100:5.1f} cm/s   spines out (mean) {np.mean(active):.1f}")
    env.close()
    print(f"-> suggested max_speed ~ {max(np.median(speeds), 0.01):.3f} m/s")


# ============================================================ visualisation ===
def visualise(env_name, model_name, record_dir=None, video_name="visualisation.mp4",
              model_dir="models"):
    env = gym.make(env_name, max_steps=10**9)       # no truncation while watching
    raw = env.unwrapped
    model = PPO.load(os.path.join(model_dir, model_name), device="auto")
    raw.set_stage(4)                                 # show every kind of command

    dt = raw.policy_dt
    fps = max(1, round(1.0 / dt))
    obs, _ = env.reset()
    frames, renderer, cam = [], None, None
    last_cmd = raw.cmd.copy()
    print("command:", last_cmd)

    if record_dir is not None:
        renderer = mujoco.Renderer(raw.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = raw.core_id
        cam.distance, cam.azimuth, cam.elevation = 0.5, 135, -25

    try:
        with mujoco.viewer.launch_passive(raw.model, raw.data) as viewer:
            while viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if not np.array_equal(raw.cmd, last_cmd):
                    last_cmd = raw.cmd.copy()
                    print("command:", last_cmd)

                if renderer is not None:
                    renderer.update_scene(raw.data, camera=cam)
                    frames.append(renderer.render().copy())

                if terminated or truncated:
                    obs, _ = env.reset()

                viewer.sync()
                time.sleep(dt)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    if record_dir is not None and frames:
        os.makedirs(record_dir, exist_ok=True)
        path = os.path.join(record_dir, video_name)
        write_video(frames, path, fps=fps)
        print(f"Saved visualisation video: {path}")