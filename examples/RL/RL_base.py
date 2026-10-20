 
import os, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco, mujoco.viewer
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from policies import *
from recorder import *
from tactile_urchin.envs import Urchin      # <- adjust to where your base class lives


def run(env_name, model_name, log_name, total_timesteps, n_envs=8,record_video=True,video_every_percent=10,video_seconds=5):
    os.makedirs("models", exist_ok=True)
    log_dir = os.path.join("logs", log_name)
    os.makedirs(log_dir, exist_ok=True)
    env = make_vec_env(env_name, n_envs=n_envs)
    model = PPO(SphericalCPGPolicy, env, n_steps=512, batch_size=1024, device="cpu", verbose=1)
    model.set_logger(configure(f"logs/{log_name}", ["stdout", "csv"]))   # add "tensorboard" if installed
    callbacks = []
    if record_video:
        video_dir = os.path.join(log_dir, "videos")

        callbacks.append(
            TrainingVideoCallback(
                env_name=env_name,
                total_timesteps=total_timesteps,
                video_dir=video_dir,
                every_percent=video_every_percent,
                video_seconds=video_seconds,
            )
        )
    model.learn(total_timesteps)
    model.save(f"models/{model_name}")
 
 
def visualise(env_name, model_name,record_dir=None,video_name="visualisation.mp4"):
    env = gym.make(env_name, render_mode="rgb_array")
    raw = env.unwrapped

    model = PPO.load(
        os.path.join("models", model_name),
        device="cpu",
    )

    dt = raw.frame_skip * raw.model.opt.timestep
    fps = max(1, round(1.0 / dt))

    obs, _ = env.reset()
    frames = []

    try:
        with mujoco.viewer.launch_passive(
            raw.model, raw.data
        ) as viewer:
            while viewer.is_running():
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, _, terminated, truncated, _ = env.step(action)

                if record_dir is not None:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame).copy())

                if terminated or truncated:
                    obs, _ = env.reset()
                    print("new command:", getattr(raw, "cmd", None))

                viewer.sync()
                time.sleep(dt)

    finally:
        env.close()

    if record_dir is not None and frames:
        video_path = os.path.join(record_dir, video_name)
        write_video(frames, video_path, fps=fps)
        print(f"Saved visualisation video: {video_path}")