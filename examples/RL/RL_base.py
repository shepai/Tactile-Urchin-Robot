 
import os, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco, mujoco.viewer
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from policies import *

from tactile_urchin.envs import Urchin      # <- adjust to where your base class lives



def run(env_name, model_name, log_name, total_timesteps, n_envs=8):
    os.makedirs("models", exist_ok=True)
    env = make_vec_env(env_name, n_envs=n_envs)
    model = PPO(SphericalCPGPolicy, env, n_steps=512, batch_size=1024, device="cpu", verbose=1)
    model.set_logger(configure(f"logs/{log_name}", ["stdout", "csv"]))   # add "tensorboard" if installed
    model.learn(total_timesteps)
    model.save(f"models/{model_name}")
 
 
def visualise(env_name, model_name):
    env = gym.make(env_name)
    raw = env.unwrapped
    model = PPO.load(f"models/{model_name}", device="cpu")
    dt = raw.frame_skip * raw.model.opt.timestep
    obs, _ = env.reset()
    with mujoco.viewer.launch_passive(raw.model, raw.data) as v:
        while v.is_running():
            action, _ = model.predict(obs, deterministic=True)
            with v.lock():
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()
                    print("new command:", raw.cmd)
            v.sync()
            time.sleep(dt)