import time
import numpy as np
import gymnasium as gym
import mujoco, mujoco.viewer
from stable_baselines3 import PPO, SAC
import sys
from RL.policies import *
import RL.policies as _policies
sys.modules["policies"] = _policies  
from tactile_urchin.envs import Urchin      # <- adjust to where your base class lives

CMD_SPEED = 0.3          # fixed commanded speed. Match the range the policy was trained on.

# GLFW keycodes. Each key sets ONE axis, so combining two keys gives a diagonal.
#   W / S -> forward / back  (cmd x = +1 / -1)
#   A / D -> left / right    (cmd y = +1 / -1)
#   X     -> stop
KEY_W, KEY_A, KEY_S, KEY_D, KEY_X = ord("W"), ord("A"), ord("S"), ord("D"), ord("X")


class KeyboardCommand:
    """Latches a unit direction from key presses. Runs on the viewer thread."""
    def __init__(self, speed):
        self.speed = speed
        self.axes = np.zeros(2)                  # [x, y] in {-1, 0, 1}

    def on_key(self, keycode):
        before = self.axes.copy()
        if keycode == KEY_W:   self.axes[0] = 1
        elif keycode == KEY_S: self.axes[0] = -1
        elif keycode == KEY_A: self.axes[1] = 1
        elif keycode == KEY_D: self.axes[1] = -1
        elif keycode == KEY_X: self.axes[:] = 0
        if not np.array_equal(before, self.axes):
            print(f"command -> {self.vector()}")

    def vector(self):
        n = np.linalg.norm(self.axes)
        return self.speed * self.axes / n if n > 0 else np.zeros(2)


def apply_cmd(raw, vec):
    """Overwrite the env's command. Assumes raw.cmd is an array whose first two
    entries are (vx, vy); any extra entries (e.g. yaw rate) are zeroed."""
    c = np.zeros_like(np.asarray(raw.cmd, dtype=float))
    c[:2] = vec
    raw.cmd = c


def visualise(env_name, model_name):
    env = gym.make(env_name)
    raw = env.unwrapped
    model = PPO.load(f"models/{model_name}", device="cpu")
    dt = raw.frame_skip * raw.model.opt.timestep
    obs, _ = env.reset()

    kb = KeyboardCommand(CMD_SPEED)
    print("Controls: W/A/S/D = move (combine for diagonals), X = stop")

    with mujoco.viewer.launch_passive(raw.model, raw.data, key_callback=kb.on_key) as v:
        while v.is_running():
            with v.lock():
                apply_cmd(raw, kb.vector())          # keyboard overrides the sampled command
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()
                    print("reset")
            v.sync()
            time.sleep(dt)


ENV_NAME = "Urchin"
MODEL_NAME = "urchin_v2"      # -> models/urchin_v2.zip

if __name__ == "__main__":
    visualise(ENV_NAME, MODEL_NAME)