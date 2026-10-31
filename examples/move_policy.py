import time
import numpy as np
import gymnasium as gym
import mujoco, mujoco.viewer
from stable_baselines3 import PPO

import tactile_urchin.envs      # noqa: F401  - importing registers the env IDs

# GLFW keycodes. Each key sets ONE axis, so pressing two keys gives a diagonal.
#   Up / Down    -> world +x / -x
#   Left / Right -> world +y / -y
#   Space or X   -> stop (hold still)
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_SPACE, KEY_X = 32, ord("X")


class KeyboardCommand:
    """Latches a direction from key presses. Runs on the viewer thread.
       The viewer only reports key presses (not releases), so a direction stays
       set until you press another key."""

    def __init__(self):
        self.axes = np.zeros(2)          # [x, y] in {-1, 0, 1}
        self.changed = True              # apply the first command straight away

    def on_key(self, keycode):
        before = self.axes.copy()
        if keycode == KEY_UP:              self.axes[0] = 1
        elif keycode == KEY_DOWN:          self.axes[0] = -1
        elif keycode == KEY_LEFT:          self.axes[1] = 1
        elif keycode == KEY_RIGHT:         self.axes[1] = -1
        elif keycode in (KEY_SPACE, KEY_X): self.axes[:] = 0
        if not np.array_equal(before, self.axes):
            self.changed = True
            print(f"command -> {self.axes.astype(int)}")


def visualise(env_name, model_name):
    # no time limit, and no automatic command changes: the keyboard is in charge
    env = gym.make(env_name, max_steps=10**9, auto_cmd=False)
    raw = env.unwrapped
    model = PPO.load(f"models/{model_name}", device="auto")
    dt = raw.policy_dt

    kb = KeyboardCommand()
    obs, _ = env.reset(options={"cmd": (0, 0)})
    print("Controls: arrow keys = move (press two for a diagonal), Space or X = stop")

    with mujoco.viewer.launch_passive(raw.model, raw.data, key_callback=kb.on_key) as v:
        while v.is_running():
            with v.lock():
                if kb.changed:                         # only on a new command, so progress
                    kb.changed = False                 # tracking isn't restarted every step
                    raw.command(*kb.axes)
                    obs = raw._get_obs()               # let the policy see it immediately
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    obs, _ = env.reset(options={"cmd": tuple(kb.axes)})
                    print("reset")
            v.sync()
            time.sleep(dt)


ENV_NAME = "Urchin"
MODEL_NAME = "urchin_v2"      # -> models/urchin_v2.zip (the old v1 model won't load into the new env)

if __name__ == "__main__":
    visualise(ENV_NAME, MODEL_NAME)