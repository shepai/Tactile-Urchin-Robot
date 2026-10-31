import numpy as np, gymnasium as gym
from policies import TaxelFlowPolicy, spine_directions, taxel_directions   # adjust import path
from policies import *
from recorder import *
from tactile_urchin.envs import Urchin  

def prior_test(env_name, steps=300, elevs=(20, 35, 50, 65, 80)):
    probe = gym.make(env_name); raw = probe.unwrapped
    S, A = spine_directions(raw.model), taxel_directions(raw.model, raw.data)
    cmds = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
    for elev in elevs:
        pol = TaxelFlowPolicy(probe.observation_space, probe.action_space, lambda _: 3e-4,
                              spine_directions=S, anchor_directions=A, elev_deg=elev)
        row = []
        for name, c in cmds.items():
            probe.reset(seed=0)
            raw.cmd = np.array(c, float)
            obs = raw._get_obs()
            p0 = raw.data.qpos[:2].copy()
            for _ in range(steps):
                a, _ = pol.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = probe.step(a)
                if term or trunc:
                    break
            d = (raw.data.qpos[:2] - p0) * 100                    # displacement in cm
            row.append(f"{name}: ({d[0]:+5.1f},{d[1]:+5.1f})")
        print(f"elev {elev:2d} | " + "   ".join(row))

if __name__ == "__main__":
    prior_test("Urchin")