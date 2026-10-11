"""
Keyboard teleop for the tactile urchin.

Windows:
  1. MuJoCo passive viewer (3D scene)
  2. Matplotlib window: the 20 pads drawn with their real 10-taxel layout,
     left = decayed event trace (what the policy sees), right = raw pressure.

Keys (click the MuJoCo viewer first so it has focus):
  Arrow keys  roll in that direction (relative to the camera view)
  SPACE       stop rolling (rods retract)
  E / R       extend all rods / retract all rods
  M           manual mode: use the sliders in the viewer's right-hand
              "Control" panel to push each rod in and out yourself
  D           back to drive mode (arrow keys)
  Z           reset the simulation
"""
import time

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

import tactile_urchin    # <- change to wherever your class lives

KEY_SPACE, KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 32, 262, 263, 264, 265
KEY = {c: ord(c) for c in "EMRDZ"}

ENV_NAME = "Urchin"

# ---------------------------------------------------------------------------
class KeyboardController:
    def __init__(self, env, gait_amp=1.0, smooth=0.3):
        self.env = env
        self.gait_amp = gait_amp        # 0-1, fraction of full rod travel used when driving
        self.smooth = smooth            # low-pass on rod targets (0-1, higher = snappier)
        self.hi = env.action_space.high.astype(float)
        self.cur = np.zeros(env.action_dim)

        self.hub_ids = [
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"hub{i:02d}")
            for i in range(env.action_dim)
        ]
        self.mode = "drive"             # drive | manual | extend | retract
        self.fwd = 0                    # +1 up arrow, -1 down arrow
        self.right = 0                  # +1 right arrow, -1 left arrow
        self.reset_requested = False

    # called from the viewer's GUI thread: only set flags here
    def key_callback(self, key):
        if key == KEY_UP:       self.fwd, self.right, self.mode = 1, 0, "drive"
        elif key == KEY_DOWN:   self.fwd, self.right, self.mode = -1, 0, "drive"
        elif key == KEY_LEFT:   self.fwd, self.right, self.mode = 0, -1, "drive"
        elif key == KEY_RIGHT:  self.fwd, self.right, self.mode = 0, 1, "drive"
        elif key == KEY_SPACE:  self.fwd = self.right = 0
        elif key == KEY["E"]:   self.mode = "extend"
        elif key == KEY["R"]:   self.mode = "retract"
        elif key == KEY["M"]:   self.mode = "manual"
        elif key == KEY["D"]:   self.mode = "drive"
        elif key == KEY["Z"]:   self.reset_requested = True

    def action(self, cam_azimuth_deg):
        if self.mode == "manual":
            # leave whatever the viewer sliders wrote into ctrl
            self.cur = self.env.data.ctrl[:self.env.action_dim].copy()
            return self.cur.astype(np.float32)
        if self.mode == "extend":
            target = self.hi.copy()
        elif self.mode == "retract":
            target = np.zeros_like(self.hi)
        else:
            target = self._gait_target(cam_azimuth_deg)
        self.cur += self.smooth * (target - self.cur)
        return self.cur.astype(np.float32)

    def _gait_target(self, az_deg):
        """Extend the rods on the trailing, downward-facing side so the body rolls
        toward the commanded direction. Re-evaluated every step from the rods'
        current world orientation, so the roll keeps going."""
        if self.fwd == 0 and self.right == 0:
            return np.zeros_like(self.hi)
        az = np.deg2rad(az_deg)
        f = np.array([np.cos(az), np.sin(az), 0.0])      # camera forward on the ground
        r = np.array([np.sin(az), -np.cos(az), 0.0])     # camera right on the ground
        d = self.fwd * f + self.right * r
        d /= np.linalg.norm(d)

        # outward direction of each rod = local z axis of its hub, in world frame
        n = self.env.data.xmat[self.hub_ids].reshape(-1, 3, 3)[:, :, 2]
        behind = np.clip(-(n @ d), 0.0, 1.0)             # rods pointing opposite the motion
        downward = np.clip(0.5 - n[:, 2], 0.0, 1.0)      # rods pointing toward the floor
        return self.gait_amp * behind * downward * self.hi


# ---------------------------------------------------------------------------
class TactileView:
    # taxel (x, y) positions inside a pad, copied from the XML (metres)
    TAXEL_XY = np.array([
        [-0.0121839, -0.0211032], [-0.0121839, -0.00703439],
        [-0.0121839,  0.00703439], [-0.0121839, 0.0211032],
        [0.0, -0.0140688], [0.0, 0.0], [0.0, 0.0140688],
        [0.0121839, -0.00703439], [0.0121839, 0.00703439],
        [0.0243678, 0.0],
    ])

    def __init__(self, n_pads=20, n_taxels=10, cols=5, cell=0.075):
        rows = int(np.ceil(n_pads / cols))
        centres = np.array([[(p % cols) * cell, -(p // cols) * cell] for p in range(n_pads)])
        xy = (centres[:, None, :] + self.TAXEL_XY[None, :, :]).reshape(-1, 2)  # index = pad*10 + taxel

        plt.ion()
        self.fig, (self.ax_ev, self.ax_pr) = plt.subplots(1, 2, figsize=(12, 5.5))
        self.fig.canvas.manager.set_window_title("Tactile pads")
        kw = dict(s=110, edgecolors="k", linewidths=0.4)
        self.sc_ev = self.ax_ev.scatter(xy[:, 0], xy[:, 1], c=np.zeros(len(xy)),
                                        cmap="coolwarm", vmin=-2, vmax=2, **kw)
        self.sc_pr = self.ax_pr.scatter(xy[:, 0], xy[:, 1], c=np.zeros(len(xy)),
                                        cmap="viridis", vmin=0, vmax=0.5, **kw)
        for ax, title in ((self.ax_ev, "Event trace (signed)"), (self.ax_pr, "Pressure (N)")):
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title)
            for p in range(n_pads):
                ax.text(centres[p, 0] + 0.008, centres[p, 1] + 0.036, f"pad {p}",
                        ha="center", fontsize=7)
        self.fig.colorbar(self.sc_ev, ax=self.ax_ev, fraction=0.04)
        self.fig.colorbar(self.sc_pr, ax=self.ax_pr, fraction=0.04)
        self.suptitle = self.fig.suptitle("")
        self.fig.tight_layout()

    def alive(self):
        return plt.fignum_exists(self.fig.number)

    def update(self, trace, pressure, text=""):
        self.sc_ev.set_array(trace.ravel())
        self.sc_pr.set_array(pressure.ravel())
        self.sc_pr.set_clim(0, max(0.2, float(pressure.max())))
        self.suptitle.set_text(text)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


# ---------------------------------------------------------------------------
def main():
    env = gym.make(ENV_NAME)
    raw = env.unwrapped   # gym wrappers hide custom attributes, so use this for model/data/tactile
    env.reset()
    ctrl = KeyboardController(raw, gait_amp=1.0)
    view = TactileView(raw.n_pads, raw.n_taxels)
    dt = raw.frame_skip * raw.model.opt.timestep

    print(__doc__)
    with mujoco.viewer.launch_passive(raw.model, raw.data,
                                      key_callback=ctrl.key_callback) as viewer:
        viewer.cam.distance = 0.6
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 90

        i = 0
        while viewer.is_running() and view.alive():
            t0 = time.time()

            if ctrl.reset_requested:
                with viewer.lock():
                    env.reset()
                ctrl.cur[:] = 0
                ctrl.reset_requested = False

            action = ctrl.action(viewer.cam.azimuth)
            with viewer.lock():
                env.step(action)
            viewer.sync()

            if i % 2 == 0:   # ~25 Hz plot refresh
                n_ev = raw._count - raw._ev_start
                view.update(raw.trace, raw.pressure_map(),
                            f"mode: {ctrl.mode}  |  t = {raw.data.time:.2f}s  |  "
                            f"events this step: {n_ev}  |  total: {raw._count}")
            i += 1

            time.sleep(max(0.0, dt - (time.time() - t0)))   # real-time pacing


if __name__ == "__main__":
    main()