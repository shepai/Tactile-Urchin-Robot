from .base import TactileUrchin

class Plane(TactileUrchin):
    def __init__(self, xml_subpath=("assets", "icosabot.xml"),
                 obs_dim=230,
                 action_dim=12,
                 **kwargs):
        super().__init__(
            xml_subpath=xml_subpath,
            obs_dim=obs_dim,
            action_dim=action_dim,
            **kwargs,   # frame_skip, threshold, refractory, etc.
        )

    def _reward(self):
        # replace with your task reward
        return 0.0

    def _done(self):
        return False