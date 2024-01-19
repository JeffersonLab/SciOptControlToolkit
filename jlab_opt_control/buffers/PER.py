import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.replay_core import Replay
from jlab_opt_control.buffers.ER import ER
import numpy as np
import os
import json

class PER(ER):
    def __init__(self, state_dim, action_dim, cfg='PER.cfg'):
        super().__init__(state_dim, action_dim, cfg)

    def update_priorities(self, indices, new_probabilities):
        for idx, probability in zip(indices, new_probabilities):
            self.probabilties[idx] = probability
