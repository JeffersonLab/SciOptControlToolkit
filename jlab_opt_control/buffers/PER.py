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

    def sample(self, nsamples):
        # Find actual size of filled buffer
        max_index = min(self.pointer, self.buffer_capacity)

        # Normalize probabilites to sum to 1
        normalized_probabilities = self.probabilities[:max_index] / np.sum(self.probabilities[:max_index])
        
        # Select indicies from buffer based on above
        self.indices = np.random.choice(max_index, size=nsamples, replace=False, p=normalized_probabilities)

        self.sample_counts[self.indices] += 1

        return (
            self.states[self.indices],
            self.actions[self.indices],
            self.rewards[self.indices],
            self.next_states[self.indices],
            self.dones[self.indices],
            self.probabilities[self.indices]
        )

    def update_priorities(self, new_probabilities):
        for idx, probability in zip(self.indices, new_probabilities):
            self.probabilities[idx] = probability
