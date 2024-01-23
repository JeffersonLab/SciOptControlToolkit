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
        self.tds = np.zeros(self.buffer_capacity)

    def sample(self, nsamples):
        # Find actual size of filled buffer
        max_index = min(self.pointer, self.buffer_capacity)

        # Normalize probabilites to sum to 1
        normalized_probabilities = self.priorities[:max_index] / np.sum(self.priorities[:max_index])
        
        # Select indicies from buffer based on above
        self.indices = np.random.choice(max_index, size=nsamples, replace=False, p=normalized_probabilities)

        self.sample_counts[self.indices] += 1

        return (
            self.states[self.indices],
            self.actions[self.indices],
            self.rewards[self.indices],
            self.next_states[self.indices],
            self.dones[self.indices],
            self.priorities[self.indices]
        )

    def record(self, memory):
        self.current_index = self.pointer % self.buffer_capacity

        self.states[self.current_index] = memory[0]
        self.actions[self.current_index] = memory[1]
        self.rewards[self.current_index] = memory[2]
        self.next_states[self.current_index] = memory[3]
        self.dones[self.current_index] = memory[4]
        self.priorities[self.current_index] = memory[5]

        # Reset td value to zero (default for new experiences)
        # Reset count of sampling experience to zero if overwriting experiences
        if (self.pointer >= self.buffer_capacity):
            self.sample_counts[self.current_index] = 0
            self.tds[self.current_index] = 0

        self.pointer += 1

    def update_priorities(self, new_tds):
        # Update the TD array with returned values
        for idx, td in zip(self.indices, new_tds):
            self.tds[idx] = td
        
        # Update the priorities with the normalized TDs
        non_zero_inices = np.nonzero(self.tds)
        normalized_tds = self.tds / np.sum(self.tds)
        self.priorities[non_zero_inices] = 1 + normalized_tds[non_zero_inices]

