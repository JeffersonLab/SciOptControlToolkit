import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.replay_core import Replay
from jlab_opt_control.buffers.er import ER
import numpy as np
import os
import json
import logging
import shutil


class PER(ER):
    def __init__(self, state_dim, action_dim, logdir, buffer_size=None, cfg='per.cfg'):
        super().__init__(state_dim, action_dim, logdir, buffer_size, cfg)

        self.tds = np.zeros(self.buffer_capacity)

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        pfn_json_file = os.path.join(full_path, cfg)
        with open(pfn_json_file) as json_file:
            data = json.load(json_file)

        self.alpha = float(cfg_utils.cfg_get(data, 'alpha', 0.6))
        self.beta = float(cfg_utils.cfg_get(data, 'beta', 0.4))
        self.beta_increment = float(
            cfg_utils.cfg_get(data, 'beta_increment', 0.001))
        self.prioritization_type = str(
            cfg_utils.cfg_get(data, 'prioritization_type', None))

        self.max_priority = 1.0

    def sample(self, nsamples):

        # Find actual size of filled buffer
        max_index = min(self.pointer, self.buffer_capacity)

        if self.prioritization_type == "proportional":
            probabilities = self.priorities[:max_index] ** self.alpha
            # Normalize probabilites to sum to 1
            normalized_probabilities = probabilities / np.sum(probabilities)

        elif self.prioritization_type == "rank":
            sorted_indices = np.argsort(-self.priorities[:max_index])
            ranks = np.argsort(sorted_indices) + 1

            rank_based_probs = (1 / ranks) ** self.alpha
            normalized_probabilities = rank_based_probs / \
                np.sum(rank_based_probs)

        else:
            print(
                "ERROR: Please select a proper prioritization type in the PER.cfg config (proportional/rank)")

        # Select indicies from buffer based on above
        self.indices = np.random.choice(
            max_index, size=nsamples, replace=False, p=normalized_probabilities)

        self.sample_counts[self.indices] += 1

        # Computing the importance-sampling weights using beta
        weights = (
            1 / (max_index * normalized_probabilities[self.indices])) ** self.beta
        weights /= weights.max()  # Normalize weights

        # Increment beta value
        self._update_beta()

        return (
            self.states[self.indices],
            self.actions[self.indices],
            self.rewards[self.indices],
            self.next_states[self.indices],
            self.dones[self.indices],
            weights
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

        # Proportional Prioritization Method

        # Update the TD array with returned values
        for idx, td in zip(self.indices, new_tds):
            self.tds[idx] = td

        if (new_tds.max() > self.max_priority):
            self.max_priority = new_tds.max()

        # Update the priorities with the normalized TDs
        non_zero_inices = np.nonzero(self.tds)
        self.priorities[non_zero_inices] = 1e-4 + self.tds[non_zero_inices]

        # Normalization Code

        # max_td_error = np.max(new_tds) if np.max(new_tds) > 0 else 1
        # normalized_new_tds = new_tds / max_td_error

        # for idx, new_td in zip (self.indices, normalized_new_tds):
        #     self.priorities[idx] = 1e-4 + 1.0 + new_td

        # if (1 + normalized_new_tds.max() > self.max_priority):
        #     self.max_priority = 1 + normalized_new_tds.max()
        #     print("New max priority: ", self.max_priority)

        # # normalized_tds = self.tds / np.sum(self.tds)
        # # self.priorities[non_zero_inices] = 1 + normalized_tds[non_zero_inices]

    def _update_beta(self):
        self.beta = min(self.beta + self.beta_increment, 1.0)
