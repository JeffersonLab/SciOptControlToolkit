import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.replay_core import Replay
import numpy as np
import os
import json

class ER(Replay):
    def __init__(self, state_dim, action_dim, cfg='ER.cfg'):
        super().__init__(None, None, None, None, None, None)

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        pfn_json_file = os.path.join(full_path, cfg)
        with open(pfn_json_file) as json_file:
            data = json.load(json_file)

        self.buffer_size = int(cfg_utils.cfg_get(data, 'buffer_capacity', 50000))
        self.pointer = 0

        self.states = np.zeros((self.buffer_size, state_dim))
        self.actions = np.zeros((self.buffer_size, action_dim))
        self.rewards = np.zeros((self.buffer_size, 1))
        self.next_states = np.zeros((self.buffer_size, state_dim))
        self.dones = np.zeros((self.buffer_size, 1))
        self.probabilities = np.ones(self.buffer_size)

        self.indices = None
    
    def record(self, memory):
        state, action, reward, next_state, done, probability = memory
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        self.probabilities[self.pointer] = probability

        self.pointer = (self.pointer + 1) % self.buffer_size

    def sample(self, nsamples):
        # Find actual size of filled buffer
        max_index = min(self.pointer, self.buffer_size)

        # Normalize probabilites to sum to 1
        normalized_probabilities = self.probabilities[:max_index] / np.sum(self.probabilities[:max_index])

        # Select indicies from buffer based on above
        self.indices = np.random.choice(max_index, size=nsamples, replace=False, p=normalized_probabilities)

        return (
            self.states[self.indices],
            self.actions[self.indices],
            self.rewards[self.indices],
            self.next_states[self.indices],
            self.dones[self.indices],
            self.probabilities[self.indices]
        )
        
    def save(self, filename='replay_buffer.npy'):
        data = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "probabilities": self.probabilities
        }
        np.save(filename, data)
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True).item()
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_states = data["next_states"]
        self.dones = data["dones"]
        self.probabilities = data["probabilities"]
    
    def size(self):
        return min(self.pointer, self.buffer_size)
