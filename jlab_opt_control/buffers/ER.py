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

        self.buffer_capacity = int(cfg_utils.cfg_get(data, 'buffer_capacity', 50000))
        self.current_index = 0
        self.pointer = 0

        self.states = np.zeros((self.buffer_capacity, state_dim))
        self.actions = np.zeros((self.buffer_capacity, action_dim))
        self.rewards = np.zeros((self.buffer_capacity, 1))
        self.next_states = np.zeros((self.buffer_capacity, state_dim))
        self.dones = np.zeros((self.buffer_capacity, 1))
        self.priorities = np.ones(self.buffer_capacity)

        self.indices = None
        self.sample_counts = np.zeros(self.buffer_capacity)
    
    def record(self, memory):
        self.current_index = self.pointer % self.buffer_capacity

        self.states[self.current_index] = memory[0]
        self.actions[self.current_index] = memory[1]
        self.rewards[self.current_index] = memory[2]
        self.next_states[self.current_index] = memory[3]
        self.dones[self.current_index] = memory[4]
        self.priorities[self.current_index] = memory[5]

        # Reset count of sampling experience to zero if overwriting experiences
        if (self.pointer >= self.buffer_capacity):
            self.sample_counts[self.current_index] = 0

        self.pointer += 1

    def sample(self, nsamples):
        # Find actual size of filled buffer
        max_index = min(self.pointer, self.buffer_capacity)

        self.indices = np.random.choice(max_index, size=nsamples, replace=False)

        self.sample_counts[self.indices] += 1

        return (
            self.states[self.indices],
            self.actions[self.indices],
            self.rewards[self.indices],
            self.next_states[self.indices],
            self.dones[self.indices],
            self.priorities[self.indices]
        )
        
    def save(self, filename='replay_buffer.npy'):
        data = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "priorities": self.priorities
        }
        np.save(filename, data)
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True).item()
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_states = data["next_states"]
        self.dones = data["dones"]
        self.priorities = data["priorities"]
    
    def size(self):
        return min(self.pointer, self.buffer_capacity)
