import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.replay_core import Replay
import numpy as np
import os
import json
import logging
import shutil

buf_log = logging.getLogger("Buffer")
buf_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class ER(Replay):
    def __init__(self, state_dim, action_dim, logdir, buffer_size=None, cfg='er.cfg'):
        super().__init__(None, None, None, None, None, None)

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)
        with open(self.pfn_json_file) as json_file:
            data = json.load(json_file)

        if buffer_size is None:
            self.buffer_capacity = int(
                cfg_utils.cfg_get(data, 'buffer_capacity', 50000))
        else:
            self.buffer_capacity = buffer_size
        self.current_index = 0
        self.pointer = 0

        self.logdir = logdir

        self.num_states = state_dim
        self.num_actions = action_dim

        self.states = np.zeros((self.buffer_capacity, self.num_states))
        self.actions = np.zeros((self.buffer_capacity, self.num_actions))
        self.rewards = np.zeros((self.buffer_capacity, 1))
        self.next_states = np.zeros((self.buffer_capacity, self.num_states))
        self.dones = np.zeros((self.buffer_capacity, 1))
        self.priorities = np.ones(self.buffer_capacity)

        self.indices = None
        self.sample_counts = np.zeros((self.buffer_capacity, 1))

        self.max_priority = 1.0

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

        self.indices = np.random.choice(
            max_index, size=nsamples, replace=False)

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

    def save_cfg(self):
        """ Save the buffer cfg """
        try:
            destination_file_path = os.path.join(self.logdir, 'cfgs/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)
            destination_file_path = os.path.join(
                destination_file_path, os.path.basename(self.pfn_json_file))
            shutil.copy(self.pfn_json_file, destination_file_path)
            buf_log.info('Buffer config saved successfully')
        except:
            buf_log.error("Error in saving the buffer cfg...")

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
