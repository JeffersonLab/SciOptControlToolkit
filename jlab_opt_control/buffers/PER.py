import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.replay_core import Replay
import numpy as np

class PER(Replay):
    def __init__(self, state_dim, action_dim):
        super().__init__(None, None, None, None, None, None)

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        pfn_json_file = os.path.join(full_path, cfg)
        td3_log.debug(f'pfn_json_file:{pfn_json_file}')
        with open(pfn_json_file) as json_file:
            data = json.load(json_file)

        self.buffer_size = int(cfg_utils.cfg_get(data, 'buffer_capacity', 1000))
        self.pointer = 0

        self.states = np.zeros((self.buffer_size, state_dim))
        self.actions = np.zeros((self.buffer_size, action_dim))
        self.next_states = np.zeros((self.buffer_size, state_dim))
        self.rewards = np.zeros(buffer_size)
        self.dones = np.zeros(buffer_size, dtype=np.bool)
        
        self.probabilities = np.ones(buffer_size)
    
    def record(self, memory):
        state, action, next_state, reward, done, probability = memory
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.next_states[self.pointer] = next_state
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.probabilities[self.pointer] = probability

        self.pointer = (self.pointer + 1) % self.buffer_size

    def sample(self, nsamples):
        # Find actual size of filled buffer
        max_index = min(self.pointer, self.buffer_size)

        # Normalize probabilites to sum to 1
        normalized_probabilities = self.probabilities[:max_index] / np.sum(self.probabilities[:max_index])

        # Select indicies from buffer based on above
        indices = np.random.choice(max_index, size=nsamples, replace=False, p=normalized_probabilities)

        return (
            self.states[indicies],
            self.actions[indices],
            self.next_states[indices],
            self.rewards[indices],
            self.dones[indices],
            self.probabilities[indices]
        )
        
    def save(self, filename='replay_buffer.npy'):
        data = {
            "states": self.states,
            "actions": self.actions,
            "next_states": self.next_states,
            "rewards": self.rewards,
            "dones": self.dones,
            "probabilities": self.probabilities
        }
        np.save(filename, data)
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True).item()
        self.states = data["states"]
        self.actions = data["actions"]
        self.next_states = data["next_states"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.probabilities = data["probabilities"]
    
    def size(self):
        return min(self.pointer, self.buffer_size)
