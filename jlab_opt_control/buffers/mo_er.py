import logging
import numpy as np
from jlab_opt_control.buffers.er import ER

buf_log = logging.getLogger("MO_Buffer")
buf_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class MO_ER(ER):
    def __init__(self, state_dim, action_dim, reward_dim, logdir, buffer_size=None, cfg='mo_er.cfg'):
        super().__init__(state_dim, action_dim, logdir, buffer_size, cfg)

        # Overwrite reward dim and reward buffer array with appropriate dimensions
        self.reward_dim = reward_dim
        self.rewards = np.zeros((self.buffer_capacity, self.reward_dim))
        
        # Create an entry for storing alphas for MO 
        self.alphas = np.zeros((self.buffer_capacity, self.reward_dim))
        

    def record(self, memory):
        super().record(memory[:6])
        self.alphas[self.current_index] = memory[6]

    def sample(self, nsamples):
        trad_tuple = super().sample(nsamples)
        return (
            *trad_tuple,
            self.alphas[self.indices]
        )

    def save(self, filename='replay_buffer.npy'):
        data = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "priorities": self.priorities,
            "alphas": self.alphas
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
        self.alphas = data['alphas']