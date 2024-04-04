import gymnasium as gym
from gymnasium import spaces
from jlab_rl.utils.circle_rdm import circle_rdm_samples

import numpy as np

class circle_constraint_env(gym.Env):
    def __init__(self, ndim=2):
        self.ndim = ndim
        self.action_space = spaces.Box(low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        self.states, _ = self.reset()
        self.delta_r = 0.25
        self.delta_r_max = 1.0
        self.target_radius = 0.95

    def step(self, action):
        self.states = action
        #radius = np.sqrt(self.states[0]*self.states[0]+self.states[1]*self.states[1])
        radius = np.sqrt(np.sum(self.states*self.states))
        radius_sqrt = np.square(radius - self.target_radius)
        reward = - radius_sqrt

        return self.states, reward, True, True, {}

    def reset(self):
        self.states, _, _ = circle_rdm_samples(self.ndim, 1, 1.0, 0.75, give_all=True)
        return self.states, ''

