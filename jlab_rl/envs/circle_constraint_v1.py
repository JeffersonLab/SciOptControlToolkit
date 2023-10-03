import gym
from gym import spaces
from jlab_rl.utils.circle_rdm import circle_rdm_samples

import numpy as np


class circle_constraint_env(gym.Env):
    def __init__(self, ndim=2, rdm_reset_mode='circle', statefull=True):
        self.ndim = ndim
        self.rdm_reset_mode = rdm_reset_mode
        self.statefull = statefull
        self.action_space = spaces.Box(low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        self.states, _ = self.reset()
        self.delta_r = 0.25
        self.delta_r_max = 1.0
        self.target_radius = 0.95

    def step(self, action):
        self.states = self.states + action
        if self.statefull==False:
            self.states = action
        sqrt_states = np.square(self.states)
        radius = np.sqrt(np.sum(sqrt_states))
        #reward = - np.log(np.abs(radius - self.target_radius)) - 100 * np.square(radius - self.target_radius)
        #reward = - np.abs(radius - self.target_radius)+100
        #reward = - - np.log(np.abs(radius - self.target_radius)+1.1)
        reward = 1000.0*np.exp(-5.0*np.abs(radius - self.target_radius)+1e-6)
        # if self.states.any() > 1:
        #     reward = -99
        # if self.states.any() < -1:
        #     reward = -99
        return self.states, reward, True, True, {}

    def reset(self):
        if self.rdm_reset_mode == 'circle':
            self.states, _, _ = circle_rdm_samples(self.ndim, 1, 1.0, 0.75, give_all=True)
        if self.rdm_reset_mode == 'uniform':
            self.states = self.observation_space.sample()
        if self.rdm_reset_mode == 'fixed':
            self.states = np.zeros(self.ndim)
        return self.states, ''

