import sys

import gym
from gym import spaces

import tensorflow as tf
import numpy as np

class sin_env(gym.Env):
    def __init__(self, rdm_reset=None):
        self.rdm_reset = rdm_reset
        self.ndim = 1
        self.action_space = spaces.Box(low=-2*np.pi*np.ones(self.ndim), high=+2*np.pi*np.ones(self.ndim), dtype=np.float64)
        self.observation_space = spaces.Box(low=-2*np.pi*np.ones(self.ndim), high=+2*np.pi*np.ones(self.ndim), dtype=np.float64)
        # self.action_space = spaces.Box(low=0.0, high=1, dtype=np.float64)
        # self.observation_space = spaces.Box(low=0, high=1, dtype=np.float64)

        self.init_state = np.zeros(self.ndim)
        print('Reset state',self.init_state)
        self.states, _ = self.reset()

    def step(self, action):
        # print('pre-action:',action)
        # action = action*(4*np.pi)-2*np.pi
        # print('post-action:',action)
        self.states = self.states + action
        #print('post-states:', self.states)
        reco_y = np.abs(np.sin(self.states))
        reward = - float(reco_y)

        if self.states.any() > 2*np.pi*1.01:
            reward = -99
        if self.states.any() < -2 * np.pi*1.01:
            reward = -99

        return self.states, reward, True, True, {}

    def sample_action(self):
        return self.action_space.sample()

    def reset(self):
        self.states = self.init_state
        #self.states = np.expand_dims(self.init_state, axis=0)
        return self.states, ''