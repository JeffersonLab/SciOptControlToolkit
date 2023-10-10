import sys

import gym
from gym import spaces

import tensorflow as tf
import numpy as np

class sin_env(gym.Env):
    def __init__(self, statefull=True):
        self.ndim = 1
        self.target_value = 0
        self.statefull = statefull
        self.action_space = spaces.Box(low=-1.5*np.pi*np.ones(self.ndim), high=+1.5*np.pi*np.ones(self.ndim), dtype=np.float64)
        self.observation_space = spaces.Box(low=-1.5*np.pi*np.ones(self.ndim), high=+1.5*np.pi*np.ones(self.ndim), dtype=np.float64)

        #self.init_state = np.ones(self.ndim)*np.pi
        self.init_state = np.zeros(self.ndim)
        print('Reset state', self.init_state)
        self.states, _ = self.reset()

    def step(self, action):
        # print('pre-action:',action)
        # action = action*(4*np.pi)-2*np.pi
        # print('post-action:',action)
        self.states = self.states + action
        if self.statefull==False:
            self.states = action

        y = np.sin(self.states)
        reward = 1000.0*np.exp(-5.0*np.abs(y - self.target_value)+1e-6)
        #reward = - np.sum(np.abs(y - self.target_value))

        #print('post-states:', self.states)
        # y = np.abs(np.sin(self.states))
        # reward = - float(y)

        # if self.states.any() > 1.5*np.pi*1.01:
        #     reward = -99
        # if self.states.any() < -1.5 * np.pi*1.01:
        #     reward = -99

        return self.states, reward, True, True, {}

    def sample_action(self):
        return self.action_space.sample()

    def reset(self):
        self.states = self.init_state
        #self.states = np.expand_dims(self.init_state, axis=0)
        return self.states, ''