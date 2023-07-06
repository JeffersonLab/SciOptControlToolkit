import gym
from gym import spaces
from jlab_rl.utils.circle_rdm import circle_rdm_samples

import numpy as np

class circle_constraint_env(gym.Env):
    def __init__(self):
        self.ndim = 2
        self.action_space = spaces.Box(low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        self.states, _ = self.reset()
        self.delta_r = 0.25
        self.delta_r_max = 1.0
        self.target_radius = 0.95

    def step(self, action):
        #print('Pre-states:', self.states)
        #print('Actions:', action)
        self.states = self.states + action
        #print('Post-states:', self.states)
        sqrt_states = np.square(self.states)
        #print('Sqrt-states:', sqrt_states)
        radius = np.sqrt(np.sum(sqrt_states))
        #radius = np.sqrt(self.states[0]*self.states[0]+self.states[1]*self.states[1])
        #print('Post-radius:', radius)
        #print('Post-radius (np):', np_radius)
        #radius_sqrt = np.square(radius - self.target_radius)
        reward = - np.log(np.abs(radius - self.target_radius)) - 100 * np.square(radius - self.target_radius)
        #reward = - radius_sqrt# -np.log(np.abs(radius-self.target_radius)) # Log-Linear reward
        if self.states.any() > 1:
            reward = -99
        if self.states.any() < -1:
            reward = -99
        # if radius > 1:
        #     reward = -99

        #print('Reward:', reward)

        # radius_sqrt = np.square(radius - self.target_radius)
        # reward = np.exp(-100 * np.square(test_state - ideal_r))
        # radius_sqrt = np.square(radius - self.target_radius)

        return self.states, reward, False, False, {}

    def reset(self):
        self.states, _, _ = circle_rdm_samples(self.ndim, 1, 1.0, 0.75, give_all=True)
        #self.states = np.abs(self.states)
        return self.states, ''

