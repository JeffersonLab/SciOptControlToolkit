# Author: Kishansingh Rajput
# Script: CEBAF cavities digital twin
# Org: Thomas Jefferson National Accelerator Facility

import numpy as np
import pandas as pd
import math
from gym import spaces
import gym
import os

from jlab_rl.envs.cebaf_surrogate_v0 import digitalTwin


class cebaf_env(gym.Env):
    def __init__(self, path_cavity_data=os.path.join(os.path.dirname(__file__),'cavity_table.pkl'),
                 linac="North", trackTime=False, max_steps=100, reward_weights=0.5, seed=22, termination_reward=-1000,
                 action_range=[-0.05, 0.05]):

        np.random.seed(seed=seed)

        # Build linac
        self.linac = digitalTwin(path_cavity_data, linac)
        self.ncavities = len(self.linac.list_cavities())

        # Get gradient ranges
        self.min_grads = self.linac.getMinGradients()
        self.max_grads = self.linac.getMaxGradients()

        # Assume everything fits on unit circle
        self.action_space = spaces.Box(low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)

        # Resent
        self.states, _ = self.reset()

        #self.reward_weights = reward_weights

        # # Define action
        # self.action_space = spaces.Box(
        #     low=-1.,
        #     high=1.,
        #     shape=(self.ncavities,),
        #     dtype=np.float32
        # )

        # Define state
        # states_low = [cavity.min_gset for cavity in self.linac.cavities]
        # states_high = [cavity.max_gset_to_use for cavity in self.linac.cavities]
        # states_low.extend([-1*self.linac.energyMargin, 0.])
        # states_high.extend([self.linac.energyMargin, 1.])
        # states_low, states_high = np.array(states_low), np.array(states_high)
        # self.observation_space = spaces.Box(
        #     low=0., #states_low,
        #     high=1., #states_high,
        #     shape=(self.ncavities,),
        #     dtype=np.float32
        # )
        #
        # self.states = self.reset() #self.linac.getGradients()
        # self.beta, self.gamma, self.alpha = 1., 1e3, 1. #0.1,50.0
        # self.max_steps = max_steps
        # self.counter = 0
        # self.termination_reward = termination_reward
        # self.best_reward = 0
        # self.action_range = action_range
        # self.minGradients = self.linac.getMinGradients()
        # self.maxGradients = self.linac.getMaxGradients()

    # def _computeReward(self):
    #     # make sure we can get the right energy
    #
    #     # c1 = self.reward_weights
    #     # c2 = 1. - c1
    #     # s = -1* np.exp((float(c1 * self.beta * self.linac.getRFHeat() + c2 * self.gamma * self.linac.getTripRates()))-21.)
    #     # if s==0:
    #     #     inverse = 100000.0
    #     # else:
    #     #     inverse = 1000.0 /s
    #
    #     return s # inverse
    
    def denormalize_action(self, action):
        denorm_action = action * (self.action_range[1] - self.action_range[0]) + self.action_range[0]
        return denorm_action
    
    def normalize_action(self, action):
        norm_action = (action - self.action_range[0])/(self.action_range[1] - self.action_range[0])
        return norm_action
    
    def normalize_state(self, state):
        ncavities = len(self.linac.cavities)
        gradients = state[:ncavities]
        norm_gradients = (gradients - self.minGradients) / (self.maxGradients - self.minGradients)
        energy = state[ncavities]
        norm_energy = (energy - (self.linac.energyConstraint - self.linac.energyMargin))/(2*self.linac.energyMargin)
        state[:ncavities] = norm_gradients
        state[ncavities] = norm_energy
        return state
    
    def denormalizeState(self, state):
        nCavities = len(self.linac.cavities)
        gradients = state[:nCavities]
        denorm_gradients = gradients * (self.maxGradients - self.minGradients) + self.minGradients
        energy = state[nCavities]
        denorm_energy = energy (2*self.linac.energyMargin) + (self.linac.energyConstraint - self.linac.energyMargin)
        state[:nCavities] = denorm_gradients
        state[nCavities] = denorm_energy
        return state
    
    def _takeAction(self, action):
        action = self.denormalizeAction(action)
        self.linac.updateGradients(action)

    def step(self, action):
        """

        """
        
        # self._takeAction(action)
        action = self.denormalize_action(action)
        self.linac.update_gradients(action)
        energy = self.linac.getEnergyGain()

        # reward = self._computeReward()
        # next_state = self.linac.getState()
        # next_state = np.append(next_state, self.reward_weights)
        # done = False
        # self.counter += 1
        #
        # if abs(self.linac.getEnergyGain() - self.linac.getEnergyConstraint()) > self.linac.getEnergyMargin():
        #     print("Invalid Action ", self.linac.getEnergyGain(), sum(self.linac.getState()[:8]))
        #     done = True
        #     reward = self.termination_reward
        #
        # elif self.counter >= self.max_steps:
        #     done = True
        #     reward = reward
        #
        # next_state = self.normalizeState(next_state)
        return next_state, reward, done, done, {'heat':self.linac.getRFHeat(), 'trip':self.linac.getTripRates(), 'energy':next_state[8]}

    def reset(self):
        #
        self.counter = 0
        self.states, _, _ = circle_rdm_samples(self.ndim, 1, 1.0, 0.75, give_all=True)

        #self.linac.reset()
        # r_w = np.random.uniform(0, 1)
        # self.reward_weights = 1. #r_w
        # self.counter = 0
        # state = np.append(self.linac.getState(), self.reward_weights)
        #return self.normalizeState(state)
        # assume normalized values
        return self.states, ''

    def getTripRates(self):
        return self.linac.getTripRates()
    def getRFHeat(self):
        return self.linac.getRFHeat()
