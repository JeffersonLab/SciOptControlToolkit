# Author: Kishansingh Rajput
# Script: CEBAF cavities digital twin
# Org: Thomas Jefferson National Accelerator Facility

import numpy as np
import pandas as pd
import math
from gym import spaces
import gym
import os

from cebaf_opt.envs.rl_envs.cebaf_surrogate_v0 import digitalTwin


class cebaf_env(gym.Env):
    def __init__(self, path_cavity_data=os.path.join(os.path.dirname(__file__),'cavity_table.pkl'),
                 linac="North", trackTime=False, max_steps=100, reward_weights=0.5, seed=22, termination_reward=-1000,
                 action_range=[-0.05, 0.05]):
        np.random.seed(seed=seed)
        self.linac = digitalTwin(path_cavity_data, linac)
        self.nCavities = len(self.linac.list_cavities())
        self.reward_weights = reward_weights 
        self.action_space = spaces.Box(
            low=0.,
            high=1.,
            shape=(self.nCavities,),
            dtype=np.float32
        )
        
        states_low = [cavity.min_gset for cavity in self.linac.cavities]
        states_high = [cavity.max_gset_to_use for cavity in self.linac.cavities]
        states_low.extend([-1*self.linac.energyMargin, 0.])
        states_high.extend([self.linac.energyMargin, 1.])
        states_low, states_high = np.array(states_low), np.array(states_high)
        self.observation_space = spaces.Box(
            low=0., #states_low,
            high=1., #states_high,
            shape=(self.nCavities+2,),
            dtype=np.float32
        )

        self.states = self.linac.getGradients()
        self.beta, self.gamma, self.alpha = 1., 1e3, 1. #0.1,50.0
        self.max_steps = max_steps
        self.counter = 0
        self.termination_reward = termination_reward
        self.best_reward = 0
        self.action_range = action_range
        self.minGradients = self.linac.getMinGradients()
        self.maxGradients = self.linac.getMaxGradients()

    def _computeReward(self):
        c1 = self.reward_weights
        c2 = 1. - c1
        s = -1* np.exp((float(c1 * self.beta * self.linac.getRFHeat() + c2 * self.gamma * self.linac.getTripRates()))-21.)
        if s==0:
            inverse = 100000.0
        else:
            inverse = 1000.0 /s

        return s # inverse
    
    def denormalizeAction(self, action):
#         print(self.action_range)
        denorm_action = action * (self.action_range[1] - self.action_range[0]) + self.action_range[0]
#         print(denorm_action)
        return denorm_action
    
    def normalizeAction(self, action):
        norm_action = (action - self.action_range[0])/(self.action_range[1] - self.action_range[0])
        return norm_action
    
    def normalizeState(self, state):
        nCavities = len(self.linac.cavities)
        gradients = state[:nCavities]
        norm_gradients = (gradients - self.minGradients) / (self.maxGradients - self.minGradients)
        energy = state[nCavities]
        norm_energy = (energy - (self.linac.energyConstraint - self.linac.energyMargin))/(2*self.linac.energyMargin)
        state[:nCavities] = norm_gradients
        state[nCavities] = norm_energy
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
#         print("Applying Actions: ", action)
        self.linac.updateGradients(action)

    def step(self, action):
        """

        """
        
        self._takeAction(action)
        reward = self._computeReward()
        next_state = self.linac.getState()
        next_state = np.append(next_state, self.reward_weights)
        done = False
        self.counter += 1

        if abs(self.linac.getEnergyGain() - self.linac.getEnergyConstraint()) > self.linac.getEnergyMargin():
            print("Invalid Action ", self.linac.getEnergyGain(), sum(self.linac.getState()[:8]))
            done = True
            reward = self.termination_reward
            
        elif self.counter >= self.max_steps:
            done = True
            reward = reward
        
        next_state = self.normalizeState(next_state)
        return next_state, reward, done, {'heat':self.linac.getRFHeat(), 'trip':self.linac.getTripRates(), 'energy':next_state[8]}

    def reset(self):
        r_w = np.random.uniform(0, 1)
        self.reward_weights = 1. #r_w 
        self.linac.reset()
        self.counter = 0
        state = np.append(self.linac.getState(), self.reward_weights)
        return self.normalizeState(state)

    def getTripRates(self):
        return self.linac.getTripRates()
    def getRFHeat(self):
        return self.linac.getRFHeat()
