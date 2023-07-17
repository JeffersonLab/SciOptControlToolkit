# Author: Kishansingh Rajput
# Script: CEBAF cavities digital twin
# Org: Thomas Jefferson National Accelerator Facility

import numpy as np
import pandas as pd
import math
from gym import spaces
import gym
import os, sys

from jlab_rl.envs.cebaf_surrogate_v0 import digitalTwin
from jlab_rl.utils.circle_rdm import circle_rdm_samples


class cebaf_env(gym.Env):
    def __init__(self, path_cavity_data=os.path.join(os.path.dirname(__file__), 'updated_cavity_table.pkl'),
                 linac="North", rdm_reset_mode='circle', seed=22):
        np.random.seed(seed=seed)

        self.rdm_reset_mode = rdm_reset_mode
        self.opt = 'other'
        self.alpha = 1.0

        # Build linac
        self.linac = digitalTwin(path_cavity_data, linac)
        self.ncavities = len(self.linac.list_cavities())
        self.mean_energy = self.linac.energyConstraint
        self.delta_energy = self.linac.energyMargin

        self.target_energy = self.mean_energy
        self.max_energy = self.mean_energy + self.delta_energy
        self.min_energy = self.mean_energy - self.delta_energy

        print('Target energy:', self.target_energy)
        print('Energy min/max: {}/{}'.format(self.min_energy, self.max_energy))

        # Get gradient ranges
        self.min_grads = self.linac.getMinGradients()
        self.max_grads = self.linac.getMaxGradients()

        # Define action range

        # Assume everything fits on unit circle
        self.action_space = spaces.Box(low=-np.ones(self.ncavities), high=np.ones(self.ncavities), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.ones(self.ncavities), high=np.ones(self.ncavities), dtype=np.float64)

        # Resent
        self.states, _ = self.reset()
        self.energy = self.linac.getEnergyGain()

    def normalize_energy(self, energy):
        return (energy-self.min_energy)/(self.max_energy-self.min_energy)

    def denormalize_energy(self, normalized_energy):
        return normalized_energy*(self.max_energy-self.min_energy) + self.min_energy

    def normalize_state(self, state):
        normalized_state = 2 * ((state - self.min_grads) / (self.max_grads - self.min_grads)) - 1
        return normalized_state

    def denormalize_state(self, normalized_state):
        diff = self.max_grads-self.min_grads
#        print('diff:', diff)
        if np.all(diff) == False:
            sys.exit('Problem with the gradients')
        normalized_state = ((normalized_state + 1) / 2) * (self.max_grads - self.min_grads) + self.min_grads
        return normalized_state

    def step(self, action):

        # Scale unit action to proper action space
        #print('action', action)
        denorm_action = self.denormalize_state(action)
        #print('denorm_action', denorm_action)

        self.linac.setGradients(denorm_action)

        # Stateful workflow
        # print('denorm_action', denorm_action)
        #
        # # Get new gradients
        # current_states = self.linac.getGradients()
        # print('current_states', current_states)
        # print('step new state', current_states + denorm_action)
        #
        # # Update gradients
        # self.linac.update_gradients(denorm_action)

        # Get new gradients
        self.states = self.linac.getGradients()
        #print('linac new state', self.states)

        # calc_action = self.states - current_states
        #
        # print('denorm_action', denorm_action)
        # print('calc_action', calc_action)

        # Need to normalize for the RL agent
        normalized_states = self.normalize_state(self.states)

        # Trip
        trip = self.linac.getTripRates()
        trip_reward = 7.5 * (np.exp(trip * 10) - np.exp(0.01 * 10))

        # Heat
        heat = self.linac.getRFHeat()
        heat_reward = (np.exp(heat / 5) - np.exp(20 / 5))

        # Combined reward
        reward = -1.0*(self.alpha*trip_reward + (1-self.alpha)*heat_reward)

        # Energy boundary
        self.energy = self.linac.getEnergyGain()
        # print('New energy: {}({}/{}/{})'.format(self.energy,
        #                                         self.min_energy,
        #                                         self.max_energy,
        #                                         self.target_energy))

        # Apply to all optimization scenarios
        if self.energy < self.min_energy or self.energy > self.max_energy:
            reward -= 100*np.abs(self.energy - self.target_energy)

        # Simple energy reward
        if self.opt == 'energy':
            reward = - np.log(np.abs(self.energy - self.target_energy)) #- 100 * np.square(self.energy - self.target_energy)

        # Extra information
        info = {'heat': self.linac.getRFHeat(),
                'trip': self.linac.getTripRates(),
                'energy': self.energy,
                'alpha': self.alpha}

        #
        #normalized_states = np.append(normalized_states,self.alpha)
        # Return
        return normalized_states, reward, True, True, info

    def reset(self):
        #
        self.alpha = 0# np.random.uniform(0,1)
        if self.rdm_reset_mode == 'circle':
            normalized_states, _, _ = circle_rdm_samples(self.ncavities, 1, 1.0, 0.0, give_all=True)
        if self.rdm_reset_mode == 'uniform':
            normalized_states = self.observation_space.sample()
        self.states = self.denormalize_state(normalized_states)
        self.linac.setGradients(self.states)

        #print(normalized_states.shape)
        #normalized_states = np.append(normalized_states, self.alpha)
        #print(normalized_states.shape)

        return normalized_states, self.states