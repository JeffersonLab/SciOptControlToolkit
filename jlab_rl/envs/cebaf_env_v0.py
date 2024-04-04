# Author: Kishansingh Rajput, Malachi Schram
# Script: CEBAF cavities digital twin
# Org: Thomas Jefferson National Accelerator Facility

import numpy as np
import pandas as pd
import math
from gymnasium import spaces
import gymnasium as gym
import os, sys

from jlab_rl.envs.cebaf_surrogate_v0 import digitalTwin
from jlab_rl.utils.circle_rdm import circle_rdm_samples


class cebaf_env(gym.Env):
    def __init__(self, path_cavity_data=os.path.join(os.path.dirname(__file__), 'updated_cavity_table.pkl'),
                 linac="North", rdm_reset_mode='circle', objective='heat', loss_type='test_nonlinear', seed=22):
        np.random.seed(seed=seed)

        self.rdm_reset_mode = rdm_reset_mode
        self.objective = objective
        self.loss_type = loss_type
        self.alpha = None

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
        print('Initial state:', self.states)
        print('Initial Energy: {}'.format(self.energy))

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

    def get_objective(self):
        if self.objective == 'heat':
            self.alpha = -1
        elif self.objective == 'trip':
            self.alpha = 1
        elif self.objective == 'mixed':
            self.alpha = 0.5
        else:
            self.alpha = np.random.uniform(0,1)

    def step(self, action):

        # Scale unit action to proper action space
        denorm_action = self.denormalize_state(action)

        self.linac.setGradients(denorm_action)

        # Get new gradients
        self.states = self.linac.getGradients()


        # Need to normalize for the RL agent
        normalized_states = self.normalize_state(self.states)

        # Trip
        trip = self.linac.getTripRates()
        if self.loss_type=='nonlinear':
            trip_reward = -1.0*(7.5 * (np.exp(trip * 10) - np.exp(0.01 * 10)))
        else:
            trip_reward = 0.05-trip
            trip_reward = -9 + trip_reward if trip_reward < 0.0 else trip_reward

        # Heat
        heat = self.linac.getRFHeat()
        if self.loss_type=='nonlinear':
            heat_reward = -(np.exp(heat / 5) - np.exp(20 / 5))
        else:
            heat_reward = 25-heat
            heat_reward = -9+heat_reward if heat_reward < 0.0 else heat_reward

        # Combined reward
        #reward = -1.0*(self.alpha*trip_reward + (1-self.alpha)*heat_reward)
        reward = (self.alpha*trip_reward + (1-self.alpha)*heat_reward)

        # Energy boundary
        self.energy = self.linac.getEnergyGain()

        # Apply to all optimization scenarios
        isValid = True
        if self.energy < self.min_energy or self.energy > self.max_energy:
            reward -= 100*np.abs(self.energy - self.target_energy)
            isValid = False
        # Simple energy reward
        if self.objective == 'energy':
            reward = - np.log(np.abs(self.energy - self.target_energy))

        # Extra information
        info = {'heat': self.linac.getRFHeat(),
                'trip': self.linac.getTripRates(),
                'energy': self.energy,
                'alpha': self.alpha,
                'valid': isValid}

        # Return
        return normalized_states, reward, True, True, info

    def reset(self):
        #
        self.get_objective()
        #self.alpha = 0.5
        if self.rdm_reset_mode == 'fixed':
            normalized_states = np.zeros(self.ncavities)
        if self.rdm_reset_mode == 'circle':
            normalized_states, _, _ = circle_rdm_samples(self.ncavities, 1, 1.0, 0.0, give_all=True)
        if self.rdm_reset_mode == 'uniform':
            normalized_states = self.observation_space.sample()
        #
        self.states = self.denormalize_state(normalized_states)
        self.linac.setGradients(self.states)

        return normalized_states, self.alpha