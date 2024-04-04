from typing import Union

import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import spaces

import numpy as np
import torch
import functorch
import tensorflow as tf
from numpy import ndarray
import sys

class proxy_app(gym.Env):
    def __init__(self, logdir=None):

        self.logdir = logdir
        self.nsteps = 0
        self.devices = 'cpu'
        self.nParameters = 6
        self.parmin = 0
        self.parmax = 1
        self.nevents = 1

        if isinstance(self.parmin, int):
            self.parmin = [self.parmin for i in range(self.nParameters)]
            self.parmin = np.array(self.parmin)
        if isinstance(self.parmax, int):
            self.parmax = [self.parmax for i in range(self.nParameters)]
            self.parmax = np.array(self.parmax)

        self.ndx = 1000
        self.xmin, self.xmax = 0.1, 0.99999
        self.dx = (self.xmax - self.xmin) / self.ndx
        #self.x_full_range = torch.arange(self.xmin, self.xmax, self.dx, device=self.devices)
        self.x_full_range = np.linspace(self.xmin, self.xmax, self.ndx) #self.dx)

        # Define action space
        self.true_params = [0.72916667, 0.25, 0.6, 0.36458333, 0.25, 0.8]
        self.np_parmin = np.array([x * 0.9for x in self.true_params]) #np.zeros(self.nParameters)
        self.np_parmax = np.array([x * 1.1for x in self.true_params]) #np.ones(self.nParameters)

        self.action_space = spaces.Box(low=self.np_parmin, high=self.np_parmax, dtype=np.float64)
        print('action_space:{}'.format(self.action_space))
        self.observation_space = spaces.Box(low=np.zeros(2*self.ndx), high=2*np.ones(2*self.ndx), dtype=np.float64)
        print('observation_space:{}'.format(self.observation_space))

        self.true_sigma1, self.true_sigma2 = self.cross_sections(self.true_params)
        self.true_sigmas = np.concatenate([self.true_sigma1, self.true_sigma2])
        self.states, _ = self.reset()
        print('reset state:{}'.format(self.states))

        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot(111)
        # ax.set_title('true_sigma1')
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # plt.plot(self.true_sigma1)
        # plt.savefig(self.logdir + '/true_sigma1.png')
        # plt.close()
        #
        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot(111)
        # ax.set_title('true_sigma2')
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # plt.plot(self.true_sigma2)
        # plt.savefig(self.logdir + '/true_sigma2.png')
        # plt.close()
        #
        # self.pdist = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        # Best results
        self.best_loss = 9999
        self.best_params = None

    def get_ud(self, p):
        u = p[0] * np.power(self.x_full_range, p[1]) * np.power((1 - self.x_full_range), p[2])
        d = p[3] * np.power(self.x_full_range, p[4]) * np.power((1 - self.x_full_range), p[5])
        return u, d

    def cross_sections(self, parameters):
        # Denormalize the parameters
        parameters = parameters * (self.parmax - self.parmin) + self.parmin

        u_full, d_full = self.get_ud(parameters)
        sigma1 = 4 * u_full + d_full
        sigma2 = 4 * d_full + u_full
        return np.array(sigma1), np.array(sigma2)

    def step(self, action):

        #
        gen_sigma1, gen_sigma2 = self.cross_sections(action)

        # print("gen_sigma1: ", gen_sigma1)
        # print("gen_sigma2: ", gen_sigma2)
        self.states = np.concatenate([gen_sigma1, gen_sigma2])
        # loss1 = np.mean(np.square(self.true_sigma1 - gen_sigma1))
        # loss2 = np.mean(np.square(self.true_sigma2 - gen_sigma2))
        # #loss2 = tf.keras.losses.mse(self.true_sigma2, gen_sigma2)
        # loss = (loss1+loss2)+1e-6
        # # Scale for linear approximation
        # reward = 1000.0*np.exp(-loss)
        r1 = 1000.0*np.exp(-5*np.mean(np.abs(self.true_sigma1-gen_sigma1)))
        r2 = 1000.0*np.exp(-5*np.mean(np.abs(self.true_sigma2-gen_sigma2)))
        reward = r1 + r2 + 1e-6
        #print('reward', reward)

        self.nsteps += 1

        return self.states, reward, False, False, {}
        #

    def reset(self):
        self.states = self.true_sigmas

        return self.states, ''
