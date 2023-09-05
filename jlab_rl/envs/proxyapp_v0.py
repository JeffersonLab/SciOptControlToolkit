from typing import Union

import gym
import matplotlib.pyplot as plt
from gym import spaces

import numpy as np
import torch
import functorch
import tensorflow as tf
from numpy import ndarray
import sys

class proxy_app(gym.Env):
    def __init__(self,loss_type='default'):

        self.logdir = 'results/'
        self.nsteps = 0
        self.devices = 'cpu'
        self.nParameters = 6
        self.parmin = 0
        self.parmax = 1
        self.nevents = 1
        self.loss_type = loss_type

        if isinstance(self.parmin, int):
            self.parmin = [self.parmin for i in range(self.nParameters)]
            self.parmin = np.array(self.parmin)
        if isinstance(self.parmax, int):
            self.parmax = [self.parmax for i in range(self.nParameters)]
            self.parmax = np.array(self.parmax)

        self.xmin, self.xmax = 0.1, 0.99999
        self.dx = (self.xmax - self.xmin) / 10
        self.x_full_range = torch.arange(self.xmin, self.xmax, self.dx, device=self.devices)

        # Define action space
        self.np_parmin = np.zeros(self.nParameters)
        self.np_parmax = np.ones(self.nParameters)

        self.action_space = spaces.Box(low=self.np_parmin, high=self.np_parmax, dtype=np.float32)
        print('action_space:{}'.format(self.action_space))
#        self.observation_space = spaces.Box(low=np.zeros(self.nParameters), high=np.ones(self.nParameters), dtype=np.float32)
#        print('observation_space:{}'.format(self.observation_space))
        self.observation_space = spaces.Box(low=np.zeros(20), high=10*np.ones(20), dtype=np.float32)
        print('observation_space:{}'.format(self.observation_space))
        #self.inital_states = self.observation_space.sample()
        self.true_params = [0.72916667, 0.25, 0.6, 0.36458333, 0.25, 0.8]
        self.true_sigma1, self.true_sigma2 = self.cross_sections(self.true_params)
        self.true_sigmas = np.concatenate([self.true_sigma1, self.true_sigma2])
        self.states, _ = self.reset()
        print('reset state:{}'.format(self.states))

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_title('true_sigma1')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.plot(self.true_sigma1)
        plt.savefig(self.logdir + 'true_sigma1.png')
        plt.close()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_title('true_sigma2')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.plot(self.true_sigma2)
        plt.savefig(self.logdir + 'true_sigma2.png')
        plt.close()

        self.pdist = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

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
        self.states = np.concatenate([gen_sigma1, gen_sigma2])
        loss1 = tf.keras.losses.mse(gen_sigma1, self.true_sigma1)
        loss2 = tf.keras.losses.mse(gen_sigma2, self.true_sigma2)
        reward = -(loss1+loss2)

        if self.nsteps%100==0:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f'sigma1 -loss: {loss1}')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.plot(self.true_sigma1)
            plt.plot(gen_sigma1)
            plt.savefig(self.logdir+'sigma1_{}.png'.format(self.nsteps))
            plt.close()
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f'sigma2 -loss: {loss2}')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.plot(self.true_sigma2)
            plt.plot(gen_sigma2)
            plt.savefig(self.logdir+'sigma2_{}.png'.format(self.nsteps))
            plt.close()

        for i in range(self.true_sigma1.shape[0]):
            tf.summary.scalar('Xsec-1 Diff #{}'.format(i),
                              data=abs(gen_sigma1[i]-self.true_sigma1[i]),
                              step=int(self.nsteps))
            tf.summary.scalar('Xsec-2 Diff #{}'.format(i),
                              data=abs(gen_sigma2[i] - self.true_sigma2[i]),
                              step=int(self.nsteps))
        self.nsteps += 1

        return self.states, reward, False, False, {}
        #

    def reset(self):
        self.states = self.true_sigmas

        return self.states, ''
