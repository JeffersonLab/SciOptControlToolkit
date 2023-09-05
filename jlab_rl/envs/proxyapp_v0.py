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
        #self.parmin = torch.as_tensor(self.parmin, device=self.devices)
        #self.parmax = torch.as_tensor(self.parmax, device=self.devices)

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

    # def integral_approx(self, y):
    #     return torch.sum(y[:-1]) * self.dx

    # def torch_interp(self, x, xs, ys):
    #     # determine the output data type
    #     ys = torch.Tensor(ys)
    #     dtype = ys.dtype
    #
    #     # We need to use float32 for the macbook mps...
    #     ys = ys.type(torch.float32).to(self.devices)
    #     xs = xs.type(torch.float32).to(self.devices)
    #     x = x.type(torch.float32).to(self.devices)
    #
    #     # pad control points for extrapolation
    #     xs = torch.cat([torch.tensor([torch.finfo(xs.dtype).min]).to(self.devices)
    #                        , xs, torch.tensor([torch.finfo(xs.dtype).max]).to(self.devices)
    #                     ], axis=0)
    #     ys = torch.cat([ys[:1], ys, ys[-1:]], axis=0)
    #
    #     # compute slopes, pad at the edges to flatten
    #     ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    #     ms = torch.nn.functional.pad(ms[:-1], (1, 1))
    #
    #     # solve for intercepts
    #     bs = ys - ms * xs
    #
    #     # search for the line parameters at each input data point
    #     # create a grid of the inputs and piece breakpoints for thresholding
    #     # rely on argmax stopping on the first true when there are duplicates,
    #     # which gives us an index into the parameter vectors
    #
    #     # Argmax not implemented for boolean on CPU
    #     double_from_bool = xs[..., None, :] > x[..., None]
    #     # double_from_bool = double_from_bool.double()
    #     double_from_bool = double_from_bool.float()  # --> Need to use float for macbook mps...
    #     i = torch.argmax(double_from_bool, dim=-1)
    #     m = ms[..., i]
    #     b = bs[..., i]
    #
    #     # apply the linear mapping at each input data point
    #     y = m * x + b
    #
    #     return torch.reshape(y, x.shape).type(dtype)

    # def inverse_cdf(self, cdf_allx1, nevents):
    #     u = torch.rand((1 * nevents,)) * 0.9999
    #     cdf_sort_indx1 = torch.argsort(cdf_allx1)
    #     cdf_sort1 = cdf_allx1[cdf_sort_indx1]
    #     x_sort1 = self.x_full_range[cdf_sort_indx1]
    #
    #     events_out1 = self.torch_interp(u, cdf_sort1, x_sort1)
    #     events_out1 = torch.reshape(events_out1, (nevents,))
    #     return events_out1

    def cross_sections(self, parameters):
        # Denormalize the parameters
        #parameters = torch.as_tensor(parameters, device=self.devices)
        parameters = parameters * (self.parmax - self.parmin) + self.parmin

        u_full, d_full = self.get_ud(parameters)
        sigma1 = 4 * u_full + d_full
        sigma2 = 4 * d_full + u_full
        return np.array(sigma1), np.array(sigma2)

    # def gen_events(self, true_params, nevents):
    #     # Denormalize the parameters
    #     true_params = torch.as_tensor(true_params, device=self.devices)
    #     true_params = true_params * (self.parmax - self.parmin) + self.parmin
    #
    #     u_full, d_full = self.get_ud(true_params)
    #     sigma1 = 4 * u_full + d_full
    #     sigma2 = 4 * d_full + u_full
    #
    #     norm1 = self.integral_approx(sigma1)
    #     norm2 = self.integral_approx(sigma2)
    #
    #     u, d = self.get_ud(true_params)
    #
    #     pdf1 = ((4 * u) + d) / norm1
    #     pdf2 = ((4 * d) + u) / norm2
    #
    #     inv_indices = torch.arange(pdf1.size(0) - 1, -1, -1, device=self.devices).long()
    #     inv_pdf1 = pdf1.index_select(0, inv_indices)
    #
    #     inv_indices = torch.arange(pdf2.size(0) - 1, -1, -1, device=self.devices).long()
    #     inv_pdf2 = pdf2.index_select(0, inv_indices)
    #
    #     inv_cdf_allx1 = (torch.cumsum(inv_pdf1 * self.dx, dim=0) / torch.sum(pdf1 * self.dx))
    #     inv_cdf_allx2 = (torch.cumsum(inv_pdf2 * self.dx, dim=0) / torch.sum(pdf2 * self.dx))
    #
    #     indices = torch.arange(inv_cdf_allx1.size(0) - 1, -1, -1, device=self.devices).long()
    #     cdf_allx1 = inv_cdf_allx1.index_select(0, indices)
    #
    #     indices = torch.arange(inv_cdf_allx2.size(0) - 1, -1, -1, device=self.devices).long()
    #     cdf_allx2 = inv_cdf_allx2.index_select(0, indices)
    #
    #     events1 = self.inverse_cdf(cdf_allx1, nevents)
    #     events2 = self.inverse_cdf(cdf_allx2, nevents)
    #
    #     events = torch.cat([torch.unsqueeze(events1, 0), torch.unsqueeze(events2, 0)], dim=0).to(self.devices)
    #
    #     return events, norm1, norm2

    # def paramsToEventsMap(self, params, nevents):
    #     return functorch.vmap(lambda x: self.gen_events(x, nevents), in_dims=0, randomness="different")(params)
    #
    # def forward(self, params, nevents=1):
    #     return self.paramsToEventsMap(params, nevents)

#    def score_es(self, YPred, YObs):
#        obs_event, obs_size = YObs.shape
#        pred_event, pred_size = YPred.shape
#        assert obs_event == pred_event, "Observations and events have different sizes"
#
#        es1 = np.zeros((obs_size,))
#        for iObs in range(obs_size):
#            es = np.mean(np.linalg.norm(YPred - YObs[:, iObs, np.newaxis], ord=2, axis=0))
#            es1[iObs] = es
#
#        score1 = np.mean(es1)
#
#        pairwise_distances = np.linalg.norm(YPred[:, :, np.newaxis] - YPred[:, np.newaxis, :], ord=2, axis=0)
#        score2 = np.sum(pairwise_distances) / (2 * pred_size * (pred_size - 1))
#
#        return score1 - score2

    # def compute_default_loss(self, x, x_pred, x_ref):
    #
    #     x = torch.Tensor(x)
    #     x_pred = torch.Tensor(x_pred)
    #     x_ref = torch.Tensor(x_ref)
    #     act_loss = torch.square(x - x_pred)
    #     ref_loss = torch.square(x - x_ref)
    #
    #     loss = torch.abs(act_loss - ref_loss)
    #     return torch.mean(loss)

    # def compute_emil_loss(self,y,y_pred):
    #     # Determine score 1, i.e. compare the predictions and true values:
    #     score_1 = torch.mean(torch.cdist(y_pred,y))
    #
    #     # Determine score 2, i.e. compare predictions amongst each other and take care of
    #     # 'false' combinations:
    #     score_2 = torch.sum(self.pdist(y_pred,y_pred)) / float(y_pred.size()[0] * (y_pred.size()[0] -1 ))
    #
    #     return score_1 - score_2

    def step(self, action):

        #self.states = self.states + action
        # print('reset state:{}'.format(self.states.shape))
        # sys.exit()

        # nout=0
        # for i in range(self.states.shape[0]):
        #     if self.states[i] < 0. or self.states[i] > 1:
        #         nout +=1
        # if nout>0:
        #     return self.states, -100*nout, True, True, 'Error'

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
        # Randomize the parameters
        #self.states = np.zeros(self.nParameters)*0.5
        # print('self.true_sigma1', self.true_sigma1)
        # print('self.true_sigma1', type(self.true_sigma1))
        # print('self.true_sigma2', self.true_sigma2)
        # print('self.true_sigma2', type(self.true_sigma2))
        #print('self.true_cat', true_sigmas)
        self.states = self.true_sigmas#,1np.random.normal()

        return self.states, ''
