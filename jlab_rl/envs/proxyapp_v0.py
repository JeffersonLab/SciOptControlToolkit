from typing import Union

import gym
from gym import spaces

import numpy as np
import torch
import functorch
import tensorflow as tf
from numpy import ndarray


class proxy_app(gym.Env):
    def __init__(self):

        self.nsteps = 0
        self.devices = 'cpu'
        self.nParameters = 6
        self.parmin = 0
        self.parmax = 1
        self.nevents = 1000

        data = np.load('/Users/schram/repositories/jlab_datascience_optimization/jlab_rl/envs/proxyapp_data.pkl.npy', allow_pickle=True)
        self.data = np.transpose(data[0], (1, 0))

        if isinstance(self.parmin, int):
            self.parmin = [self.parmin for i in range(self.nParameters)]
            self.parmin = np.array(self.parmin)
        if isinstance(self.parmax, int):
            self.parmax = [self.parmax for i in range(self.nParameters)]
            self.parmax = np.array(self.parmax)
        self.parmin = torch.as_tensor(self.parmin, device=self.devices)
        self.parmax = torch.as_tensor(self.parmax, device=self.devices)

        self.xmin, self.xmax = 0.1, 0.99999
        self.dx = (self.xmax - self.xmin) / 1000
        self.x_full_range = torch.arange(self.xmin, self.xmax, self.dx, device=self.devices)

        # Define action space
        self.np_parmin = np.zeros(self.nParameters)
        self.np_parmax = np.ones(self.nParameters)

        self.action_space = spaces.Box(low=self.np_parmin, high=self.np_parmax, dtype=np.float32)
        print('action_space:{}'.format(self.action_space))
        self.observation_space = spaces.Box(low=np.zeros(self.nParameters), high=np.ones(self.nParameters), dtype=np.float32)
        print('observation_space:{}'.format(self.observation_space))
        #self.inital_states = self.observation_space.sample()
        self.states, _ = self.reset()
        print('reset state:{}'.format(self.states))

    def get_ud(self, p):
        u = p[0] * torch.pow(self.x_full_range, p[1]) * torch.pow((1 - self.x_full_range), p[2])
        d = p[3] * torch.pow(self.x_full_range, p[4]) * torch.pow((1 - self.x_full_range), p[5])
        return u, d

    def integral_approx(self, y):
        return torch.sum(y[:-1]) * self.dx

    def torch_interp(self, x, xs, ys):
        # determine the output data type
        ys = torch.Tensor(ys)
        dtype = ys.dtype

        # We need to use float32 for the macbook mps...
        ys = ys.type(torch.float32).to(self.devices)
        xs = xs.type(torch.float32).to(self.devices)
        x = x.type(torch.float32).to(self.devices)

        # pad control points for extrapolation
        xs = torch.cat([torch.tensor([torch.finfo(xs.dtype).min]).to(self.devices)
                           , xs, torch.tensor([torch.finfo(xs.dtype).max]).to(self.devices)
                        ], axis=0)
        ys = torch.cat([ys[:1], ys, ys[-1:]], axis=0)

        # compute slopes, pad at the edges to flatten
        ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        ms = torch.nn.functional.pad(ms[:-1], (1, 1))

        # solve for intercepts
        bs = ys - ms * xs

        # search for the line parameters at each input data point
        # create a grid of the inputs and piece breakpoints for thresholding
        # rely on argmax stopping on the first true when there are duplicates,
        # which gives us an index into the parameter vectors

        # Argmax not implemented for boolean on CPU
        double_from_bool = xs[..., None, :] > x[..., None]
        # double_from_bool = double_from_bool.double()
        double_from_bool = double_from_bool.float()  # --> Need to use float for macbook mps...
        i = torch.argmax(double_from_bool, dim=-1)
        m = ms[..., i]
        b = bs[..., i]

        # apply the linear mapping at each input data point
        y = m * x + b

        return torch.reshape(y, x.shape).type(dtype)

    def inverse_cdf(self, cdf_allx1, nevents):
        u = torch.rand((1 * nevents,)) * 0.9999
        cdf_sort_indx1 = torch.argsort(cdf_allx1)
        cdf_sort1 = cdf_allx1[cdf_sort_indx1]
        x_sort1 = self.x_full_range[cdf_sort_indx1]

        events_out1 = self.torch_interp(u, cdf_sort1, x_sort1)
        events_out1 = torch.reshape(events_out1, (nevents,))
        return events_out1

    def gen_events(self, true_params, nevents):
        # Denormalize the parameters
        true_params = torch.as_tensor(true_params, device=self.devices)
        true_params = true_params * (self.parmax - self.parmin) + self.parmin

        u_full, d_full = self.get_ud(true_params)
        sigma1 = 4 * u_full + d_full
        sigma2 = 4 * d_full + u_full

        norm1 = self.integral_approx(sigma1)
        norm2 = self.integral_approx(sigma2)

        u, d = self.get_ud(true_params)

        pdf1 = ((4 * u) + d) / norm1
        pdf2 = ((4 * d) + u) / norm2

        inv_indices = torch.arange(pdf1.size(0) - 1, -1, -1, device=self.devices).long()
        inv_pdf1 = pdf1.index_select(0, inv_indices)

        inv_indices = torch.arange(pdf2.size(0) - 1, -1, -1, device=self.devices).long()
        inv_pdf2 = pdf2.index_select(0, inv_indices)

        inv_cdf_allx1 = (torch.cumsum(inv_pdf1 * self.dx, dim=0) / torch.sum(pdf1 * self.dx))
        inv_cdf_allx2 = (torch.cumsum(inv_pdf2 * self.dx, dim=0) / torch.sum(pdf2 * self.dx))

        indices = torch.arange(inv_cdf_allx1.size(0) - 1, -1, -1, device=self.devices).long()
        cdf_allx1 = inv_cdf_allx1.index_select(0, indices)

        indices = torch.arange(inv_cdf_allx2.size(0) - 1, -1, -1, device=self.devices).long()
        cdf_allx2 = inv_cdf_allx2.index_select(0, indices)

        events1 = self.inverse_cdf(cdf_allx1, nevents)
        events2 = self.inverse_cdf(cdf_allx2, nevents)

        events = torch.cat([torch.unsqueeze(events1, 0), torch.unsqueeze(events2, 0)], dim=0).to(self.devices)

        return events, norm1, norm2

    def paramsToEventsMap(self, params, nevents):
        return functorch.vmap(lambda x: self.gen_events(x, nevents), in_dims=0, randomness="different")(params)

    def forward(self, params, nevents=1):
        return self.paramsToEventsMap(params, nevents)

    def score_es(self, YPred, YObs):
        obs_event, obs_size = YObs.shape
        pred_event, pred_size = YPred.shape
        assert obs_event == pred_event, "Observations and events have different sizes"

        es1 = np.zeros((obs_size,))
        for iObs in range(obs_size):
            es = np.mean(np.linalg.norm(YPred - YObs[:, iObs, np.newaxis], ord=2, axis=0))
            es1[iObs] = es

        score1 = np.mean(es1)

        pairwise_distances = np.linalg.norm(YPred[:, :, np.newaxis] - YPred[:, np.newaxis, :], ord=2, axis=0)
        score2 = np.sum(pairwise_distances) / (2 * pred_size * (pred_size - 1))

        return score1 - score2

    def compute_loss(self, x, x_pred, x_ref):

        x = torch.Tensor(x)
        x_pred = torch.Tensor(x_pred)
        x_ref = torch.Tensor(x_ref)
        act_loss = torch.square(x - x_pred)
        ref_loss = torch.square(x - x_ref)

        loss = torch.abs(act_loss - ref_loss)
        return torch.mean(loss)

    def step(self, action):

        # Update the model parameters
        # print('state:{}'.format(self.states))
        # print('action:{}'.format(action))
        self.states = self.states + action
        self.states = action#self.states + action
        nout=0
        for i in range(self.states.shape[0]):
            if self.states[i] < 0. or self.states[i] > 1:
                nout +=1
        #in_range = ((self.states >= -0.1) & (self.states <= 1.1)).all()
        if nout>0:
            return self.states, -100*nout, True, True, 'Error'
        #
        # print('state:{}'.format(self.states))
        # print('in range:{}'.format(in_range))
        # print('updated state:{}'.format(self.states))
        # print('updated state:{}'.format(self.states.shape))
        # print('nevents:{}'.format(self.nevents))
        #policy_data = self.forward(self.states, self.nevents)
        policy_data, norm1, norm2 = self.gen_events(self.states, self.nevents)
        policy_data = np.transpose(policy_data.numpy(), (1, 0))

        # print('policy_data:{}'.format(policy_data.shape))
        # print('data:{}'.format(self.data.shape))
        real_data = self.data[torch.randint(self.data.shape[0], (self.nevents,))]
        ref_data = self.data[torch.randint(self.data.shape[0], (self.nevents,))]
        # print('real_data:{}'.format(real_data.shape))
        # print('ref_data:{}'.format(ref_data.shape))
        loss = np.abs(self.compute_loss(real_data, policy_data, ref_data))
        # Find a cleaver reward
        reward = 1/loss#-np.log(loss)
        for i in range(self.nParameters):
            tf.summary.scalar('Parameter #{}'.format(i), data=self.states[i], step=int(self.nsteps))
        self.nsteps += 1
        return self.states, reward, False, False, {}
        #

    def reset(self):
        # Randomize the parameters
        self.states = np.ones(self.nParameters)*0.5
        return self.states, ''
