import gym
from gym import spaces
import numpy as np
import tensorflow as tf
import torch
import functorch
from torch.utils.tensorboard import SummaryWriter
# import tomography_toolkit_dev.sampler_module as samplers
from scipy.spatial.distance import pdist, cdist
import matplotlib.pyplot as plt

def gaussian_sampler(parameters, parmin, parmax, n):
    mean = (parameters * (parmax[0] - parmin[0])) + parmin[0]
    std = 0.2 #(parameters[1] * (parmax[1] - parmin[1])) + parmin[1]
    d = np.random.normal(mean, std, n)
    return d


def torch_score_es(y,y_pred):
    # Determine score 1, i.e. compare the predictions and true values:
    print(y_pred.shape, y.shape)
    score_1 = np.mean(cdist(y_pred.reshape(-1,1),y.reshape(-1,1)))

    # Determine score 2, i.e. compare predictions amongst each other and take care of
    # 'false' combinations:
    score_2 = np.sum(pdist(y_pred.reshape(-1,1))) / float(y_pred.shape[0] * (y_pred.shape[0] -1 ))

    return score_1 - score_2

class ProxyGaussian(gym.Env):
    def __init__(self):

        data_path = "../data/gaussian_events.npy"
        true_param_path = "../data/true_mean_dist.npy"
        self.devices = 'cpu'
        self.theory = gaussian_sampler
        # self.sampler = samplers.make(module_names['sampler'], config=sampler_config, devices=self.devices)
        
        self.loss_func = torch_score_es
        #TODO: We can use a loss registry if we have multiple losses implementation!
        # self.loss = losses.make(module_names['loss'], config=loss_config, devices=self.devices)

        self.nsteps = 0
        self.nevents = 10000
        self.nParameters = 1
        self.parmin = [0.]
        self.parmax = [1.]

        self.data = np.load(data_path, allow_pickle=True)
        self.true_params = np.load(true_param_path, allow_pickle=True)
        # self.data = np.transpose(data[0], (1, 0))


        # self.action_space = spaces.Box(low=self.theory.parmin.numpy(), high=self.theory.parmax.numpy(), dtype=np.float32)
        self.action_space = spaces.Box(low=np.zeros(self.nParameters), high=np.ones(self.nParameters), dtype=np.float32)
        print('action_space:{}'.format(self.action_space))
        self.observation_space = spaces.Box(low=np.zeros(self.nParameters), high=np.ones(self.nParameters), dtype=np.float32)
        print('observation_space:{}'.format(self.observation_space))
        
        self.noise_dim = 1

        self.writer = SummaryWriter()

        self.states, _ = self.reset()
        print('reset state:{}'.format(self.states))
        self.counter = 0

    def step(self, action):
        """
        
        """
        self.states = action
        input_states = torch.tensor(np.expand_dims(self.states, 0))
        policy_data = self.theory(input_states, self.parmin, self.parmax, self.nevents)
        policy_data = np.squeeze(policy_data)
        # policy_data = np.transpose(policy_data, (1, 0))

        real_data = self.data[torch.randint(self.data.shape[0], (self.nevents,))]
        # ref_data = self.data[torch.randint(self.data.shape[0], (self.nevents,))]
        
        loss = np.abs(self.loss_func(real_data, policy_data))

        if self.counter % 1000 == 0:
            plt.clf()
            plt.figure(figsize=(8,5))
            plt.hist(real_data, bins=100, histtype='step', color ='green', label="real_data")
            plt.hist(policy_data, bins=100, histtype='step', color ='red', label="generated_data")
            plt.title('data_at_epoch'+str(self.counter).zfill(6))
            plt.savefig("data_"+str(self.counter).zfill(6)+".png")
            plt.legend()
        self.counter += 1
        
        # Find a cleaver reward
        reward = 1/loss #-np.log(loss)
        # for i in range(self.nParameters):
        #     self.writer.add_scalar('Parameter #{}'.format(i), self.states[i], int(self.nsteps))
        self.nsteps += 1
        
        return self.states, reward, True, False, {}
        

    def reset(self):
        # Randomize the parameters
        self.states = np.random.normal(self.observation_space.low, self.observation_space.high, (self.noise_dim))
        return self.states, ''