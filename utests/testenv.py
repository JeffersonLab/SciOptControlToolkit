import jlab_rl.envs as gym
env = gym.make('ProxyApp-v0')
print(env.reset())

import numpy as np
data = np.load('events_data.pkl.npy', allow_pickle=True)
print(data.shape)
print(data[0].shape)

data_t = np.transpose(data[0], (1, 0))
print(data_t.shape)
# print(data_t[0])
