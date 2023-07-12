import numpy as np
import jlab_rl.envs as gym

env_id = 'CEBAF2DEnv-v0'
env = gym.make(env_id)
linac = env.linac
print(env.reset())
reset_states = np.array([env.reset() for _ in range(2)])
print('reset states:\n', reset_states)
normalized = reset_states[:,0]
raw = reset_states[:,1]
print('normalized states:\n', normalized)
print('raw states:\n', raw)
