import numpy as np
import jlab_rl.envs as gym
env = gym.make('Circle2DEnv-v0')
print(env.reset())
empty_action = np.array([0,0])
env.step(empty_action)