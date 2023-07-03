import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import torch
import jlab_rl.agents
from jlab_rl.utils.git_utilts import get_git_revision_short_hash
from tqdm import tqdm

# import mujoco_py
# import os
# mj_path = mujoco_py.utils.discover_mujoco()
# print('mj_path:{}'.format(mj_path))
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')


# Seed value
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def run_opt(index, max_nepisodes, max_nsteps, agent_id, warmup_size, env_id, logdir):
    if env_id == 'ProxyApp-v0' or 'Circle2DEnv-v0':
        import jlab_rl.envs as gym
    else:
        import gym
    #
    # Environment
    print('Running env: {}'.format(env_id))
    env = gym.make(env_id)
    env._max_episode_steps = max_nsteps

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    githash = get_git_revision_short_hash()
    print(githash)
    print(logdir)
    if logdir == 'None':
        logdir = "./results/index" + str(index) + "_agent_" + agent_id + "_env_" + env_id + "_hash" \
                 + githash + "_results_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        logdir = logdir + "/index" + str(index) + "_agent_" + agent_id + "_env_" + env_id + "_date_" \
                 + datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        os.mkdir(logdir)
    except OSError as error:
        print('Error:', error)
    file_writer = tf.summary.create_file_writer(logdir + '/metrics')
    file_writer.set_as_default()

    # Agent
    agent = jlab_rl.agents.make(agent_id, env=env, warmup_size=warmup_size, logdir=logdir)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    total_nsteps = 0

    for ep in tqdm(range(max_nepisodes), desc='Index {} - Episodes'.format(index)):
        time_start = time.process_time()
        prev_state, _ = env.reset()
        nsteps = 0
        episodic_reward = 0
        for _ in tqdm(range(int(max_nsteps)), desc='Index {} - Steps'.format(index)):
            total_nsteps += 1
            if 'Torch' in agent_id:
                tf_prev_state = torch.Tensor([prev_state])
                action = agent.action(tf_prev_state)
            else:
                action, noise = agent.action(tf.convert_to_tensor(prev_state))
                if np.isnan(noise).any():
                    print('action:', action)
                    sys.exit(-11)
                # TODO: We suspect this is to the the num_actions > 1
                if env_id == "LunarLanderContinuous-v2":
                    action = action[0]
                    # noise = noise[0]

            # Receive state and reward from environment.
            if env_id != 'Pendulum-v1':
                action = np.squeeze(action)
            state, reward, done_old, done, info = env.step(action)
            #print('main loop - reward: {}'.format(reward))
            nsteps += 1
            agent.memory((prev_state, action, reward, state))
            episodic_reward += reward
            agent.train()
            prev_state = state

            # Save information
            tf.summary.scalar('Step Reward', data=episodic_reward, step=int(total_nsteps))

            #
            if env_id == 'Circle2DEnv-v0':
                radius = np.sqrt(state[0]*state[0]+state[1]*state[1])
                tf.summary.scalar('Radial Distribution', data=radius, step=int(total_nsteps))

            # End this episode when `done` is True
            if done_old:
                break

            # End this episode when `done` is True
            if done:
                break

        ep_reward_list.append(episodic_reward)
        tf.summary.scalar('Reward', data=episodic_reward, step=int(ep))

        # Mean of last 40 episodes
        nepisode_mod = 10
        avg_reward = np.mean(ep_reward_list[-nepisode_mod:])
        time_end = time.process_time()
        print("\nEpisode Elapsed Time {}".format((time_end - time_start)))
        print("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index for tracking", type=int, default=0)
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=100)
    parser.add_argument("--nsteps", help="Number of steps", type=int, default=200)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasTD3-v0')
    parser.add_argument("--nwarmup", help="Agent warm-up size", type=int, default=0)
    parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')#HalfCheetah-v4')
    parser.add_argument("--logdir", help="Directory to save results", type=str, default='None')

    # Get input arguments
    args = parser.parse_args()
    args_index = args.index
    args_nepisodes = args.nepisodes
    args_nsteps = args.nsteps
    args_agent_id = args.agent
    args_warmup_size = args.nwarmup
    args_env_id = args.env
    args_logdir = args.logdir

    # Print input settings
    run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id, args_warmup_size, args_env_id, args_logdir)
