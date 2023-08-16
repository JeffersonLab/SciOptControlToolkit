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
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

# Seed value
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

import gym
import jlab_rl.envs as jlab_envs

def get_env(env_id):
    try:
        env = gym.make(env_id)
        return env
    except:
        print('Non-standard Gym Environment. Trying JLab Environments...')
        try:
            env = jlab_envs.make(env_id)
            return env
        except:
            raise Exception(f'Failed to load environment {env_id}')

def run_opt(index, max_nepisodes, max_nsteps, agent_id, warmup_size, env_id, logdir):
    print('Running env: {}'.format(env_id))

    env = get_env(env_id)
    #
    # Environment
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
    nsavefig = agent.batch_size

    outer_pbar = tqdm(range(max_nepisodes), desc='Index {} - Episodes'.format(index))
    inner_pbar = tqdm(range(int(max_nsteps)), desc='Index {} - Steps'.format(index), leave=False)
    nep = 0
    for ep in outer_pbar:
        nep += 1
        time_start = time.process_time()
        prev_state, alpha = env.reset()
        # Define the objection fraction
        obj_frac = [alpha, 1.0-alpha]
        obj_frac = [1, 0]
        nsteps = 0
        episodic_reward = 0
        for estep in range(int(max_nsteps)):
            total_nsteps += 1

            action, noise = agent.action(obj_frac, tf.convert_to_tensor(prev_state))
            if np.isnan(noise).any():
                print('action:', action)
                sys.exit(-11)

            # Receive state and reward from environment.
            if env_id != 'Pendulum-v1':
                action = np.squeeze(action)
            state, reward, done_old, done, info = env.step(action)
            agent.memory((prev_state, action, reward, state, done, obj_frac))
            episodic_reward += reward
            agent.train()
            prev_state = state

            if "CEBAF" in env_id:
                tf.summary.scalar('Energy Distribution', data=env.energy, step=int(total_nsteps))
                tf.summary.scalar('Trip Rate', data=info['trip'], step=int(total_nsteps))
                tf.summary.scalar('Heat Load', data=info['heat'], step=int(total_nsteps))

            # Plot Pareto front
            if nep%100==0:
                print('Testing Pareto Front...')
                fig = plt.figure(figsize=(12, 12))
                test_scan_fraction = np.linspace(0, 1, 200)
                test_env = get_env(env_id)
                test_heats, test_trips = [], []
                for test_fraction in test_scan_fraction:
                    test_prev_state, _ = test_env.reset()
                    test_env.alpha = test_fraction
                    test_obj_frac = np.array([test_fraction, 1.0-test_fraction])
                    test_action, _ = agent.action(test_obj_frac, tf.convert_to_tensor(test_prev_state))
                    test_action = np.squeeze(test_action)
                    state, reward, done_old, done, info = test_env.step(test_action)
                    test_heat = info['heat']
                    test_trip = info['trip']
                    test_heats.append(test_heat)
                    test_trips.append(test_trip)

                plt.plot(test_heats, test_trips, 'o')
                plt.savefig(logdir + '/pareto_{}.png'.format(nep))


            # End this episode when `done` is True
            if done_old:
                break

            # End this episode when `done` is True
            if done:
                break



            # fig = plt.figure(figsize=(6, 6))
            # ax = fig.add_subplot(111)
            # ax.set_title('Action - {}/{}'.format(agent_id, env_id), fontsize=14)
            # ax.set_xlabel("X", fontsize=12)
            # ax.set_ylabel("Y", fontsize=12)
            # ax.grid(True, linestyle='-', color='0.75')
            # this_actions = agent.action_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
            # this_actions = env.denormalize_state(this_actions)
            # x = this_actions[:, 0]
            # y = this_actions[:, 1]
            # # scatter with colormap mapping to z value
            # cb = ax.scatter(x, y, s=20, marker='o');
            # plt.xlim(np.min(x), np.max(x))
            # plt.ylim(np.min(y), np.max(y))
            # plt.colorbar(cb)
            # plt.savefig(logdir + '/denormalized_action_{}.png'.format(agent.buffer_counter / nsavefig))
            inner_pbar.update()
        
        inner_pbar.refresh()

        ep_reward_list.append(episodic_reward)
        tf.summary.scalar('Reward', data=episodic_reward, step=int(ep))

        # Mean of last 40 episodes
        nepisode_mod = 10
        avg_reward = np.mean(ep_reward_list[-nepisode_mod:])
        time_end = time.process_time()
        avg_reward_list.append(avg_reward)
        inner_pbar.reset()
        outer_pbar.set_postfix({'Avg Reward': avg_reward, 'Ep Reward': episodic_reward})
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index for tracking", type=int, default=0)
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=10000)
    parser.add_argument("--nsteps", help="Number of steps", type=int, default=1)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasMultiObjTD3-v0')
    parser.add_argument("--nwarmup", help="Agent warm-up size", type=int, default=0)
    parser.add_argument("--env", help="Environment used for RL", type=str, default='MultObj-CEBAF8DEnv-v0')
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
