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

# import warnings
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore",category=DeprecationWarning)
import gym
import jlab_rl.envs as jlab_envs

def run_opt(
        index,
        max_episodes,
        max_steps,
        steps_per_episode,
        agent_id, 
        warmup_size,
        env_id, 
        logdir
    ):

    print('Running env: {}'.format(env_id))

    try:
        env = gym.make(env_id)
    except:
        print('Non-standard Gym Environment. Trying JLab Environments...')
        try:
            env = jlab_envs.make(env_id)
        except:
            raise Exception(f'Failed to load environment {env_id}')
    #
    # Environment
    env._max_episode_steps = steps_per_episode


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
    # To store all rewards
    step_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    # To store actions
    action_list = []
    # To store steps per epsiode
    ep_steps_list = []

    total_steps = 0
    nsavefig = agent.batch_size

    pbar = tqdm(range(max_steps), desc='Index {} - Steps'.format(index))
    
    ## Reusing max_nepisodes for total steps
    total_episodes = 0
    
    while total_steps < max_steps and total_episodes < max_episodes:
        total_episodes += 1

        prev_state, _ = env.reset()
        episodic_reward = 0
        episode_steps = 0
        while True:
            pbar.update()
            total_steps += 1
            episode_steps += 1

            action, noise = agent.action(tf.convert_to_tensor(prev_state))
            state, reward, done_old, done, info = env.step(action)
            step_reward_list.append(np.array(reward))
            action_list.append(np.array(action))

            agent.memory([prev_state, action, reward, state, done])
            episodic_reward += reward
            agent.train()
            prev_state = state

            tf.summary.scalar('Step Reward', data=episodic_reward, step=int(total_steps))


            # End this episode when `done` is True
            if done_old:
                break

            # End this episode when `done` is True
            if done:
                break


        tf.summary.scalar('Reward', data=episodic_reward, step=int(total_episodes))

        ep_reward_list.append(np.array([episodic_reward]))
        ep_steps_list.append(np.array([episode_steps]))

        pbar.set_postfix({'Ep Reward': episodic_reward})

    np.save(file=logdir+'/step_rewards.npy', arr=np.array(step_reward_list))
    np.save(file=logdir+'/episode_rewards.npy', arr=np.array(ep_reward_list))
    np.save(file=logdir+'/actions.npy', arr=np.array(action_list))
    np.save(file=logdir+'/episode_steps.npy', arr=np.array(ep_steps_list))
    print(f'Numpy files saved to {logdir}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index for tracking", type=int, default=0)
    parser.add_argument("--max_episodes", help="Number of episodes", type=int, default=1000)
    parser.add_argument("--max_steps", help="Maximum # of steps to run in total", default=1_000_000)
    parser.add_argument("--steps_per_episode", help="Number of steps", type=int, default=200)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasTD3-v0')
    parser.add_argument("--nwarmup", help="Agent warm-up size", type=int, default=1000)
    parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')
    parser.add_argument("--logdir", help="Directory to save results", type=str, default='None')

    # Get input arguments
    args = parser.parse_args()

    run_opt(
        index = args.index,
        max_episodes = int(args.max_episodes),
        max_steps =int(args.max_steps),
        steps_per_episode = int(args.steps_per_episode),
        agent_id = args.agent, 
        warmup_size = args.nwarmup,
        env_id = args.env, 
        logdir = args.logdir
    )

    # Print input settings
    # run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id, args_warmup_size, args_env_id, args_logdir)
