import argparse
import os

import time
from datetime import datetime

import gymnasium as gym
import tensorflow as tf

import numpy as np
import jlab_opt_control.agents
from jlab_opt_control.utils.git_utils import get_git_revision_short_hash
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import logging

run_openai_log = logging.getLogger("RunOpenAI")
run_openai_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

def run_opt(index, max_nepisodes, max_nsteps, agent_id, warmup_size, env_id, logdir):
    githash = get_git_revision_short_hash()
    run_openai_log.debug(githash)
    run_openai_log.debug(logdir)
    if logdir == 'None':
        logdir = "./results/index" + str(index) + "_agent_" + agent_id + "_env_" + env_id + "_hash" \
                 + githash + "_results_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        logdir = logdir + "/index" + str(index) + "_agent_" + agent_id + "_env_" + env_id + "_date_" \
                 + datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        os.mkdir(logdir)
    except OSError as error:
        run_openai_log.error('Error:', error)

    #
    # Environment
    run_openai_log.info('Running env: {}'.format(env_id))
    if ('HalfCheetah' or 'Hopper') in env_id:
        env = gym.make(env_id, exclude_current_positions_from_observation=False)
    else:
        env = gym.make(env_id)

    env._max_episode_steps = max_nsteps
    run_openai_log.info("Environment max steps ->  {}".format(env._max_episode_steps))

    num_states = env.observation_space.shape[0]
    run_openai_log.info("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    run_openai_log.info("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    run_openai_log.info("Max Value of Action ->  {}".format(upper_bound))
    run_openai_log.info("Min Value of Action ->  {}".format(lower_bound))

    file_writer = tf.summary.create_file_writer(logdir + '/metrics')
    file_writer.set_as_default()

    # Agent
    agent = jlab_opt_control.agents.make(agent_id, env=env)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    total_nsteps = 0

    for ep in tqdm(range(max_nepisodes), desc='Index {} - Episodes'.format(index)):
        time_start = time.process_time()
        prev_state, _ = env.reset()
        episodic_reward = 0
        for estep in range(max_nsteps):
            total_nsteps += 1
            action, action_noise = agent.action(tf.convert_to_tensor(prev_state))
            assert 'numpy.ndarray' in str(type(action))
            run_openai_log.debug(f'action: {action}')
            run_openai_log.debug(f'action_noise: {action_noise}')

            # Take a step
            state, reward, terminate, truncate, info = env.step(action)
            run_openai_log.debug(f'reward: {reward}')
            run_openai_log.debug(f'reward: {type(reward)}')

            # Check shapes and data types
            assert 'numpy.ndarray' in str(type(state))
            assert state.shape == (num_states,)
            assert 'float' in str(type(reward)), str(type(reward))

            agent.memory((prev_state, action, reward, state, truncate))
            episodic_reward += reward
            agent.train()
            prev_state = state

            # End this episode when `done` is True
            if terminate:
                break

            # End this episode when `done` is True
            if truncate:
                break

        ep_reward_list.append(episodic_reward)

        tf.summary.scalar('Reward', data=episodic_reward, step=int(ep))

        # Mean of last 40 episodes
        nepisode_mod = 10
        avg_reward = np.mean(ep_reward_list[-nepisode_mod:])
        time_end = time.process_time()
        if total_nsteps % 1000 == 0:
            run_openai_log.info("\nEpisode Elapsed Time {}".format((time_end - time_start)))
            run_openai_log.info("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
            run_openai_log.info("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        with open(logdir + '/results.npy', 'wb') as f:
            np.save(f, np.array(ep_reward_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index for tracking", type=int, default=0)
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=1000)
    parser.add_argument("--nsteps", help="Number of steps", type=int, default=200)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasTD3-v0')
    parser.add_argument("--nwarmup", help="Agent warm-up size", type=int, default=0)
    parser.add_argument("--env", help="Environment used for RL", type=str, default='MountainCarContinuous-v0')
    parser.add_argument("--logdir", help="Directory to save results", type=str, default='None')
    parser.add_argument("--profile", help="Profiling overrides all setting", type=bool, default=False)

    # Get input arguments
    args = parser.parse_args()
    args_index = args.index
    args_nepisodes = args.nepisodes
    args_nsteps = args.nsteps
    args_agent_id = args.agent
    args_warmup_size = args.nwarmup
    args_env_id = args.env
    args_logdir = args.logdir

    run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id, args_warmup_size, args_env_id, args_logdir)
