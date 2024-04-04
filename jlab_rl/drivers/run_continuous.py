# Copyright (c) 2020, Jefferson Science Associates, LLC. All Rights Reserved. Redistribution
# and use in source and binary forms, with or without modification, are permitted as a
# licensed user provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# This material resulted from work developed under a United States Government Contract.
# The Government retains a paid-up, nonexclusive, irrevocable worldwide license in such
# copyrighted data to reproduce, distribute copies to the public, prepare derivative works,
# perform publicly and display publicly and to permit others to do so.
#
# THIS SOFTWARE IS PROVIDED BY JEFFERSON SCIENCE ASSOCIATES LLC "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# JEFFERSON SCIENCE ASSOCIATES, LLC OR THE U.S. GOVERNMENT BE LIABLE TO LICENSEE OR ANY
# THIRD PARTES FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
import os

import time
from datetime import datetime

import tensorflow as tf

import numpy as np
import jlab_rl.agents
from jlab_rl.utils.git_utilts import get_git_revision_short_hash
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import logging

run_openai_log = logging.getLogger("RunOpenAI")
run_openai_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

#seed = 1#time.time_ns()
#tf.random.set_seed(seed)
#np.random.seed(seed)
#run_openai_log.info(f'seeds {tf.random.}')

def run_opt(index, max_nepisodes, max_nsteps, nwarmup, agent_id, env_id, logdir):
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
        os.makedirs(logdir)
    except OSError as error:
        run_openai_log.error('Error making file:', error)

    #
    # Environment
    run_openai_log.info('Running env: {}'.format(env_id))
    if ('HalfCheetah' or 'Hopper') in env_id:
        import gym as gym
        env = gym.make(env_id, exclude_current_positions_from_observation=False)
        inference_env = gym.make(env_id, exclude_current_positions_from_observation=False)
    elif 'DnC2s' in env_id:
        import jlab_rl.envs as gym
        env = gym.make(env_id)
        inference_env = gym.make(env_id)
    if 'PACES' in env_id:
        import paces.paces_envs as gym
        env = gym.make(env_id)
        inference_env = gym.make(env_id)
    else:
        import gymnasium as gym
        env = gym.make(env_id)
        inference_env = gym.make(env_id)

    if max_nsteps!=-1:
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

    run_openai_log.info(f'Log Path: {logdir}')
    tfb_path = os.path.join(logdir, 'metrics')
    run_openai_log.info(f'TFB Path: {tfb_path}')
    file_writer = tf.summary.create_file_writer(tfb_path)
    file_writer.set_as_default()

    # Agent
    agent = jlab_rl.agents.make(agent_id, env=env, warmup_size=nwarmup, logdir=logdir)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    total_nsteps = int(1e+6)
    total_steps = 0
    ep = 0
    pbar = tqdm(total=total_nsteps)
    while total_steps != total_nsteps:#, desc='Index {} - Step'.format(total_steps)):
#    for ep in tqdm(range(max_nepisodes), desc='Index {} - Episodes'.format(index)):
        time_start = time.process_time()
        prev_state, _ = env.reset()
        episodic_reward = 0
        done = False
        while (done==False):
            total_steps += 1
            pbar.update(1)
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
            done = (terminate or truncate)
            agent.memory((prev_state, action, reward, state, done))
            episodic_reward += reward
            agent.train()
            prev_state = state

            if done:
                ep += 1
            # End this episode when `done` is True
            # if done:
            #     break

        ep_reward_list.append(episodic_reward)
        tf.summary.scalar('Episodic Training Reward', data=episodic_reward, step=int(total_steps))

        # Run inference test
        if total_steps%100 == 0:
            inference_episodic_reward = 0
            inference_prev_state, _ = inference_env.reset()
            inference_done = False
            inference_x, inference_y, inference_z = [],[],[]
            for i in range(100):
                while (inference_done==False):
                    inference_action, inference_action_noise = agent.action(tf.convert_to_tensor(inference_prev_state), train=False)
                    inference_state, inference_reward, inference_terminate, inference_truncate, inference_info = inference_env.step(inference_action)
                    inference_episodic_reward += inference_reward
                    inference_prev_state = inference_state
                    inference_done = (inference_terminate or inference_truncate)
                    inference_x.append(inference_action[0])
                    inference_y.append(inference_action[1])
                    inference_z.append(inference_reward)

                #
            print(f'Number of trial {len(inference_z)}')
            if agent.num_actions == 2:
                import matplotlib.pyplot as plt
                from matplotlib import cm

                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.grid(True, linestyle='-', color='0.75')
                # scatter with colormap mapping to z value
                cb = ax.scatter(inference_x, inference_y, s=35, c=inference_z, marker='o',
                                vmin=0.0, vmax=1, cmap=cm.jet);
                plt.xlim(-1.1, 1.1)
                plt.ylim(-1.1, 1.1)
                plt.colorbar(cb)
                plt.tight_layout
                ax.set_title(f'Average Reward {np.mean(inference_reward):.4f}+-{np.std(inference_reward):.4f}')

                plt.savefig(logdir + '/inference_xy_action_reward_{}.png'.format(int(total_steps/100)))
                plt.close()
                # if 'DnC2s' in env_id:
                #     env.plot
                # if done:
                #     break
            tf.summary.scalar('Inference Reward', data=inference_episodic_reward, step=int(total_steps))

        # Mean of last 40 episodes
        nepisode_mod = 10
        avg_reward = np.mean(ep_reward_list[-nepisode_mod:])
        time_end = time.process_time()
        if total_nsteps % 1000 == 0:
            run_openai_log.info("Episode Elapsed Time {}".format((time_end - time_start)))
            run_openai_log.info("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
            run_openai_log.info("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        with open(logdir + '/results.npy', 'wb') as f:
            np.save(f, np.array(ep_reward_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index for tracking", type=int, default=0)
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=100)
    parser.add_argument("--nsteps", help="Number of steps", type=int, default=-1)
    parser.add_argument("--nwarmup", help="Number of warmup steps", type=int, default=1000)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasTD3-v0')
    parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')
    parser.add_argument("--logdir", help="Directory to save results", type=str, default='None')

    # Get input arguments
    args = parser.parse_args()
    args_index = args.index
    args_nepisodes = args.nepisodes
    args_nsteps = args.nsteps
    args_nwarmup = args.nwarmup
    args_agent_id = args.agent
    args_env_id = args.env
    args_logdir = args.logdir

    run_opt(args_index, args_nepisodes, args_nsteps, args_nwarmup, args_agent_id, args_env_id, args_logdir)
