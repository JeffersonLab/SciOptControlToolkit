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
# Standard Library Imports

import argparse
import logging
import os
import time
from datetime import datetime
import warnings

# Third-Party Imports
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack, RescaleAction, TimeLimit

# Local Application/Library Specific Imports
import jlab_opt_control.agents
from jlab_opt_control.utils.git_utils import get_git_revision_short_hash
import jlab_opt_control.envs as custom_gym

warnings.filterwarnings("ignore")

run_openai_log = logging.getLogger("RunOpenAI")
run_openai_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

# PACEs
try:
    import paces.paces_envs as paces_gym
    from paces.paces_envs.lcls.utils.rescale_observation import RescaleObservation
    run_openai_log.info("PACEs environments successfully imported")
except ImportError:
    run_openai_log.info("PACEs environments not installed")

seed = time.time_ns() % np.power(2, 32)  # Numpy seed must be between 0 and 2^32 - 1
tf.random.set_seed(seed)
np.random.seed(seed)
# run_openai_log.info(f'seeds {tf.random.}')


def run_opt(index, max_nepisodes, max_nsteps, agent_id, env_id, logdir, buffer_type, buffer_size, inference_flag, difficulty, nepisode_avg, model_save_threshold):
    githash = get_git_revision_short_hash()
    run_openai_log.debug(githash)
    run_openai_log.debug(logdir)

    # Checks for buffer logging information, will default to config if not set in command line
    if buffer_type is None:
        buffer_type_log = "cfg"
    else:
        buffer_type_log = str(buffer_type)

    if buffer_size is None:
        buffer_size_log = "cfg"
    else:
        buffer_size_log = str(buffer_size)

    if logdir == 'None':
        logdir = "./results/index" + str(index) + "_agent_" + agent_id + "_buf_" + buffer_type_log + "_bsize_" + buffer_size_log + "_env_" + env_id + "_hash" \
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

    if env_id in gym.envs.registry:
        env = gym.make(env_id)
    elif env_id in custom_gym.list_registered_modules():
        env = custom_gym.make(env_id)
    elif 'paces_gym' in globals() and env_id in paces_gym.list_registered_modules():
        env = paces_gym.make(env_id)
        if 'LCLS' in env_id:
            env.set_curriculum_difficulty(difficulty)
            # Check if max_nsteps is not defined, set it to default (10) for lcls env
            if max_nsteps <= 0:
                max_nsteps = 10 
            env = TimeLimit(env, max_nsteps)
            env = RescaleObservation(env, -1, 1)
            env = RescaleAction(env, -1, 1)
            env = FlattenObservation(env)
            # env = FrameStack(env, 1)
    else:
        run_openai_log.error('Error finding environment')

    if max_nsteps != -1:
        env._max_episode_steps = max_nsteps
    run_openai_log.info(
        "Environment max steps ->  {}".format(env._max_episode_steps))

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
    tfb_path = os.path.join(logdir, 'metrics')
    file_writer.set_as_default()

    # Agent
    agent = jlab_opt_control.agents.make(
        agent_id, env=env, logdir=logdir, buffer_type=buffer_type, buffer_size=buffer_size)

    agent.save_cfg()
    agent.save("init")

    # To store reward history of each episode
    ep_reward_list = []
    inf_ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Variable to hold previous max
    # Init at very small number
    inference_episodic_hold = 0

    total_nsteps = 0

    # Only inference
    if inference_flag:
        for ep in tqdm(range(max_nepisodes), desc='Index {} - Episodes'.format(index)):
            time_start = time.process_time()
            inference_episodic_reward = 0
            inference_prev_state, _ = env.reset()
            inference_done = False
            while inference_done is False:
                total_nsteps += 1
                inference_action, inference_action_noise = agent.action(
                    tf.convert_to_tensor(inference_prev_state), train=False, inference=True)
                inference_state, inference_reward, inference_terminate, inference_truncate, inference_info = env.step(
                    inference_action)
                inference_episodic_reward += inference_reward
                inference_prev_state = inference_state
                inference_done = (inference_terminate or inference_truncate)
            
            tf.summary.scalar('Inference Reward',
                            data=inference_episodic_reward, step=int(ep))

            ep_reward_list.append(inference_episodic_reward)

            avg_reward = np.mean(ep_reward_list[-nepisode_avg:])
            time_end = time.process_time()
            if total_nsteps % 1000 == 0:
                run_openai_log.info(
                    "Episode Elapsed Time {}".format((time_end - time_start)))
                run_openai_log.info(
                    "Episode * {} * Inference Episodic Reward is ==> {}".format(ep, inference_episodic_reward))
                run_openai_log.info(
                    "Episode * {} * Inference Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)

            with open(logdir + '/results.npy', 'wb') as f:
                np.save(f, np.array(ep_reward_list))
            
    # Training w/ inference
    else:
        for ep in tqdm(range(max_nepisodes), desc='Index {} - Episodes'.format(index)):
            time_start = time.process_time()
            prev_state, _ = env.reset()
            episode_timesteps = 0
            episodic_reward = 0
            done = False
            while done is False:
                total_nsteps += 1
                episode_timesteps += 1
                action, action_noise = agent.action(
                    tf.convert_to_tensor(prev_state))
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
                done_buffer = terminate

                agent.memory((prev_state, action, reward, state, done_buffer))
                episodic_reward += reward
                agent.train()
                prev_state = state
                if done:
                    break

            ep_reward_list.append(episodic_reward)
            tf.summary.scalar('Training Reward',
                            data=episodic_reward, step=int(ep))

            # Run inference test
            if ep % 1 == 0 and agent.buffer.size() > np.max([agent.batch_size, agent.warmup_size]):
                agent.buffer.save()
                inference_episodic_reward = 0
                inference_prev_state, _ = env.reset()
                inference_done = False
                while inference_done is False:
                    inference_action, inference_action_noise = agent.action(
                        tf.convert_to_tensor(inference_prev_state), train=False, inference=True)
                    inference_state, inference_reward, inference_terminate, inference_truncate, inference_info = env.step(
                        inference_action)
                    inference_episodic_reward += inference_reward
                    inference_prev_state = inference_state
                    inference_done = (inference_terminate or inference_truncate)
                    if inference_done:
                        break
                
                # init for first epoch
                if (ep == 0):
                    inference_episodic_hold = inference_episodic_reward
                

                inf_ep_reward_list.append(inference_episodic_reward)
                inf_avg_reward = np.mean(inf_ep_reward_list[-nepisode_avg:])

                if inf_avg_reward != 0:
                    percentage_change = (inf_avg_reward - inference_episodic_hold) / abs(inference_episodic_hold)
                    if percentage_change >= model_save_threshold:
                        str_pct_inc = 'epoch_' + str(ep) + '_' + f"{int(100*percentage_change):03d}"
                        agent.save(str_pct_inc)
                        inference_episodic_hold = inf_avg_reward

                tf.summary.scalar('Inference Reward',
                                data=inference_episodic_reward, step=int(ep))

            avg_reward = np.mean(ep_reward_list[-nepisode_avg:])
            time_end = time.process_time()
            if total_nsteps % 1000 == 0:
                run_openai_log.info(
                    "Episode Elapsed Time {}".format((time_end - time_start)))
                run_openai_log.info(
                    "Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
                run_openai_log.info(
                    "Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)

            with open(logdir + '/results.npy', 'wb') as f:
                np.save(f, np.array(ep_reward_list))
        agent.buffer.save()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", help="Index for tracking", type=int, default=0)
    parser.add_argument(
        "--nepisodes", help="Number of episodes", type=int, default=100)
    parser.add_argument("--nsteps", help="Number of steps",
                        type=int, default=-1)
    parser.add_argument("--bsize", help="Buffer size", type=int, default=None)
    parser.add_argument("--btype", help="Buffer Type", type=str, default=None)
    parser.add_argument("--agent", help="Agent used for RL",
                        type=str, default='KerasTD3-v0')
    parser.add_argument("--env", help="Environment used for RL",
                        type=str, default='Pendulum-v1')
    parser.add_argument(
        "--logdir", help="Directory to save results", type=str, default='None')
    parser.add_argument(
        "--inference", help="Inference only run flag", type=str, default=None)
    parser.add_argument(
        "--difficulty", help="Curriculum difficulty level for LCLS env", type=float, default=0.08)
    parser.add_argument(
        "--nepisode_avg", help="Number of episodes to average the reward over", type=int, default=20)
    parser.add_argument(
        "--model_save_threshold", help="Percentage increase threshold (in fraction) to save the model", type=float, default=0.05)

    # Get input arguments
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    
    args_index = args.index
    args_nepisodes = args.nepisodes
    args_nsteps = args.nsteps
    args_agent_id = args.agent
    args_env_id = args.env
    args_logdir = args.logdir
    args_buf_size = args.bsize
    args_buf_type = args.btype
    args_inference = args.inference
    args_difficulty = args.difficulty
    args_nepisode_avg = args.nepisode_avg
    args_model_save_threshold = args.model_save_threshold

    run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id,
            args_env_id, args_logdir, args_buf_type, args_buf_size, args_inference, args_difficulty, args_nepisode_avg, args_model_save_threshold)

if __name__ == "__main__":
    main()
    
