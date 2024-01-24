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
import jlab_opt_control.agents
from jlab_opt_control.utils.git_utils import get_git_revision_short_hash
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import logging

config = {
        # Environment
        "action_mode": "delta",
        "clip_magnets": True,
        "max_quad_delta": 0.1
        * np.array(
            [
                0.131072,
                0.262144,
                0.524288,
                0.262144,
                0.262144,
                0.262144,
                0.262144,
                0.262144,
                0.262144,
                0.262144,
                0.065536,
                0.065536,
                0.065536,
                0.131072,
            ]
        ),
        "magnet_init_mode": "design",
        "target_threshold": None,  # I think 5.0 is detector limit
        "threshold_hold": 1,
        "backend": "cheetah",  # Don't change this
        "tws0_mode": "curriculum",
        "magnet_space_mode": "narrow",
        "terminate_on_beam_loss": True,
        # Reward (also environment)
        "fel_intensity_transform": "ClippedLinear",
        "magnet_change_transform": "Sigmoid",
        "magnet_change_combiner": "Mean",
        "magnet_change_combiner_args": {},
        "magnet_change_combiner_weights": [1] * 14,
        "magnet_setting_transform": "SoftPlus",
        "magnet_setting_combiner": "Mean",
        "magnet_setting_combiner_args": {},
        "magnet_setting_combiner_weights": [1] * 14,
        "final_combiner": "Mean",
        "final_combiner_args": {},
        "final_combiner_weights": [3, 0.5, 0.5]
}

run_openai_log = logging.getLogger("RunOpenAI")
run_openai_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

seed = 1#time.time_ns()
tf.random.set_seed(seed)
np.random.seed(seed)
#run_openai_log.info(f'seeds {tf.random.}')

def run_opt(index, max_nepisodes, max_nsteps, agent_id, env_id, logdir):
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
        import gymnasium as gym
        env = gym.make(env_id, exclude_current_positions_from_observation=False)
    elif 'DnC2s' in env_id:
        import jlab_opt_control.envs as gym
        env = gym.make(env_id)
    elif 'fel' in env_id:
        from lcls_fel_tuning.environments.lcls_li26hxr import FELIntensityTuning
        # env = FELIntensityTuning(backend="cheetah", magnet_init_mode="design") 
        env = FELIntensityTuning(
            backend=config["backend"],
            backend_args={"tws0_mode": config["tws0_mode"]},
            action_mode=config["action_mode"],
            magnet_init_mode=config["magnet_init_mode"],
            magnet_space_mode=config["magnet_space_mode"],
            max_quad_delta=config["max_quad_delta"],
            target_threshold=config["target_threshold"],
            threshold_hold=config["threshold_hold"],
            clip_magnets=config["clip_magnets"],
            terminate_on_beam_loss=config["terminate_on_beam_loss"],
            fel_intensity_transform=config["fel_intensity_transform"],
            magnet_change_transform=config["magnet_change_transform"],
            magnet_change_combiner=config["magnet_change_combiner"],
            magnet_change_combiner_args=config["magnet_change_combiner_args"],
            magnet_change_combiner_weights=config["magnet_change_combiner_weights"],
            magnet_setting_transform=config["magnet_setting_transform"],
            magnet_setting_combiner=config["magnet_setting_combiner"],
            magnet_setting_combiner_args=config["magnet_setting_combiner_args"],
            magnet_setting_combiner_weights=config["magnet_setting_combiner_weights"],
            final_combiner=config["final_combiner"],
            final_combiner_args=config["final_combiner_args"],
            final_combiner_weights=config["final_combiner_weights"]
        )
    else:
        import gymnasium as gym
        env = gym.make(env_id)

    #if max_nsteps!=-1:
    #    env._max_episode_steps = max_nsteps
    #run_openai_log.info("Environment max steps ->  {}".format(env._max_episode_steps))
    if "fel" in env_id:
        num_states = 0
        for key in sorted(list(env.observation_space.keys())):
            num_states += env.observation_space[key].shape[0]
    else:
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
    agent = jlab_opt_control.agents.make(agent_id, env=env, logdir=logdir)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    total_nsteps = 0
    step = 0
    for ep in tqdm(range(max_nepisodes), desc='Index {} - Episodes'.format(index)):
        time_start = time.process_time()
        prev_state, _ = env.reset()
        
        ########################################################################
        if "fel" in env_id:
            state = []
            for key in sorted(list(env.observation_space.keys())):
                state.extend(prev_state[key])
            prev_state = np.array(state)
        ########################################################################
        
        episodic_reward = 0
        done = False
        count = 0
        while (done==False):
            total_nsteps += 1
            action, action_noise = agent.action(tf.convert_to_tensor(prev_state))
            assert 'numpy.ndarray' in str(type(action))
            run_openai_log.debug(f'action: {action}')
            run_openai_log.debug(f'action_noise: {action_noise}')

            # Take a step
            state, reward, terminate, truncate, info = env.step(action)
            
            ########################################################################
            if "fel" in env_id:
                temp_state = []
                for key in sorted(list(env.observation_space.keys())):
                    temp_state.extend(state[key])
                state = np.array(temp_state)
            ########################################################################
            
            run_openai_log.debug(f'reward: {reward}')
            run_openai_log.debug(f'reward: {type(reward)}')

            # Check shapes and data types
            assert 'numpy.ndarray' in str(type(state))
            assert state.shape == (num_states,)
            assert 'float' in str(type(reward)), str(type(reward))
            done = (terminate or truncate)
            step += 1
            agent.memory((prev_state, action, reward, state, done))
            episodic_reward += reward
            agent.train()
            prev_state = state
            count += 1
            # print(count)
            if count > max_nsteps:
                count = 0
                break
            # End this episode when `done` is True
            # if done:
            #     break

        ep_reward_list.append(episodic_reward)
        tf.summary.scalar('Training Reward', data=episodic_reward, step=int(ep))

        # Run inference test
        if ep % 1 == 0:
            inference_episodic_reward = 0
            inference_prev_state, _ = env.reset()
            
            ########################################################################
            if "fel" in env_id:
                state = []
                for key in sorted(list(env.observation_space.keys())):
                    state.extend(inference_prev_state[key])
                inference_prev_state = np.array(state)
            ########################################################################
            
            inference_done = False
            count = 0
            while (inference_done==False):
                inference_action, inference_action_noise = agent.action(tf.convert_to_tensor(inference_prev_state), train=False)
                inference_state, inference_reward, inference_terminate, inference_truncate, inference_info = env.step(inference_action)
                ########################################################################
                if "fel" in env_id:
                    state = []
                    for key in sorted(list(env.observation_space.keys())):
                        state.extend(inference_state[key])
                    inference_state = np.array(state)
                ########################################################################
                inference_episodic_reward += inference_reward
                inference_prev_state = inference_state
                inference_done = (inference_terminate or inference_truncate)
                count += 1
                if count > max_nsteps:
                    break

        tf.summary.scalar('Inference Reward', data=inference_episodic_reward, step=int(ep))

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
        
#         if ep%200==0:
#             plt.plot(ep_reward_list)
#             plt.xlabel("Episode #", fontsize=20)
#             plt.ylabel("Episodic Reward", fontsize=20)
#             plt.grid()
#             plt.savefig(os.path.join(logdir, "Episodic_reward.png"), dpi=120)
# #             plt.show()
#             plt.clf()
#             plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index for tracking", type=int, default=0)
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=100)
    parser.add_argument("--nsteps", help="Number of steps", type=int, default=100)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasTD3-v0')
    parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')
    parser.add_argument("--logdir", help="Directory to save results", type=str, default='None')

    # Get input arguments
    args = parser.parse_args()
    args_index = args.index
    args_nepisodes = args.nepisodes
    args_nsteps = args.nsteps
    args_agent_id = args.agent
    args_env_id = args.env
    args_logdir = args.logdir

    run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id, args_env_id, args_logdir)
