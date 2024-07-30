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
import io
import logging
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import (MultipleLocator)
from scipy.spatial import distance
from tqdm import tqdm

import jlab_opt_control.agents
from jlab_opt_control.utils.git_utils import get_git_revision_short_hash

warnings.filterwarnings("ignore")

run_openai_log = logging.getLogger("RunOpenAI")
run_openai_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

seed = 1  # time.time_ns()
tf.random.set_seed(seed)
np.random.seed(seed)
# run_openai_log.info(f'seeds {tf.random.}')


def run_opt(index, max_nepisodes, max_nsteps, agent_id, env_id, logdir, buffer_type, buffer_size):
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
    if ('HalfCheetah' or 'Hopper') in env_id:
        import gymnasium as gym
        env = gym.make(
            env_id, exclude_current_positions_from_observation=False)
    elif 'DnC2s' in env_id:
        import jlab_opt_control.envs as gym
        env = gym.make(env_id)
    elif 'PACES' in env_id:
        import paces.paces_envs as gym
        env = gym.make(env_id)
    else:
        import gymnasium as gym
        env = gym.make(env_id)

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
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Variable to hold previous max
    # Init at very small number
    inference_episodic_hold = 0

    total_nsteps = 0
    inference_best_total_reward = 0.0

    nscans = 20
    np_alphas = np.zeros((nscans,env.reward_space.shape[0]), dtype=np.float32)
    np_alphas[:, 1] = np.linspace(0.0,1.0, nscans) #array([1.00, 0.99, 0.95, 0.90, 0.85, 0.75, 0.50, 0])
    np_alphas[:, 0] = 1.0 - np_alphas[:,1]
    print(f'np_alphas: {np_alphas}')
    print(f'np_alphas: {np_alphas.shape}')

    # max_nscans = 10
    # current_nscans = 0
    max_nepochs = 100
    max_nepisodes = 500
    for epoch in tqdm(range(max_nepochs), desc='Epochs'):
        #time_start = time.process_time()
        # nscans = 2 + int(ep/5000.0)
        # if nscans>current_nscans:
        #     current_nscans=nscans
        #     run_openai_log.info(f'Running nscans: {nscans}')
        # nscans = epoch+3
        # if nscans>=max_nscans:
        #     nscans = max_nscans
        #nscans = 10
        run_openai_log.info(f'Scanning {nscans}')
        #rdm_dirichlet = np.random.dirichlet((1, 2), size=max_nscans)
        for scan in tqdm(range(nscans), desc='Alpha Scan'):
            for episode in tqdm(range(max_nepisodes), desc='Episodes'):
                prev_state, _ = env.reset()
                alphas = np.zeros(env.reward_space.shape[0], dtype=np.float32)
                alphas[0] = np_alphas[scan, 0]
                alphas[1] = np_alphas[scan, 1]
                # alphas[1] = 1.0 - float(scan / (nscans - 1.0))
                # alphas[0] = 1.0 - alphas[1]
                # alphas[1] = rdm_dirichlet[0,0]# 1.0 - float(scan / (nscans - 1.0))
                # alphas[0] = rdm_dirichlet[0,1] #1.0 - alphas[1]
                #run_openai_log.info(f'Running alphas: {alphas}')

    #            alphas[1] = rdm_dirichlet[s,1]#1.0 - float(s / nscans)
    #            alphas[0] = rdm_dirichlet[s,0]#1.0 - alphas[1]
                alphas.astype(dtype=np.float32)
                alphas = np.expand_dims(alphas, 0)

                episode_timesteps = 0
                episodic_reward = np.zeros(env.reward_space.shape[0], dtype=np.float32)
                done = False
                while done is False:
                    total_nsteps += 1
                    episode_timesteps += 1
                    action, action_noise = agent.action(tf.convert_to_tensor(prev_state), tf.convert_to_tensor(alphas))
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
                    assert reward.shape == (env.reward_space.shape[0],)
                    done = (terminate or truncate)
                    #if terminate==False:
                    agent.memory((prev_state, action, reward, state, done, alphas))
                    episodic_reward += reward
                    prev_state = state
                    if epoch>1:
                        agent.train()

        # Run inference test
        if epoch>=1:# and epoch % 5 == 0:
            run_openai_log.info(f'Running inference ...')
            inference_nscans = 250
            # if ep % 1000 == 0:
            #     inference_nscans = 1000
            scan_trips, scan_heats, scan_alphas, scan_rewards, scan_qvalues_alphas, scan_energy = [], [], [], [], [], []
            inference_total_reward = 0.0
            # gfg = np.random.dirichlet((1, 2), size=inference_nscans)
            # alphas = np.zeros(env.reward_space.shape[0], dtype=np.float32)
            # alphas[1] = 1.0 - float(s / (inference_nscans - 1.0))
            # alphas[0] = 1.0 - alphas[1]
            for s in tqdm(range(inference_nscans), desc='Inference Scan'):
                #for r in range(env.reward_space.shape[0]):
                # inference_episodic_reward = np.zeros(env.reward_space.shape[0], dtype=np.float32)
                # inference_alphas = np.zeros(env.reward_space.shape[0], dtype=np.float32)
                # inference_alphas[1] = gfg[s,1]#1.0 - float(s/inference_nscans)
                # inference_alphas[0] = gfg[s,0]#1.0 -  inference_alphas[1]
                inference_alphas = np.zeros(env.reward_space.shape[0], dtype=np.float32)
                inference_alphas[1] = 1.0 - float(s / (inference_nscans - 1.0))
                inference_alphas[0] = 1.0 - inference_alphas[1]
                inference_alphas.astype(dtype=np.float32)
                inference_alphas  = np.expand_dims(inference_alphas, 0)
                inference_prev_state, _ = env.reset()
                inference_done = False
                while inference_done is False:
                    inference_action, inference_action_noise = agent.action(
                        tf.convert_to_tensor(inference_prev_state),
                        tf.convert_to_tensor(inference_alphas))
                    inference_state, inference_reward, inference_terminate, inference_truncate, inference_info = \
                        env.step(inference_action)
                    inference_prev_state = np.expand_dims(inference_prev_state, 0)
                    inference_action = np.expand_dims(inference_action, 0)
                    inference_qvalues = agent.critic_model1(tf.convert_to_tensor(inference_prev_state), tf.convert_to_tensor(inference_action), training=False)
                    inference_qvalues_alphas = tf.multiply(inference_qvalues, inference_alphas)
                    scan_energy.append(inference_info['energy'])
                    if inference_terminate==False:
                        scan_trips.append(inference_info['trip'])
                        scan_heats.append(inference_info['heat'])
                        scan_alphas.append(inference_alphas[0])#[0][1])
                        scan_rewards.append(inference_reward)#50*(float(np.sum(inference_reward))))
                        scan_qvalues_alphas.append(inference_qvalues_alphas[0])
                        #scan_rewards.append(50*np.exp(np.exp(float(np.sum(inference_reward))))-2.7)
                        #run_openai_log.info(f'inference_reward: {inference_reward}')
                        inference_total_reward += np.sum(inference_reward)
                        #run_openai_log.info(f'inference_total_reward: {inference_total_reward}')
                    inference_prev_state = inference_state
                    inference_done = (inference_terminate or inference_truncate)

            inference_total_reward = inference_total_reward/inference_nscans
            run_openai_log.debug(f'inference_total_reward: {inference_total_reward} '
                                f'and inference_best_total_reward: {inference_best_total_reward}')

            if inference_best_total_reward<=inference_total_reward:
                if inference_best_total_reward>0:
                    percentage_change = (inference_total_reward - inference_best_total_reward) / abs(inference_best_total_reward)
                    run_openai_log.info(f'Improved model {inference_total_reward} by {percentage_change}')
                    str_pct_inc = 'epoch_' + str(epoch) + '_' + f"{int(100 * percentage_change):03d}"
                    agent.save(str_pct_inc)
                inference_best_total_reward = inference_total_reward
                run_openai_log.info(f'Settinginference_best_total_reward to: {inference_best_total_reward}')

            tf.summary.scalar('Total MO Inference Reward', data=inference_total_reward, step=int(epoch))
            print(f'Number of valid scans: {len(scan_trips)}')
            if len(scan_trips)>0:
                # if ep % 1000 == 0:
                #     # Sort and filter:
                #     scan_heats = np.array(scan_heats)
                #     scan_trips = np.array(scan_trips)
                #     scan_rewards = np.array(scan_rewards)
                #     scan_alphas = np.array(scan_alphas)
                #     trips_idx_asc = scan_trips.argsort()
                #     trips_idx_des = trips_idx_asc[::-1]
                #     plt.scatter(scan_heats[trips_idx_des[0:100]],scan_trips[trips_idx_des[0:100]],
                #                 s=scan_rewards[trips_idx_des[0:100]], c=scan_alphas[trips_idx_des[0:100]])
                # else:
                # Distance between Q and Alpha
                fig, ax1 = plt.subplots(dpi=90)
                distances1 = []
                for s in range(len(scan_alphas)):
                    distances1.append(distance.cosine(scan_alphas[s], scan_rewards[s]))
                plt.hist(distances1, bins=20, range=[0,1], color='blue', alpha=0.5)
                plt.xlim(0.0,1.0)
                plt.tight_layout()
                plt.savefig(logdir+f'/distance1_ep{epoch}_{inference_total_reward:.4f}.jpeg')
                plt.clf()
                plt.close("all")

                fig, ax1 = plt.subplots(dpi=90)
                distances2 = []
                for s in range(len(scan_alphas)):
                    distances2.append(distance.cosine(scan_alphas[s], scan_qvalues_alphas[s]))
                plt.hist(distances2, bins=20, range=[0,1], color='red', alpha=0.5)
                plt.xlim(0.0, 1.0)
                plt.tight_layout()
                plt.savefig(logdir + f'/distance2_ep{epoch}_{inference_total_reward:.4f}.jpeg')
                plt.clf()
                plt.close("all")

                # Plot reward vs alpha
                fig, ax1 = plt.subplots(dpi=90)
                plt.xlim(0.0, 1.0)
                plt.xlabel('Alpha')
                scan_alphas = np.array(scan_alphas)
                scan_qvalues_alphas = np.array(scan_qvalues_alphas)
                scan_distances1 = np.array(distances1)
                ax1.scatter(scan_alphas[:, 0], scan_distances1[:], s=50, c='red')
                ax1.scatter(scan_alphas[:, 1], scan_distances1[:], s=50, c='blue')
                ax1.set_ylabel('distance')
                plt.tight_layout()
                plt.savefig(logdir + f'/distance_scatter_ep{epoch}_{inference_total_reward:.4f}.jpeg')
                plt.clf()
                plt.close("all")

                # Plot reward vs alpha
                fig, ax1 = plt.subplots(dpi=90)
                plt.xlim(0.0,1.0)
                plt.xlabel('Alpha')
                scan_alphas = np.array(scan_alphas)
                scan_qvalues_alphas = np.array(scan_qvalues_alphas)
                # print(f'scan_alphas {scan_alphas.shape}')
                # print(f'scan_rewards {scan_qvalues_alphas.shape}')
                ax1.scatter(scan_alphas[:,0],scan_qvalues_alphas[:,0], s=50, c='red')
                ax1.set_ylabel('Qvalue 0', c='red');
                ax2 = ax1.twinx()
                ax2.scatter(scan_alphas[:,1],scan_qvalues_alphas[:,1], s=50, c='black')
                ax2.set_ylabel('Qvalue 1', c='black');
                plt.tight_layout()
                plt.savefig(logdir+f'/reward1_alpha_ep{epoch}_{inference_total_reward:.4f}.jpeg')
                plt.clf()
                plt.close("all")
                #
                #
                fig, ax = plt.subplots(dpi=90)
                plt.scatter(scan_heats, scan_trips, s=100,
                            c=scan_distances1[:])  # np.sum(scan_rewards,axis=1))#scan_alphas)
                # plt.xlim(20.8, 22.4)
                # plt.ylim(0.015, 0.05)
                # Change major ticks to show every 20.
                ax.xaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_major_locator(MultipleLocator(0.005))
                plt.text(.95, .99, f'Total MO Reward: {inference_total_reward:.4f}',
                         ha='right', va='top', transform=ax.transAxes)
                plt.grid()
                plt.xlabel('Heat Load [W]')
                plt.ylabel('Trip Rate [per hour]');
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(logdir + f'/pareto_distance_ep{epoch}_{inference_total_reward:.4f}.jpeg')
                plt.clf()
                plt.close("all")
                #
                fig, ax = plt.subplots(dpi=90)
                plt.scatter(scan_heats,scan_trips, s=100, c=scan_alphas[:,0])#np.sum(scan_rewards,axis=1))#scan_alphas)
                # plt.xlim(20.8, 22.4)
                # plt.ylim(0.015, 0.05)
                # Change major ticks to show every 20.
                ax.xaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_major_locator(MultipleLocator(0.005))
                plt.text(.95, .99, f'Total MO Reward: {inference_total_reward:.4f}',
                         ha='right', va='top', transform=ax.transAxes)
                plt.grid()
                plt.xlabel('Heat Load [W]')
                plt.ylabel('Trip Rate [per hour]');
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(logdir+f'/pareto_ep{epoch}_{inference_total_reward:.4f}.jpeg')
                # Convert figure to an image tensor and log
                buf = io.BytesIO()
                canvas = FigureCanvasAgg(fig)
                canvas.print_png(buf)
                tensor = tf.image.decode_png(buf.getvalue(), channels=4)
                tf.summary.image(
                    "Pareto", data=tensor[None], step=int(epoch)
                )
                plt.clf()
                plt.close("all")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument( "--index", help="Index for tracking", type=int, default=0)
    parser.add_argument( "--nepisodes", help="Number of episodes", type=int, default=50000)
    parser.add_argument("--nsteps", help="Number of steps",type=int, default=-1)
    parser.add_argument("--bsize", help="Buffer size", type=int, default=None)
    parser.add_argument("--btype", help="Buffer Type", type=str, default=None)
    parser.add_argument("--agent", help="Agent used for RL",type=str, default='MO-KerasTD3-v0')
    parser.add_argument("--env", help="Environment used for RL",type=str, default='PACES-MO-CEBAF-8D-v0')
    parser.add_argument("--logdir", help="Directory to save results", type=str, default='None')

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

    run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id,
            args_env_id, args_logdir, args_buf_type, args_buf_size)

if __name__ == "__main__":
    main()
    
