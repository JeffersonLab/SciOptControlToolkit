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
from pymoo.indicators.hv import Hypervolume
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
    elif 'SCORE' in env_id:
        import score.envs as gym
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

    ga_results_loc = '../notebooks'
    if ga_results_loc is not None:
        if '8D' in env_id:
            ga_results = np.load(os.path.join(ga_results_loc, "1L10_TEST8_nsga_II_results.npy"))
            ref = [22.0, 0.04]
            # ideal = [20.0, 0.01]
            # metric = Hypervolume(ref_point= ref,
            #              norm_ref_point=False,
            #              zero_to_one=False,
            #              ideal=ideal,
            #              nadir=ref)
            # max_vol = (ref[0] - ideal[0]) * (ref[1] - ideal[1])
            # ga_hv = np.round(metric.do(ga_results)*100/max_vol, 3)
            metric = Hypervolume(ref_point= ref)
            ga_hv = metric.do(ga_results)
        elif '-N-' in env_id:
            ga_results = np.load(os.path.join(ga_results_loc, "NORTH_nsga_II_results.npy"))
            #ga_results = np.load(os.path.join(ga_results_loc, "NORTH_nsga_II_results_045000.npy"))
            #ref = [2530.0, 6.0]
            ref = [3000.0, 20.0]
            metric = Hypervolume(ref_point=ref)
            # ideal = [2380.0, 1.0]
            # metric = Hypervolume(ref_point= ref,
            #              norm_ref_point=False,
            #              zero_to_one=False,
            #              ideal=ideal,
            #              nadir=ref)
            # max_vol = (ref[0] - ideal[0]) * (ref[1] - ideal[1])
            # ga_hv = np.round(metric.do(ga_results)*100/max_vol, 3)
            ga_hv = metric.do(ga_results)#*100/max_vol, 3)
        else:
            ga_results = None
    else:
        ga_results = None

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Variable to hold previous max
    # Init at very small number
    inference_episodic_hold = 0

    total_nsteps = 0
    inference_best_total_reward = 0.0

    nscans = 100
    np_alphas = np.zeros((nscans,env.reward_space.shape[0]), dtype=np.float32)
    np_alphas[:, 1] = np.linspace(0.0,1.0, nscans) #array([1.00, 0.99, 0.95, 0.90, 0.85, 0.75, 0.50, 0])
    np_alphas[:, 0] = 1.0 - np_alphas[:,1]
    #print(f'np_alphas: {np_alphas}')
    #print(f'np_alphas: {np_alphas.shape}')

    nvalid_solutions = 0
    max_nepochs = 50000
    for epoch in tqdm(range(max_nepochs), desc='Epochs'):
        scans = np.random.rand(nscans)
        alphas = np.stack([scans, (1-scans)*1.5], axis=1)
        for scan in tqdm(range(nscans), desc='Alpha Scan'):
            prev_state, _ = env.reset()

            episode_timesteps = 0
            episodic_reward = np.zeros(env.reward_space.shape[0], dtype=np.float32)
            done = False
            while done is False:
                total_nsteps += 1
                episode_timesteps += 1
                action, action_noise = agent.action(tf.convert_to_tensor(prev_state), tf.convert_to_tensor(alphas[scan]))
                assert 'numpy.ndarray' in str(type(action))
                run_openai_log.debug(f'action: {action}')
                run_openai_log.debug(f'action_noise: {action_noise}')

                # Take a step
                state, reward, terminate, truncate, info = env.step(action)
                run_openai_log.debug(f'reward: {reward}')
                run_openai_log.debug(f'reward: {type(reward)}')
                run_openai_log.debug(f'energy: {info["energy"]}')
                run_openai_log.debug(f'terminate: {terminate}/{int(terminate)}')
                if terminate==False:
                    nvalid_solutions += 1
                    #run_openai_log.info(f'info: {info}')

                #sys.exit()
                if "Pareto" in agent_id:
                    reward = np.array([info['heat'], info['trip']])
                    run_openai_log.debug(f'new reward: {reward}')

                # Check shapes and data types
                assert 'numpy.ndarray' in str(type(state))
                assert state.shape == (num_states,)
                assert 'float' in str(type(reward)), str(type(reward))
                assert reward.shape == (env.reward_space.shape[0],)
                done = (terminate or truncate)
                agent.memory((prev_state, action, reward, state, done, alphas[scan]))
                episodic_reward += reward
                prev_state = state
                agent.train()

        # Run inference test
        run_openai_log.info(f'total_nsteps {total_nsteps}')
        if total_nsteps!=0:
            run_openai_log.info(f'fraction of valid solutions: {float(nvalid_solutions/total_nsteps)}')
        if total_nsteps % 1000 == 0:
            run_openai_log.info(f'Running inference ...')
            inference_nscans = 250
            inference_scans = np.random.rand(inference_nscans)
            inference_alphas = np.stack([inference_scans, (1 - inference_scans) * 1.5], axis=1)

            scan_trips, scan_heats, scan_alphas, scan_rewards, scan_qvalues_alphas, scan_energy = [], [], [], [], [], []
            inference_total_reward = 0.0

            for s in tqdm(range(inference_nscans), desc='Inference Scan'):
                inference_prev_state, _ = env.reset()
                inference_done = False
                while inference_done is False:
                    inference_action, inference_action_noise = agent.action(
                        tf.convert_to_tensor(inference_prev_state),
                        tf.convert_to_tensor(inference_alphas[s]), train=False)
                    inference_state, inference_reward, inference_terminate, inference_truncate, inference_info = \
                        env.step(inference_action)
                    if inference_terminate==False:
                        scan_trips.append(inference_info['trip'])
                        scan_heats.append(inference_info['heat'])
                        scan_alphas.append(inference_alphas[s, 0])
                    inference_prev_state = inference_state
                    inference_done = (inference_terminate or inference_truncate)

            rl_points = np.stack([scan_heats, scan_trips], axis=1)
            good_indices = np.where((rl_points[:, 0] <= ref[0]) & (rl_points[:, 1] <= ref[1]))[0]
            print(f'Number of valid scans: {len(good_indices)}')
            #print(f'Number of valid scans: {(good_indices)}')
            scan_alphas = np.array(scan_alphas)
            if len(good_indices)>0:
                if ga_results is not None:
                    ga_index = np.argsort(ga_results[:,0])
                    ga_heat = ga_results[:,0]
                    ga_trip = ga_results[:,1]
                    plt.plot(ga_heat[ga_index], ga_trip[ga_index], c='black', linestyle='dashed', label=f'NSGA II (HV: {ga_hv:.3f})')

                ##
                rl_points = rl_points[good_indices]
                trimmed_alphas = scan_alphas[good_indices]
                print(rl_points.shape)
                print(trimmed_alphas.shape)
                # rl_hv = np.round(metric.do(rl_points)*100/max_vol, 3)
                rl_hv = metric.do(rl_points)

                fig, ax = plt.subplots(dpi=90)
                # run_openai_log.info(f'scan_heats: {scan_heats}')
                # run_openai_log.info(f'scan_trips: {scan_trips}')
                # run_openai_log.info(f'scan_alphas: {scan_alphas}')
                #plt.scatter(scan_heats,scan_trips, s=100, c=scan_alphas)
                plt.scatter(rl_points[:, 0],rl_points[:, 1], s=50, c=trimmed_alphas, label=f'MOTD3 (HV: {rl_hv:.3f})')
                #np.sum(scan_rewards,axis=1))#scan_alphas)
                if ga_results is not None:
                    ga_index = np.argsort(ga_results[:,0])
                    ga_heat = ga_results[:,0]
                    ga_trip = ga_results[:,1]
                    plt.plot(ga_heat[ga_index], ga_trip[ga_index], c='black', linestyle='dashed', label=f'NSGA II (HV: {ga_hv:.3f})')

                # plt.xlim(20.8, 22.4)
                #plt.ylim(0.015, 0.05)
                # Change major ticks to show every 20.
                # ax.xaxis.set_major_locator(MultipleLocator(0.2))
                # ax.yaxis.set_major_locator(MultipleLocator(0.005))
                # plt.text(.95, .99, f'Total MO Reward: {inference_total_reward:.4f}',
                #          ha='right', va='top', transform=ax.transAxes)
                plt.grid()
                plt.xlabel('Heat Load [W]')
                plt.ylabel('Trip Rate [per hour]');
                plt.colorbar()
                plt.tight_layout()
                plt.legend()
                plt.savefig(logdir+f'/pareto_steps{total_nsteps}_{inference_total_reward:.4f}.jpeg')
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
                # save numpy file
                np.save(logdir+ f'/inference_results_steps{total_nsteps}.npy',
                        np.concatenate([scan_heats, scan_trips, scan_alphas]))


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
    
