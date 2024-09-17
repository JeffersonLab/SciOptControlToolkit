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
from pymoo.indicators.hv import Hypervolume 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import (MultipleLocator)
from tqdm import tqdm

import jlab_opt_control.agents
from jlab_opt_control.utils.git_utils import get_git_revision_short_hash
from jlab_opt_control.utils.mo_utils import get_fraction_mono_tuning

warnings.filterwarnings("ignore")

run_openai_log = logging.getLogger("RunOpenAI")
run_openai_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

seed = 1  # time.time_ns()
tf.random.set_seed(seed)
np.random.seed(seed)
# run_openai_log.info(f'seeds {tf.random.}')


def run_opt(index, max_nepisodes, max_nsteps, agent_id, env_id, logdir, buffer_type, buffer_size, ga_results_loc):
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
    if 'control' in agent_id:
        agent = jlab_opt_control.agents.make(agent_id, 
                                             env=env, 
                                             logdir=logdir, 
                                             buffer_type=buffer_type, 
                                             buffer_size=buffer_size,
                                             max_nsteps=max_nsteps)
    else:
        agent = jlab_opt_control.agents.make(agent_id, 
                                             env=env, 
                                             logdir=logdir, 
                                             buffer_type=buffer_type, 
                                             buffer_size=buffer_size)

    agent.save_cfg()
    agent.save("init")

    
    if ga_results_loc is not None:
        if '8D' in env_id:
            ga_results = np.load(os.path.join(ga_results_loc, "1L10_TEST8_nsga_II_results.npy"))
            ref = [22.0, 0.04]
            ideal = [20.0, 0.01]
            metric = Hypervolume(ref_point= ref,
                         norm_ref_point=False,
                         zero_to_one=False,
                         ideal=ideal,
                         nadir=ref)
            max_vol = (ref[0] - ideal[0]) * (ref[1] - ideal[1])
            ga_hv = np.round(metric.do(ga_results)*100/max_vol, 3)
        elif '-N-' in env_id:
            # ga_results = np.load(os.path.join(ga_results_loc, "NORTH_nsga_II_results.npy"))
            ga_results = np.load(os.path.join(ga_results_loc, "NORTH_nsga_II_results.npy"))
            ref = [2530.0, 6.0]
            ideal = [2380.0, 1.0]
            metric = Hypervolume(ref_point= ref,
                         norm_ref_point=False,
                         zero_to_one=False,
                         ideal=ideal,
                         nadir=ref)
            max_vol = (ref[0] - ideal[0]) * (ref[1] - ideal[1])
            ga_hv = np.round(metric.do(ga_results)*100/max_vol, 3)
        else:
            ga_results = None
    else:
        ga_results = None

    total_nsteps = 0
    inference_best_total_reward = 0.0

    for ep in tqdm(range(1, max_nepisodes+1), desc='Index {} - Episodes'.format(index)):
        agent.train()
        #print(f'agent.batch_size: {agent.batch_size}')
        total_nsteps += 1

        # Run inference test
        if ep>=1 and total_nsteps % 1000 == 0:
            run_openai_log.info(f'Running inference ...')
            states = env.reset()[0].numpy()
            states = tf.convert_to_tensor(np.array([states]*1500))
            scans = np.random.rand(states.shape[0])
            alphas = tf.convert_to_tensor(np.stack([scans, (1-scans)*1.5], axis=1), dtype=tf.float32)
            scan_trips, scan_heats, scan_alphas, scan_rewards = [], [], [], []
            inference_total_reward = 0.0

            for step in range(max_nsteps):
                inference_actions, inference_action_noise, inference_alphas = agent.action(states, alphas, train=False)
                next_state, rewards, done, _, info = env.step(inference_actions)
                heat, trip = info['heat'], info['trip']
                scan_trips = trip.numpy()
                scan_heats = heat.numpy()
                scan_alphas = inference_alphas.numpy()
                scan_rewards = rewards.numpy()            
                inference_total_reward = np.sum(scan_rewards)
                states = next_state
                
            
            energy = np.sum(env.denormalize_state(states.numpy()) * env.linac.lengths, axis=1)
            out_of_bound = np.where((energy < env.min_energy) | (energy > env.max_energy))[0]
            scan_heats = np.delete(scan_heats, out_of_bound, axis=0)
            scan_trips = np.delete(scan_trips, out_of_bound, axis=0)
            scan_alphas = np.delete(scan_alphas, out_of_bound, axis=0)
             
            inference_total_reward = inference_total_reward/100
            run_openai_log.debug(f'inference_total_reward: {inference_total_reward} '
                                f'and inference_best_total_reward: {inference_best_total_reward}')

            if inference_best_total_reward<=inference_total_reward:
                if inference_best_total_reward>0:
                    percentage_change = (inference_total_reward - inference_best_total_reward) / abs(inference_best_total_reward)
                    run_openai_log.info(f'Improved model {inference_total_reward} by {percentage_change}')
                    str_pct_inc = 'epoch_' + str(ep) + '_' + f"{int(100 * percentage_change):03d}"
                    agent.save(str_pct_inc)
                inference_best_total_reward = inference_total_reward
                run_openai_log.info(f'Settinginference_best_total_reward to: {inference_best_total_reward}')

            tf.summary.scalar('Total MO Inference Reward', data=inference_total_reward, step=int(ep))
            print(f'Number of valid scans: {len(scan_trips)}')

            rl_points = np.stack([scan_heats, scan_trips], axis=1)
            good_indices = np.where((rl_points[:, 0] <= ref[0]) & (rl_points[:, 1] <= ref[1]))[0]
            rl_points = rl_points[good_indices]
            trimmed_alphas = scan_alphas[good_indices]
            print(rl_points.shape)
            rl_hv = np.round(metric.do(rl_points)*100/max_vol, 3)

            print("Fraction of monotonic alphas: ", get_fraction_mono_tuning(rl_points[:, 0], rl_points[:, 1], trimmed_alphas[:, 0]))
            
            if len(scan_trips)>0:
                fig, ax = plt.subplots(dpi=100)
                plt.scatter(rl_points[:, 0],rl_points[:, 1], s=50, c=trimmed_alphas[:,0], label="DDRL (HV: "+str(rl_hv)+")") #np.sum(scan_rewards,axis=1))#scan_alphas)
                if ga_results is not None:
                    ga_index = np.argsort(ga_results[:,0])
                    ga_heat = ga_results[:,0]
                    ga_trip = ga_results[:,1]
                    plt.plot(ga_heat[ga_index], ga_trip[ga_index], c='black', linestyle='dashed', label="NSGA II (HV: "+str(ga_hv)+")")
                    
                # plt.text(.95, .99, f'Total MO Reward: {inference_total_reward:.4f}',
                        #  ha='right', va='top', transform=ax.transAxes)
                plt.grid()
                plt.xlabel('Heat Load [W]')
                plt.ylabel('Trip Rate [per hour]');
                plt.colorbar()
                plt.tight_layout()
                plt.legend()
                plt.savefig(logdir+f'/pareto_ep{ep}_{inference_total_reward:.4f}.png')
                # Convert figure to an image tensor and log
                buf = io.BytesIO()
                canvas = FigureCanvasAgg(fig)
                canvas.print_png(buf)
                tensor = tf.image.decode_png(buf.getvalue(), channels=4)
                tf.summary.image(
                    "Pareto", data=tensor[None], step=int(ep)
                )
                plt.clf()
                plt.close("all")
                # save numpy file
                # print(f'scan_heats: {scan_heats.shape}')
                # print(f'scan_trips: {scan_trips.shape}')
                # print(f'scan_alphas: {scan_alphas.shape}')
                np.save(logdir + f'/inference_results_steps{total_nsteps}.npy',
                        np.concatenate([scan_heats, scan_trips, scan_alphas[:,0]]))


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument( "--index", help="Index for tracking", type=int, default=0)
    parser.add_argument( "--nepisodes", help="Number of episodes", type=int, default=500000)
    parser.add_argument("--nsteps", help="Number of steps",type=int, default=-1)
    parser.add_argument("--bsize", help="Buffer size", type=int, default=None)
    parser.add_argument("--btype", help="Buffer Type", type=str, default=None)
    parser.add_argument("--agent", help="Agent used for RL",type=str, default='MO-KerasTD3-v0')
    parser.add_argument("--env", help="Environment used for RL",type=str, default='PACES-MO-CEBAF-8D-v0')
    parser.add_argument("--logdir", help="Directory to save results", type=str, default='None')
    parser.add_argument("--ga_results", help="directory where GA results are stored", type=str, default=None)


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
    args_ga_results = args.ga_results

    run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id,
            args_env_id, args_logdir, args_buf_type, args_buf_size, args_ga_results)

if __name__ == "__main__":
    main()
    
