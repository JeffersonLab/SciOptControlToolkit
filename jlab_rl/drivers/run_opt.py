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
import matplotlib.pyplot as plt
from matplotlib import cm

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['font.family'] = [u'serif']
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = 10, 7

# import mujoco_py
# import os
# mj_path = mujoco_py.utils.discover_mujoco()
# print('mj_path:{}'.format(mj_path))
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')


# Seed value
# seed_value = 0
# os.environ['PYTHONHASHSEED'] = str(seed_value)
# random.seed(seed_value)
# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)

plasma = plt.get_cmap('GnBu_r')

def run_opt(index, max_nepisodes, max_nsteps, agent_id, warmup_size, env_id, logdir):

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

    if 'DnC2s' in env_id:
        import jlab_rl.envs as gym
    else:
        import gym
    #
    # Environment
    print('Running env: {}'.format(env_id))
    if ('HalfCheetah' or 'Hopper') in env_id:
        env = gym.make(env_id, exclude_current_positions_from_observation=False)
    elif 'Proxy' in env_id:
        env = gym.make(env_id,logdir=logdir)
        test_env = gym.make(env_id, logdir=logdir)
    else:
        env = gym.make(env_id)
        test_env = gym.make(env_id)

    env._max_episode_steps = max_nsteps

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

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

    best_heat = 9999
    for ep in tqdm(range(max_nepisodes), desc='Index {} - Episodes'.format(index)):
        time_start = time.process_time()
        prev_state, _ = env.reset()
        nsteps = 0
        episodic_reward = 0
#        for estep in tqdm(range(int(max_nsteps)), desc='Index {} - Steps'.format(index)):
        for estep in range(max_nsteps):
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
            if 'Pendulum' not in env_id:# != 'Pendulum-v1' or :
                action = np.squeeze(action)
            # if agent_id == 'KerasGenerativeTD3-v0':
            #     action = np.squeeze(action)
            # print('action: ', action.shape)
            state, reward, done_old, done, info = env.step(action)
            # done_old = float(done_old)
            # done = float(done)
            # nsteps += 1
            # if done:
            #     print('old/new done: {}/{}({})'.format(done_old, done, estep))
            agent.memory((prev_state, action, reward, state, done))
            episodic_reward += reward
            agent.train()
            prev_state = state

            if env_id == 'CEBAF2DEnv-v0':
                # Plot
                if agent.buffer_counter % nsavefig == 0 and agent.buffer_counter > 0:
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111)
                    ax.set_title('Episode {}\n{}\n{}'.format(ep,agent_id, env_id))#, fontsize=14)
                    ax.set_xlabel("X")#, fontsize=12)
                    ax.set_ylabel("Y")#, fontsize=12)
                    ax.grid(True, linestyle='-', color='0.75')
                    this_actions = agent.action_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                    this_actions = env.denormalize_state(this_actions)
                    x = this_actions[:, 0]
                    y = this_actions[:, 1]
                    # scatter with colormap mapping to z value
                    cb = ax.scatter(x, y, s=20, marker='o');
                    plt.xlim(np.min(x), np.max(x))
                    plt.ylim(np.min(y), np.max(y))
                    plt.colorbar(cb)
                    plt.savefig(logdir + '/denormalized_action_{}.png'.format(agent.buffer_counter / nsavefig))

                # Plot
                if agent.buffer_counter % nsavefig == 0 and agent.buffer_counter > 0:
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111)
                    ax.set_title('Episode {}\n{}\n{}'.format(ep,agent_id, env_id))#, fontsize=14)
                    ax.set_xlabel("X")#, fontsize=12)
                    ax.set_ylabel("Y")#, fontsize=12)
                    ax.grid(True, linestyle='-', color='0.75')
                    this_states = agent.state_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                    this_states = env.denormalize_state(this_states)
                    x = this_states[:, 0]
                    y = this_states[:, 1]
                    # scatter with colormap mapping to z value
                    cb = ax.scatter(x, y, s=20, marker='o');
                    plt.xlim(np.min(x), np.max(x))
                    plt.ylim(np.min(y), np.max(y))
                    plt.colorbar(cb)
                    plt.savefig(logdir + '/denormalized_state_{}.png'.format(agent.buffer_counter / nsavefig))

                # Plot
                if agent.buffer_counter % nsavefig == 0 and agent.buffer_counter > 0:
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111)
                    ax.set_title('Episode {}\n{}\n{}'.format(ep,agent_id, env_id))#, fontsize=14)
                    ax.set_xlabel("X")#, fontsize=12)
                    ax.set_ylabel("Y")#, fontsize=12)
                    ax.grid(True, linestyle='-', color='0.75')
                    this_next_states = agent.next_state_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                    this_next_states = env.denormalize_state(this_next_states)
                    x = this_next_states[:, 0]
                    y = this_next_states[:, 1]
                    z = agent.reward_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                    # scatter with colormap mapping to z value
                    cb = ax.scatter(x, y, s=20, c=z, marker='o', cmap=cm.jet);
                    plt.xlim(np.min(x), np.max(x))
                    plt.ylim(np.min(y), np.max(y))
                    plt.colorbar(cb)
                    plt.savefig(logdir+'/denormalized_nextstate_reward_{}.png'.format(agent.buffer_counter / nsavefig))

            # Plot the
            if agent.buffer_counter % nsavefig == 0 and agent.buffer_counter > 0:
                if "2D" in env_id:
                # Plot
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111)
                    ax.set_title('Episode {}\n{}\n{}'.format(ep,agent_id, env_id))#, fontsize=14)
                    ax.set_xlabel("X")#, fontsize=12)
                    ax.set_ylabel("Y")#, fontsize=12)
                    ax.grid(True, linestyle='-', color='0.75')
                    x = agent.next_state_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter, 0]
                    y = agent.next_state_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter, 1]
                    z = agent.reward_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                    # scatter with colormap mapping to z value
                    cb = ax.scatter(x, y, s=20, c=z, marker='o', cmap=cm.jet);
                    plt.xlim(-1.5, 1.5)
                    plt.ylim(-1.5, 1.5)
                    plt.colorbar(cb)
                    plt.savefig(logdir+'/xy_reward_{}.png'.format(agent.buffer_counter / nsavefig))
                    plt.close()
                if "Sin" in env_id:
                    # Plot
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111)
                    ax.set_title('Episode {}\n{}\n{}'.format(ep,agent_id, env_id))#, fontsize=14)
                    ax.set_xlabel("X")#, fontsize=12)
                    ax.set_ylabel("Y")#, fontsize=12)
                    ax.grid(True, linestyle='-', color='0.75')
                    x = agent.next_state_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter, 0]
                    y = np.sin(x)
                    z = agent.reward_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                    # scatter with colormap mapping to z value
                    cb = ax.scatter(x, y, s=20, c=z, marker='o', cmap=cm.jet);
                    plt.xlim(-2*np.pi*1.01, +2*np.pi*1.01)
                    plt.ylim(-1.05, +1.05)
                    plt.colorbar(cb)
                    plt.savefig(logdir + '/sinx_reward_{}.png'.format(agent.buffer_counter / nsavefig))
                    plt.close()

                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111)
                    ax.set_title('Episode {}\n{}\n{}'.format(ep,agent_id, env_id))#, fontsize=14)
                    ax.set_xlabel("X")#, fontsize=12)
                    ax.set_ylabel("Y")#, fontsize=12)
                    _ = plt.hist(x, weights=z, bins=50, range=(float(-2*np.pi*1.01), float(+2*np.pi*1.01)))
                    plt.savefig(logdir + '/hist_sinx_reward_{}.png'.format(agent.buffer_counter / nsavefig))
                    plt.close()

            # Save information
            tf.summary.scalar('Step Reward', data=episodic_reward, step=int(total_nsteps))

            if 'Sin' in env_id:
                tf.summary.scalar('Action Theta', data=action, step=int(total_nsteps))

            #
            if 'Circle' in env_id:
                radius = np.sqrt(np.sum(state*state))
                tf.summary.scalar('Radial Distribution', data=radius, step=int(total_nsteps))

            if 'Proxy' in env_id \
                    and agent.buffer_counter > np.max([agent.batch_size, agent.min_buffer_counter])\
                    and ep%100==0:
                test_actions, test_rewards = [], []
                for _ in range(100):
                    test_prev_state, _ = test_env.reset()
                    test_action, _ = agent.action(tf.convert_to_tensor(test_prev_state), train=False)
                    test_action = np.squeeze(test_action)
                    _, test_reward, _, _, _ = test_env.step(test_action)
                    test_actions.append(test_action)
                    test_rewards.append(test_reward)
                test_actions = np.array(test_actions)
                test_nactions = test_actions.shape[1]
                fig, axs = plt.subplots(test_nactions, figsize=(16,20))
                fig.suptitle(f'Parameter Episode {ep}')
                for i in range(test_nactions):
                    axs[i].hist(test_actions[:,i], bins=25, range=[0,1], label='GenAI Parameter')
                    axs[i].axvline(x=env.true_params[i], color='r', label='True Parameter')
                    axs[i].legend()
                plt.tight_layout()
                plt.savefig(logdir+f'/episode{ep}.png')

            if "CEBAF" in env_id:
                tf.summary.scalar('Energy Distribution', data=env.energy, step=int(total_nsteps))
                tf.summary.scalar('Trip Rate', data=info['trip'], step=int(total_nsteps))
                tf.summary.scalar('Heat Load', data=info['heat'], step=int(total_nsteps))
                tf.summary.scalar('Reward', data=reward, step=int(total_nsteps))
                if info['valid']:
                    tf.summary.scalar('Valid Energy Distribution', data=env.energy, step=int(total_nsteps))
                    tf.summary.scalar('Valid Trip Rate', data=info['trip'], step=int(total_nsteps))
                    tf.summary.scalar('Valid Heat Load', data=info['heat'], step=int(total_nsteps))
                    tf.summary.scalar('Valid Reward', data=reward, step=int(total_nsteps))
                # Ideal results
                ideal_action, _ = agent.action(tf.convert_to_tensor(prev_state), train=False)
                ideal_action = np.squeeze(ideal_action)
                ideal_state, ideal_reward, _, _, ideal_info = env.step(ideal_action)
                tf.summary.scalar('Ideal Energy Distribution', data=ideal_info['energy'], step=int(total_nsteps))
                tf.summary.scalar('Ideal Trip Rate', data=ideal_info['trip'], step=int(total_nsteps))
                tf.summary.scalar('Ideal Heat Load', data=ideal_info['heat'], step=int(total_nsteps))
                tf.summary.scalar('Ideal Reward', data=ideal_reward, step=int(total_nsteps))

                if best_heat > ideal_info['heat'] and ideal_info['valid']:
                    best_heat = ideal_info['heat']
                    print('Best heat:', best_heat)
                # Plot Pareto front
                if agent.buffer_counter%100==0:
                    print('Testing Optimal Solution...')
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(111)
                    test_heats, test_trips, test_rewards = [], [], []
                    for _ in range(100):
                        test_prev_state, _ = test_env.reset()
                        test_action, _ = agent.action(tf.convert_to_tensor(test_prev_state), train=False)
                        test_action = np.squeeze(test_action)
                        test_state, test_reward, test_done_old, test_done, test_info = test_env.step(test_action)
                        test_heat = test_info['heat']
                        test_trip = test_info['trip']
                        test_energy = test_info['energy']
                        if test_energy > test_env.min_energy and test_energy < test_env.max_energy:
                            test_heats.append(test_heat)
                            test_trips.append(test_trip)
                            test_rewards.append(float(test_reward))
                    ax.set_title('Episode {}\n{}\n{}'.format(ep,agent_id, env_id))#, fontsize=14)
                    cb = plt.scatter(test_heats, test_trips, c=test_rewards, cmap=cm.jet)
                    #'o', c=test_rewards[0], marker='o', cmap=cm.jet)
                    plt.xlim(21.15, 22.65)
                    plt.ylim(0.01, 0.4)
                    ax.set_xlabel("Heat Load [W]")#, fontsize=12)
                    ax.set_ylabel("Trip Rate [per hour]")#, fontsize=12)
                    plt.grid()
                    plt.colorbar(cb)
                    plt.savefig(logdir + '/pareto_{}.png'.format(agent.buffer_counter/100))
                    plt.close()
            # End this episode when `done` is True
            if done_old:
                break

            # End this episode when `done` is True
            if done:
                break

        if 'Gaussian' in env_id and ep % 1000 == 0 and ep > 0:
            predictions = []
            for t in range(env.true_params.shape[0]):
                st, _ = env.reset()
                prediction, _ = agent.action(tf.convert_to_tensor(st), train=False)
                predictions.append(np.squeeze(prediction))
            predictions = np.squeeze(predictions)
            # tf.summary.histogram('predictions', data=predictions, step=int(total_nsteps))
            # tf.summary.histogram('real_data', data=env.data, step=int(total_nsteps))
            plt.clf()
            plt.figure(figsize=(8,5))
            plt.hist(predictions, bins=100, range=(0, 1), histtype='step', color = 'red', label="GAN")
            plt.title('param_at_epoch'+str(ep).zfill(6))
            plt.hist(env.true_params, bins=100, range=(0, 1), histtype='step', color='green', label="True")
            plt.savefig(os.path.join("Params_"+str(ep).zfill(6)+".png"))
            plt.legend()
            plt.show()

        ep_reward_list.append(episodic_reward)
        tf.summary.scalar('Reward', data=episodic_reward, step=int(ep))

        # Mean of last 40 episodes
        nepisode_mod = 10
        avg_reward = np.mean(ep_reward_list[-nepisode_mod:])
        time_end = time.process_time()
        if total_nsteps%1000==0:
            print("\nEpisode Elapsed Time {}".format((time_end - time_start)))
            print("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        with open(logdir+'/test.npy', 'wb') as f:
            np.save(f, np.array(ep_reward_list))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index for tracking", type=int, default=0)
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=1000)
    parser.add_argument("--nsteps", help="Number of steps", type=int, default=200)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasTD3-v0')
    parser.add_argument("--nwarmup", help="Agent warm-up size", type=int, default=1000)
    parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')
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
    args_profile = args.profile

    profiler = None
    if args_profile:
        import cProfile
        import pstats
        print('###### Overriding setting to run profiling ###### ')
        args_nepisodes = 10
        args_nsteps = 25
        args_warmup_size = 0
        profiler = cProfile.Profile()
        profiler.enable()

    run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id, args_warmup_size, args_env_id, args_logdir)

    if args_profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        # Print the stats report
        stats.print_stats()