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
import seaborn as sns

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

    is_ref_plot = False

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

            if (agent.buffer_counter % agent.batch_size == 0) \
                    and (agent.buffer_counter > agent.batch_size)\
                    and (agent.buffer_counter > agent.min_buffer_counter):
                # Plot
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111)
                ax.set_title('Episode {}\n{}\n{}'.format(ep,agent_id, env_id))#, fontsize=14)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.grid(True, linestyle='-', color='0.75')
                x = agent.action_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter, 0]
                y = agent.next_state_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter, 1]
                z = agent.reward_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                a = agent.action_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                # scatter with colormap mapping to z value
                cb = ax.scatter(x, y, s=20, c=z, marker='o', cmap=cm.jet);
                plt.xlim(-1.2, 1.2)
                plt.ylim(-1.2, 1.2)
                plt.colorbar(cb)
                plt.savefig(logdir+'/xy_reward_{}.png'.format(agent.buffer_counter / nsavefig))
                plt.close()
                #
                figure, axis = plt.subplots(nrows=agent.num_actions, ncols=2, figsize=(20, 5 * agent.num_actions))
                if agent.num_actions == 1:
                    figure, axis = plt.subplots(2, figsize=(20, 16))
                    sns.kdeplot(x=np.squeeze(a), ax=axis[0],
                                color='green', fill=True, alpha=.5, linewidth=1, label='Warmup Parameter')
                    sns.kdeplot(x=np.squeeze(a), y=np.squeeze(z), ax=axis[1],
                                alpha=.5, linewidth=1, kind="kde", cmap="Purples_d", label='Warmup Parameter')
                else:
                    for i in range(agent.num_actions):
                        sns.kdeplot(x=a[:, i], ax=axis[i, 0],
                                    color='green', fill=True, alpha=.5, linewidth=1, label='Warmup Parameter')
                        sns.kdeplot(x=a[:, i], y=np.squeeze(z), ax=axis[i, 1],
                                    alpha=.5, linewidth=1, kind="kde", cmap="Purples_d", label='Warmup Parameter')
                plt.savefig(logdir + '/reward_action_dist_{}.png'.format(agent.buffer_counter / nsavefig))
                plt.close()

                if (agent.buffer_counter > agent.min_buffer_counter) and is_ref_plot==False:
                    warmup_actions = agent.action_buffer[0:agent.min_buffer_counter]
                    warmup_rewards = agent.reward_buffer[0:agent.min_buffer_counter]
                    isort_z = np.argsort(np.squeeze(warmup_rewards))
                    thr = 0.05
                    idx_thr = int( (1-thr) * agent.min_buffer_counter)
                    idx_top_z = isort_z[idx_thr:]
                    top_warmup_actions = warmup_actions[idx_top_z]
                    top_warmup_rewards = warmup_rewards[idx_top_z]
                    print(top_warmup_rewards.shape)
                    #print('top_warmup_actions:', top_warmup_actions)
                    if agent.num_actions==1:
                        figure, axis = plt.subplots(2, figsize=(20, 16))
                        sns.kdeplot(x=np.squeeze(top_warmup_actions), ax=axis[0],
                                    color='green', fill=True, alpha=.5, linewidth=1, label='Warmup Parameter')
                        sns.kdeplot(x=np.squeeze(top_warmup_actions), y=np.squeeze(top_warmup_rewards), ax=axis[1],
                                    alpha=.5, linewidth=1, kind="kde", cmap="Purples_d", label='Warmup Parameter')
                    else:
                        figure, axis = plt.subplots(nrows=agent.num_actions, ncols=2, figsize=(20, 5 * agent.num_actions))
                        for i in range(agent.num_actions):
                            sns.kdeplot(x=top_warmup_actions[:, i], ax=axis[i,0],
                                        color='green', fill=True, alpha=.5, linewidth=1, label='Warmup Parameter')
                            sns.kdeplot(x=top_warmup_actions[:, i], y=np.squeeze(top_warmup_rewards), ax=axis[i,1],
                                          alpha=.5, linewidth=1, kind="kde", cmap="Purples_d", label='Warmup Parameter')
                            if "Proxy" in env_id:
                                axis[i, 0].axvline(x=env.true_params[i], color='r', label='True Parameter')
                                axis[i, 1].axvline(x=env.true_params[i], color='r', label='True Parameter')
                                axis[i, 1].set_ylabel('Reward')
                                axis[i, 0].set_xlim(0, 1)
                                axis[i, 1].set_xlim(0, 1)
                        # sns.kdeplot(top_warmup_actions[:,i], weights=np.squeeze(top_warmup_rewards), ax=axis[i,1],
                        #             color='orange', fill=True, alpha=.5, linewidth=1, label='Warmup Parameter')
                    plt.tight_layout()
                    plt.savefig(logdir + f'/top{int(thr*100)}_warmup_action.png')
                    plt.close()
                    is_ref_plot=True

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
    parser.add_argument("--nwarmup", help="Agent warm-up size", type=int, default=0)
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