import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
# import torch
import jlab_rl.agents
from jlab_rl.utils.git_utilts import get_git_revision_short_hash
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy import stats

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

plasma = plt.get_cmap('GnBu_r')

def get_rchi2(obs, true, bins=10):
    true_counts, top_bins = np.histogram(true, range=[-1, 1], bins=bins)
    counts, bins = np.histogram(obs, range=[-1, 1], bins=bins)
    rchi2 = np.sum(np.square(true_counts - counts) / true_counts) / (len(true_counts) - 1)
    return rchi2

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
    if ('HalfCheetah-v4' or 'Hopper') in env_id:
        env = gym.make(env_id, exclude_current_positions_from_observation=False)
    elif 'Proxy' in env_id:
        env = gym.make(env_id,logdir=logdir)
    else:
        env = gym.make(env_id)

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
    ep_chi2_warmup_to_truth = []
    ep_chi2_inference_to_warmup = []
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
        for estep in range(max_nsteps):
            total_nsteps += 1
            action, action_type = agent.action(tf.convert_to_tensor(prev_state))
            assert 'numpy.ndarray' in str(type(action))
            # assert action.shape == (num_actions,), print("Action shape does not match: ", action.shape)
            # TODO: We suspect this is to the the num_actions > 1
            if env_id == "LunarLanderContinuous-v2":
                action = action[0]

            # Receive state and reward from environment.
            if 'Pendulum' in env_id:
                action = [action]
            else:
                action = np.squeeze(action)

            state, reward, done_old, done, info = env.step(action)

            # Check shapes and data types
            assert 'numpy.ndarray' in str(type(state))
            assert state.shape == (num_states,)
            assert 'numpy.float' in str(type(reward)), str(type(reward))

            agent.memory((prev_state, action, reward, state, done, action_type))
            episodic_reward += reward
            agent.train()
            prev_state = state

            if (agent.buffer_counter % agent.batch_size == 0) \
                and (agent.buffer_counter >= agent.batch_size):#\
                #and (agent.buffer_counter >= agent.min_buffer_counter):
                # Plot
                z = agent.reward_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                a = agent.action_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                #print(f'shape {a.shape}')
                # Only does for 2D problem(s)
                if agent.next_state_buffer.shape[1] == 2:
                    # Latest buffer
                    #action_types = agent.action_type_buffer[agent.buffer_counter - nsavefig:agent.buffer_counter]
                    x = a[:,0]
                    y = a[:,1]
                    #sample_idx = np.where(action_types==0)[0]
                    sample_x = x#[sample_idx]
                    sample_y = y#[sample_idx]
                    sample_z = z#[sample_idx]

                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(111)
                    ax.set_title(f'Sampled {sample_z.shape[0]} \n Average Reward: {np.mean(sample_z):.5f}')
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.grid(True, linestyle='-', color='0.75')
                    # scatter with colormap mapping to z value
                    cb = ax.scatter(sample_x, sample_y, s=35, c=sample_z, marker='o',
                                    vmin=0.0,vmax=1, cmap=cm.jet);
                    plt.xlim(-1.1, 1.1)
                    plt.ylim(-1.1, 1.1)
                    plt.colorbar(cb)
                    plt.savefig(logdir+'/sampled_xy_action_reward_{}.png'.format(agent.buffer_counter / nsavefig))
                    plt.close()

                    # Inference

                    inference_states = agent.state_buffer[0:agent.batch_size]
                    # Long test
                    inference_actions, inference_rewards = agent.action_inference(1000)
                    # Quick test
                    #inference_actions, inference_rewards = agent.action_inference(10000)

                    #sys.exit()
#                    inference_actions, inference_rewards = agent.action_inference(inference_states)
                    policy_x = inference_actions[:, 0]
                    policy_y = inference_actions[:, 1]
                    policy_z = inference_rewards
                    # print(f'policy_x: {policy_x.shape}')
                    # print(f'policy_y: {policy_y.shape}')
                    #print(f'policy_z: {policy_z.shape}')

                    #tf.abs(tf.sqrt(tf.reduce_sum(tf.square(inference_actions), axis=1))-env.target_value)
                    #print(f'inference_rewards: {inference_rewards}')
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(111)
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.grid(True, linestyle='-', color='0.75')
                    # scatter with colormap mapping to z value
                    cb = ax.scatter(policy_x, policy_y, s=35, c=policy_z, marker='o',
                                    vmin=0.0, vmax=1, cmap=cm.jet);
                    plt.xlim(-1.1, 1.1)
                    plt.ylim(-1.1, 1.1)
                    plt.colorbar(cb)
                    plt.tight_layout
                    ax.set_title(f'Average Reward {np.mean(policy_z):.5f}')

                    plt.savefig(logdir+'/inference_xy_action_reward_{}.png'.format(agent.buffer_counter / nsavefig))
                    plt.close()

                #
                if 'KerasGenerative' in agent_id:
                    figure, axis = plt.subplots(nrows=agent.num_actions, ncols=1, figsize=(20, 5 * agent.num_actions))

                    if agent.num_actions == 1:
                        if 'sin' in env_id.lower():
                            set_range = [-1.5*np.pi, 1.5*np.pi]
                        elif 'square' in env_id.lower():
                            set_range = [-1.5, 1.5]

                        #figure, axis = plt.subplots(1, figsize=(12, 10))
                        ref_counts, ref_bins, _ = plt.hist(agent.top_actions, bins=27, linewidth=3, density=True, alpha=1, histtype='step', range=set_range,
                                         color='red', label='Reference')
                        model_counts, model_bins, _ = plt.hist(agent.training_actions, bins=27, linewidth=3, density=True, alpha=1, histtype='step', range=set_range,
                                         color='blue', label='Inference')
                        # sns.kdeplot(x=agent.top_actions[:, 0], ax=axis[0],
                        #                 color='red', fill=False, alpha=.75, linewidth=3, bw_adjust=0.5,
                        #                 label='Reference')
                        # sns.kdeplot(x=np.squeeze(a), ax=axis[0],
                        #             color='green', fill=False, alpha=.5, linewidth=3, bw_adjust=0.5, label='Warmup Parameter')
                        #p_val = stats.ttest_ind(agent.top_actions[:, 0], np.squeeze(a)).pvalue
                        #axis[0].set_title("P-value: "+str(np.round(p_val, 4)))
                        # rchi2 = get_rchi2(agent.top_actions[:, 0], np.squeeze(a))
                        inf_rchi2 = np.sum(np.square(ref_counts-model_counts)/(ref_counts+1)) #/(len(top_counts)-1)
                        legend_title=r'$\chi^{2}_{\nu}$ Fit: '+str(np.round(inf_rchi2, 2))
                        plt.legend(title=legend_title)
                        plt.xlabel("Action")
                        plt.tight_layout()
                    else:
                        for i in range(agent.num_actions):
                            # axis[i,0].title.set_text(f'Action #{i}: {agent.scores[i]}')
                            # sns.kdeplot(x=agent.top_actions[:,i], ax=axis[i, 0],
                            #             color='red', fill=False, alpha=.75, linewidth=3, bw_adjust=0.5,
                            #             label='Reference')
                            # sns.kdeplot(x=agent.training_actions[:,i], ax=axis[i, 0],
                            #             color='blue', fill=False, alpha=.25, linewidth=3, bw_adjust=0.5,label='Current')
                            ref_counts, ref_bins, _ = axis[i].hist(agent.top_actions[:, i], bins=25, range=[-1, 1],density=True,
                                                                alpha=1, linewidth=3, histtype='step', color='red', label='Reference')
                            model_counts, model_bins, _ = axis[i].hist(agent.training_actions[:, i], bins=25, range=[-1, 1], density=True,
                                                                       alpha=1, linewidth=3, histtype='step', color='blue', label='Inference')
                            inf_rchi2 = np.sum(np.square(ref_counts-model_counts)/ref_counts)#/(len(ref_counts)-1)
                            axis[i].set_xlabel(f'Action #{i}')
                            legend_title=r'$\chi^{2}_{\nu}$ Fit: '+str(np.round(inf_rchi2, 2))
                            axis[i].legend(title=legend_title)
                            plt.tight_layout()
                            # top_counts, top_bins = np.histogram(agent.top_actions[:,i], range=[-1,1], bins=25)
                            # counts, bins = np.histogram(agent.training_actions[:,i], range=[-1,1], bins=25)
                            #rchi2 = get_rchi2(agent.top_actions[:, i], agent.training_actions[:,i])
                            #print('my chi2:', chi2)
                            #from scipy.stats import chisquare
                            #chi2, p = chisquare(top_counts, counts)
                            #print('stats chi2:', chi2)
                            #print('stats p:', p)
                            #p_val = stats.ttest_ind(agent.top_actions[:, i], agent.training_actions[:, i]).pvalue
                            #axis[i, 0].set_title("P-value: "+str(np.round(p_val, 4)))
                            #axis[i].set_title(r'$\chi^{2}_{\nu}$: '+str(np.round(rchi2, 4)) )# +\
                            #                    r'$p-value$: ' + f'{p:.2E}')
                            #
                            # sns.kdeplot(x=a[:, i], y=np.squeeze(z), ax=axis[i, 1],
                            #             alpha=.5, linewidth=1, kind="kde", cmap="Purples_d", bw_adjust=0.5,label='Warmup Parameter')
                    #plt.suptitle(f'Episode {ep}')
                    plt.savefig(logdir + '/training_action_reward_dist_{}.png'.format(agent.buffer_counter / nsavefig))
                    plt.close()

                if ('ECGTD3' in agent_id or 'KerasGenerative' in agent_id) and (agent.buffer_counter >= agent.min_buffer_counter) and is_ref_plot==False:
                    top_warmup_actions = agent.top_actions
                    top_warmup_rewards = agent.top_rewards
                    print(top_warmup_rewards.shape)
                    #print('top_warmup_actions:', top_warmup_actions)
                    if 'sin' in env_id.lower():
                        ref_truth = np.concatenate([[-np.pi]*333, [0]*334, [np.pi]*333])
                    elif 'square' in env_id.lower():
                        ref_truth = np.concatenate([[-1]*500, [1]*500])
                    elif 'circle' in env_id.lower():
                        theta = np.linspace(0, 2*np.pi, num=1000)
                        r = np.array([1.]*1000)
                        x_ref = r * np.cos(theta)
                        y_ref = r * np.sin(theta)
                        ref_truth = [x_ref, y_ref]
                    if agent.num_actions==1:
                        figure, axis = plt.subplots(nrows=agent.num_actions, ncols=1, figsize=(20, 5 * agent.num_actions))
                        # figure, axis = plt.subplots(2, figsize=(12, 10))
                        # sns.kdeplot(x=ref_truth, ax=axis[0],
                        #             color='red', fill=False, alpha=1, linewidth=3, bw_adjust=0.5,
                        #             label='True Distribution')
                        # sns.kdeplot(x=np.squeeze(top_warmup_actions), ax=axis[0],
                        #             color='green', fill=False, alpha=1, linewidth=3, bw_adjust=0.5, label='Reference')
                        # axis[0].legend()
                        # # p_val = stats.ttest_ind(ref_truth, np.squeeze(top_warmup_actions)).pvalue
                        # # axis[0].set_title("P-value: "+str(np.round(p_val, 4)))
                        # rchi2 = get_rchi2(ref_truth, np.squeeze(top_warmup_actions))
                        # #axis[0].set_title(r'$\chi^{2}_{\nu}$: '+str(np.round(rchi2, 4)) )

                        # sns.kdeplot(x=np.squeeze(top_warmup_actions), y=np.squeeze(top_warmup_rewards), ax=axis[1],
                        #             alpha=.5, linewidth=1, kind="kde", cmap="Purples_d", bw_adjust=0.5, label='Warmup Parameter')
                        if 'sin' in env_id.lower():
                            set_range = [-1.5*np.pi, 1.5*np.pi]
                        elif 'square' in env_id.lower():
                            set_range = [-1.5, 1.5]

                        ref_counts, ref_bins, _ = plt.hist(ref_truth, bins=27, linewidth=3, density=True, alpha=1, histtype='step', range=set_range,
                                         color='black', label='Truth')
                        top_counts, model_bins, _ = plt.hist(np.squeeze(top_warmup_actions), bins=27, linewidth=3, density=True, alpha=1, histtype='step', range=set_range,
                                         color='red', label='Reference')
                        # sns.kdeplot(x=agent.top_actions[:, 0], ax=axis[0],
                        #                 color='red', fill=False, alpha=.75, linewidth=3, bw_adjust=0.5,
                        #                 label='Reference')
                        # sns.kdeplot(x=np.squeeze(a), ax=axis[0],
                        #             color='green', fill=False, alpha=.5, linewidth=3, bw_adjust=0.5, label='Warmup Parameter')
                        #p_val = stats.ttest_ind(agent.top_actions[:, 0], np.squeeze(a)).pvalue
                        #axis[0].set_title("P-value: "+str(np.round(p_val, 4)))
                        # rchi2 = get_rchi2(agent.top_actions[:, 0], np.squeeze(a))
                        warmup_rchi2 = np.sum(np.square(top_counts-ref_counts)/(top_counts+1)) #/(len(top_counts)-1)
                        legend_title=r'$\chi^{2}_{\nu}$ Fit: '+str(np.round(warmup_rchi2, 2))
                        plt.legend(title=legend_title)
                        plt.xlabel("Action")
                        plt.tight_layout()
                    else:
                        figure, axis = plt.subplots(nrows=agent.num_actions, ncols=1, figsize=(20, 5 * agent.num_actions))
                        for i in range(agent.num_actions):
                            # sns.kdeplot(x=ref_truth[i], ax=axis[i, 0],
                            #         color='red', fill=False, alpha=.75, linewidth=3, bw_adjust=0.5,
                            #         label='True Distribution')
                            # sns.kdeplot(x=top_warmup_actions[:, i], ax=axis[i,0],
                            #             color='green', fill=False, alpha=.5, linewidth=3, bw_adjust=0.5, label='Reference')
                            # sns.kdeplot(x=top_warmup_actions[:, i], y=np.squeeze(top_warmup_rewards), ax=axis[i,1],
                            #               alpha=.5, linewidth=1, kind="kde", cmap="Purples_d", bw_adjust=0.25,  label='Warmup Parameter')
                            # axis[i, 0].legend()
                            top_counts, _, _ = axis[i].hist(top_warmup_actions[:, i], bins=25, range=[-1, 1],
                                                            alpha=1, linewidth=3, histtype='step', color='red',
                                                            label='Reference', density=True)
                            ref_counts, _, _ = axis[i].hist(ref_truth[i], bins=25, range=[-1, 1],
                                                            alpha=1, linewidth=3, histtype='step', color='black',
                                                            label='Truth', density=True)
                            warmup_rchi2 = np.sum(np.square(top_counts-ref_counts)/top_counts)#/(len(top_counts)-1)
                            axis[i].set_xlabel(f'Action #{i}')
                            legend_title=r'$\chi^{2}_{\nu}$ Fit: '+str(np.round(warmup_rchi2, 2))
                            # p_val = stats.ttest_ind(ref_truth[i], np.squeeze(top_warmup_actions[:, i])).pvalue
                            # axis[i, 0].set_title("P-value: "+str(np.round(p_val, 4)))
                            #rchi2 = get_rchi2(ref_truth[i], np.squeeze(top_warmup_actions[:, i]))
                            axis[i].legend(title=legend_title)
                            #axis[i].set_title(r'$\chi^{2}_{\nu}$: ' + str(np.round(rchi2, 4)))
                            # if "Proxy" in env_id:
                            #     axis[i].axvline(x=env.true_params[i], color='r', label='True Parameter')
                            #     axis[i, 1].axvline(x=env.true_params[i], color='r', label='True Parameter')
                            #     axis[i, 1].set_ylabel('Reward')
                            #     axis[i, 0].set_xlim(0, 1)
                            #     axis[i, 1].set_xlim(0, 1)

                    plt.tight_layout()
                    plt.savefig(logdir + f'/top{int(agent.batch_size)}_action_dist_{agent.buffer_counter / nsavefig}.png')
                    plt.close()

                    if agent.num_actions == 2:
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(111)
                        ax.set_title('Top Episode {}\n{}\n{}'.format(ep, agent_id, env_id))
                        ax.set_xlabel("X")
                        ax.set_ylabel("Y")
                        ax.grid(True, linestyle='-', color='0.75')
                        x = top_warmup_actions[:, 0]
                        y = top_warmup_actions[:, 1]
                        # scatter with colormap mapping to z value
                        cb = ax.scatter(x, y, s=35, c=top_warmup_rewards, marker='o'
                                        ,vmin=0.0,vmax=1, cmap=cm.jet);
                        plt.xlim(-1.2, 1.2)
                        plt.ylim(-1.2, 1.2)
                        plt.colorbar(cb)
                        plt.savefig(logdir + f'/top{int(agent.batch_size)}_xy_action_reward_{agent.buffer_counter / nsavefig}.png')
                        plt.close()
                    # is_ref_plot=True
                    
                    # Append chi2 to list
                    ep_chi2_warmup_to_truth.append(warmup_rchi2)
                    ep_chi2_inference_to_warmup.append(inf_rchi2)

            # End this episode when `done` is True
            if done_old:
                break

            # End this episode when `done` is True
            if done:
                break

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

        with open(logdir+'/results.npy', 'wb') as f:
            np.save(f, np.array(ep_reward_list))
        with open(logdir+'/inf_chi2.npy', 'wb') as f:
            np.save(f, np.array(ep_chi2_inference_to_warmup))
        with open(logdir+'/warmup_chi2.npy', 'wb') as f:
            np.save(f, np.array(ep_chi2_warmup_to_truth))



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