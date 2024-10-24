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
import math
import psutil
from datetime import datetime
from collections import deque
import warnings

# Third-Party Imports
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import gym
import pybulletgym
from gym.wrappers import FlattenObservation, FrameStack, RescaleAction, TimeLimit

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

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    run_openai_log.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def generate_logdir(index, env_id, agent_id, use_env_subdir=False):
    """Generate a log directory path based on the given parameters.
    
    Args:
        index (int): Index for tracking multiple runs.
        env_id (str): Identifier for the environment.
        agent_id (str): Identifier for the agent.
        use_env_subdir (bool, optional): Whether to use environment as a subdirectory. Defaults to False.
    
    Returns:
        str: The generated log directory path.
    """

    githash = get_git_revision_short_hash()
    run_openai_log.debug(f"Git Hash: {githash}")

    # Format timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Construct the folder name
    folder_name = f"env_{env_id}_agent_{agent_id}_hash_{githash}_time_{timestamp}"

    # Add index to the beginning of the folder name
    folder_name = f"index{index:03d}_{folder_name}"

    # Construct the full path
    if use_env_subdir:
        logdir = os.path.join("./results", env_id, folder_name)
    else:
        logdir = os.path.join("./results", folder_name)

    run_openai_log.debug(f"Generated logdir: {logdir}")

    return logdir

def create_and_configure_env(env_id, difficulty=None, max_nsteps=0):
    """Create and configure an environment based on the given environment ID.
    
    Args:
        env_id (str): Identifier for the environment to create.
        difficulty (float, optional): Difficulty level for curriculum learning. Defaults to None.
        max_nsteps (int, optional): Maximum number of steps per episode. Defaults to 0.
    
    Returns:
        gym.Env: The created and configured environment.
    
    Raises:
        ValueError: If the environment is not found in any registered modules.
    """

    run_openai_log.info(f'Creating environment: {env_id}')

    env_creators = {
        'gym': (gym.envs.registry, gym.make),
        'custom': (custom_gym.list_registered_modules(), custom_gym.make),
        'paces': (paces_gym.list_registered_modules() if 'paces_gym' in globals() else set(), paces_gym.make if 'paces_gym' in globals() else None)
    }

    for env_type, (registry, make_func) in env_creators.items():
        if env_id in registry:
            env = make_func(env_id)
            run_openai_log.info(f'Created {env_type} environment: {env_id}')
            break
    else:
        raise ValueError(f'Environment {env_id} not found in any registered modules')

    return env

def run_opt(index, max_nepisodes, max_nsteps, agent_id, env_id, logdir, buffer_type, buffer_size, inference_flag, difficulty, nepisode_avg, model_save_threshold, use_env_subdir=False, inference_interval=10):
    """Run the optimization process for reinforcement learning.
    
    Args:
        index (int): Index for tracking multiple runs or ran.
        max_nepisodes (int): Maximum number of episodes to run.
        max_nsteps (int): Maximum number of steps per episode.
        agent_id (str): Identifier for the agent to use.
        env_id (str): Identifier for the environment to use.
        logdir (str): Directory to save logs and results.
        buffer_type (str): Type of replay buffer to use.
        buffer_size (int): Size of the replay buffer.
        inference_flag (bool): Whether to run in inference mode only.
        difficulty (float): Difficulty level for curriculum learning.
        nepisode_avg (int): Number of episodes to average over for logging and saving.
        model_save_threshold (float): Threshold for improvement to trigger model saving.
        use_env_subdir (bool, optional): Whether to use environment as a subdirectory. Defaults to False.
    
    Returns:
        None
    """
    
    # Generate Log Directory
    if logdir == 'None':
        logdir = generate_logdir(index, env_id, agent_id, use_env_subdir)
    else:
        # If a custom logdir is provided, append the generated folder name to it
        folder_name = os.path.basename(generate_logdir(index, env_id, agent_id, use_env_subdir))
        logdir = os.path.join(logdir, folder_name)

    # Environment Handling
    run_openai_log.info('Running env: {}'.format(env_id))
    
    try:
        env = create_and_configure_env(env_id, difficulty, max_nsteps)
    except ValueError as e:
        run_openai_log.error(str(e))
        return

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

    try:
        os.makedirs(logdir)
        os.makedirs(logdir + "/buffers/", exist_ok=True)
    except OSError as error:
        run_openai_log.error('Error making file:', error)

    run_openai_log.info(f'Log Path: {logdir}')
    tfb_path = os.path.join(logdir, 'metrics')
    run_openai_log.info(f'TFB Path: {tfb_path}')
    file_writer = tf.summary.create_file_writer(tfb_path)
    tfb_path = os.path.join(logdir, 'metrics')
    file_writer.set_as_default()

    # Agent Handling
    agent = jlab_opt_control.agents.make(
        agent_id, env=env, logdir=logdir, buffer_type=buffer_type, buffer_size=buffer_size)

    # Save agents initial configuration and models
    agent.save_cfg()
    agent.save("init")

    # Logging information
    total_nsteps = 0
    ep_reward_list = []
    avg_reward_list = []
    inference_reward_list = []
    best_avg_inference_reward = float('-inf')

    stats_window_size = 100
    ep_lengths = deque(maxlen=stats_window_size)
    ep_rewards = deque(maxlen=stats_window_size)

    for ep in tqdm(range(max_nepisodes), desc=f'Index {index} - Episodes'):
        time_start = time.process_time()
        episode_start_time = time.time()
        run_openai_log.info(f"Starting episode {ep} at {episode_start_time}")

        if ep % 10 == 0:
            log_memory_usage()

        # Training episode
        if not inference_flag:
            episodic_reward, steps_taken = run_episode(env, agent, train=True, max_steps=max_nsteps)
            total_nsteps += steps_taken
            ep_reward_list.append(episodic_reward)
            tf.summary.scalar('Training Reward', data=episodic_reward, step=ep)

            # Add episode length and reward to deques
            ep_lengths.append(steps_taken)
            ep_rewards.append(episodic_reward)

            # Calculate and log rolling averages
            ep_len_mean = np.mean(ep_lengths) if ep_lengths else 0
            ep_rew_mean = np.mean(ep_rewards) if ep_rewards else 0
            
            tf.summary.scalar('Training Reward', data=episodic_reward, step=ep)
            tf.summary.scalar('ep_len_mean', data=ep_len_mean, step=ep)
            tf.summary.scalar('ep_rew_mean', data=ep_rew_mean, step=ep)

        # Inference episode (run every episode if inference_flag, otherwise every 10 episodes)
        if inference_flag or ep % inference_interval == 0:
            inference_episodic_reward, inference_steps_taken = run_episode(env, agent, train=False, max_steps=max_nsteps)
            inference_reward_list.append(inference_episodic_reward)
            
            tf.summary.scalar('Inference Reward', data=inference_episodic_reward, step=ep)
            tf.summary.scalar('Inference Episode Length', data=inference_steps_taken, step=ep)

            # Calculate average of last nepisode_avg inference rewards
            avg_inference_reward = np.mean(inference_reward_list[-nepisode_avg:])

            # Model saving logic
            if avg_inference_reward > best_avg_inference_reward * (1 + model_save_threshold):
                percentage_increase = (avg_inference_reward - best_avg_inference_reward) / abs(best_avg_inference_reward) if best_avg_inference_reward != float('-inf') else 0
                str_pct_inc = f'epoch_{ep:05d}_{min(int(percentage_increase * 100), 999):03d}'
                agent.save(str_pct_inc)
                best_avg_inference_reward = avg_inference_reward
            
            run_openai_log.info(f"Episode {ep}: Inference reward: {inference_episodic_reward}, Avg Inference reward: {avg_inference_reward:.2f}, Best Avg: {best_avg_inference_reward:.2f}")

        # Periodic saving
        if ep % math.ceil(max_nepisodes/10) == 0:
            agent.save(f'epoch_{ep:05d}')
            if not inference_flag:
                agent.buffer.save(f'{logdir}/buffers/buffer.npy')

        # Logging
        avg_reward = np.mean(ep_reward_list[-nepisode_avg:])
        avg_reward_list.append(avg_reward)
        time_end = time.process_time()
        
        # Log every episode
        run_openai_log.info(f"Episode: {ep}, Total Steps: {total_nsteps}")
        run_openai_log.info(f"Episode Elapsed Time {time_end - time_start}")
        run_openai_log.info(f"Episode * {ep} * {'Inference' if inference_flag else 'Training'} Episodic Reward is ==> {episodic_reward if not inference_flag else inference_episodic_reward}")
        run_openai_log.info(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")
        run_openai_log.info(f"Episode * {ep} * Mean Episode Length is ==> {ep_len_mean}")
        run_openai_log.info(f"Episode * {ep} * Mean Episode Reward is ==> {ep_rew_mean}")

        # Save results
        with open(f'{logdir}/results.npy', 'wb') as f:
            np.save(f, np.array(ep_reward_list))

        episode_end_time = time.time()
        # run_openai_log.info(f"Finished episode {ep} at {episode_end_time}. Duration: {episode_end_time - episode_start_time:.2f} seconds")

    # Final save
    agent.save(f'epoch_{max_nepisodes:05d}')
    if not inference_flag:
        agent.buffer.save(f'{logdir}/buffers/buffer.npy')

def run_episode(env, agent, train=True, max_steps=-1):
    """Run a single episode in the given environment with the specified agent.
    
    Args:
        env (gym.Env): The environment to run the episode in.
        agent (Any): The agent to use for actions.
        train (bool, optional): Whether to train the agent during this episode. Defaults to True.
        max_steps (int, optional): Maximum number of steps for this episode. Defaults to -1 (no limit).
    
    Returns:
        Tuple[float, int]: A tuple containing the total episodic reward and the number of steps taken.
    """
    
    state = env.reset()
    episodic_reward = 0
    done = False
    steps = 0

    # Get environment dimensions
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    while not done and (max_steps == -1 or steps < max_steps):
        # Assert state shape
        assert isinstance(state, np.ndarray), f"State should be a numpy array, but got {type(state)}"
        assert state.shape == (num_states,), f"State shape should be ({num_states},), but got {state.shape}"

        action_start_time = time.time()
        action, _ = agent.action(tf.convert_to_tensor(state), train=train)
        action_end_time = time.time()
        # run_openai_log.info(f"Agent action took {action_end_time - action_start_time:.4f} seconds")

        # Assert action shape
        assert isinstance(action, np.ndarray), f"Action should be a numpy array, but got {type(action)}"
        assert action.shape == (num_actions,), f"Action shape should be ({num_actions},), but got {action.shape}"

        next_state, reward, terminate, truncate, _ = env.step(action)

        # Assert next_state shape
        assert isinstance(next_state, np.ndarray), f"Next state should be a numpy array, but got {type(next_state)}"
        assert next_state.shape == (num_states,), f"Next state shape should be ({num_states},), but got {next_state.shape}"

        # Assert reward type
        assert isinstance(reward, (int, float)), f"Reward should be a number, but got {type(reward)}"

        episodic_reward += reward
        done = terminate or truncate

        if train:
            agent.memory((state, action, reward, next_state, done))
            train_start_time = time.time()
            agent.train()
            train_end_time = time.time()
            # run_openai_log.info(f"Agent training took {train_end_time - train_start_time:.4f} seconds")

        state = next_state
        run_openai_log.info(f"Step {steps} with done signal showing: {done}")
        steps += 1

    return episodic_reward, steps

def main(args=None):
    """Main entry point of the script. Handles argument parsing and calls run_opt.
    
    Args:
        args (List[str], optional): Command line arguments. Defaults to None.
    
    Returns:
        None
    """
    
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
        "--inference", action="store_true", help="Inference only run flag")
    parser.add_argument(
        "--difficulty", help="Curriculum difficulty level for LCLS env", type=float, default=0.08)
    parser.add_argument(
        "--nepisode_avg", help="Number of episodes to average the reward over", type=int, default=20)
    parser.add_argument(
        "--model_save_threshold", help="Percentage increase threshold (in fraction) to save the model", type=float, default=0.05)
    parser.add_argument(
        "--use_env_subdir", action="store_true", help="Use environment as subdirectory in results folder")
    parser.add_argument(
        "--inference_interval", help="Use environment as subdirectory in results folder", type=int, default=10)

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
    args_use_env_subdir = args.use_env_subdir
    args_inference_interval = args.inference_interval

    run_opt(args_index, args_nepisodes, args_nsteps, args_agent_id,
            args_env_id, args_logdir, args_buf_type, args_buf_size, args_inference, args_difficulty, args_nepisode_avg, args_model_save_threshold, args_use_env_subdir, args_inference_interval)

if __name__ == "__main__":
    main()
    
