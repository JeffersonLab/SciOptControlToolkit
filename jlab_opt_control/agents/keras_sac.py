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

import logging
import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
import jlab_opt_control.buffers
import jlab_opt_control.models
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from os.path import join
import time
import json
import platform
import sys
import shutil
processor = platform.processor()

sac_log = logging.getLogger("SAC-Agent")
sac_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class KerasSAC(jlab_opt_control.Agent):

    def __init__(self, env, logdir, buffer_type=None, buffer_size=None, cfg='keras_sac.json'):
        """ Define all key variables required for all agent """

        # Get env info
        self.target_critic2 = None
        self.critic_model2 = None
        self.target_critic1 = None
        self.critic_model1 = None
        # self.target_actor = None
        self.actor_model = None
        sac_log.info('Running KerasSAC __init__')

        # Environment setup
        self.env = env
        try:
            assert "Box" in str(type(env.action_space)), 'Invalid action space'
            self.num_states = env.observation_space.shape[0]
            self.num_actions = env.action_space.shape[0]
            self.upper_bound = env.action_space.high
            self.lower_bound = env.action_space.low
            sac_log.info(f'Action upper bound: {self.upper_bound}')
            sac_log.info(
                f'Action upper bound: {float(env.action_space.high[0])}')
            sac_log.info(f'Action lower bound: {self.lower_bound}')
            sac_log.info(
                f'Action lower bound: {float(env.action_space.low[0])}')
            self.range = self.upper_bound - self.lower_bound
            sac_log.info(f'Action range: {self.range}')
        except:
            sac_log.error('Action space not valid for this agent.')
            sys.exit(0)

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)
        sac_log.debug(f'pfn_json_file:{self.pfn_json_file}')
        with open(self.pfn_json_file) as json_file:
            data = json.load(json_file)
        self.warmup_size = int(cfg_utils.cfg_get(data, 'warmup_size', 10000))
        self.batch_size = int(cfg_utils.cfg_get(data, 'batch_size', 100))
        self.model_load_path = cfg_utils.cfg_get(data, 'load_model', None)

        self.actor_model_type = cfg_utils.cfg_get(
            data, 'actor_model', "actor_gaussian-v0")
        self.critic_model_type = cfg_utils.cfg_get(
            data, 'critic_model', "critic_fcnn-v0")

        self.logdir = logdir

        self.mse_loss = tf.keras.losses.MeanSquaredError()

        # Buffer
        if buffer_type is None:
            self.buffer_type = cfg_utils.cfg_get(data, 'buffer_type', None)
        else:
            self.buffer_type = buffer_type

        self.buffer = jlab_opt_control.buffers.make(
            self.buffer_type, state_dim=self.num_states, action_dim=self.num_actions, logdir=self.logdir, buffer_size=buffer_size)
        self.buffer.save_cfg()

        # Used to update target networks
        self.tau = float(cfg_utils.cfg_get(data, 'tau', 0.005))
        self.gamma = float(cfg_utils.cfg_get(data, 'discount', 0.99))
        self.alpha = float(cfg_utils.cfg_get(data, 'alpha', 0.2))
        self.target_update_interval = int(cfg_utils.cfg_get(data, 'target_update_interval', 1))
        self.automatic_entropy_tuning = bool(cfg_utils.cfg_get(data, 'automatic_entropy_tuning', False))


        # Setup Optimizers
        self.critic_lr = float(cfg_utils.cfg_get(
            data, 'critic_learning_rate', 5e-4))
        self.actor_lr = float(cfg_utils.cfg_get(
            data, 'actor_learning_rate', 1e-4))

        if processor == 'arm':
            sac_log.info('Using legacy Adam')
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(
                self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(
                self.actor_lr, epsilon=1e-08)
            self.alpha_optimizer = tf.keras.optimizers.legacy.Adam(
                self.actor_lr, epsilon=1e-08) #Adam([self.log_alpha], lr=args.lr)
        else:
            self.critic_optimizer = tf.keras.optimizers.Adam(
                self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.Adam(
                self.actor_lr, epsilon=1e-08)
            self.alpha_optimizer = tf.keras.optimizers.Adam(
                self.actor_lr, epsilon=1e-08) #Adam([self.log_alpha], lr=args.lr)

        self.initialize_new_models()

        # Load models for retraining
        if self.model_load_path is not None:
            self.load()

        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq = int(
            cfg_utils.cfg_get(data, 'actor_update_freq', 2))
        self.critic_update_freq = int(
            cfg_utils.cfg_get(data, 'critic_update_freq', 2))

        self.noise_clip = 0.5

        try:
            os.mkdir(self.logdir)
        except OSError as error:
            sac_log.warning(error)
        file_writer = tf.summary.create_file_writer(self.logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = 0
        self.inf_nactions = 0

        if self.actor_model_type.lower() == "gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -self.num_actions #-torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = tf.Variable(tf.ones(1)) #torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False


    def initialize_new_models(self):
        """ Initialize new models from scratch """
        sac_log.info('Running KerasSAC initialize_new_models()')

        self.actor_model = jlab_opt_control.models.make(
            self.actor_model_type, state_dim=self.num_states, action_dim=self.num_actions, min_action=self.lower_bound, max_action=self.upper_bound, logdir=self.logdir)
        
        # Run through model once to initialize variables
        self.actor_model(tf.zeros([1, self.num_states]))

        self.actor_model.save_cfg()

        seed1 = time.time_ns()
        str_seed1 = str(seed1)
        seed1 = int(str_seed1[9:-3])
        sac_log.debug(f'seed1:{seed1}')
        tf.random.set_seed(seed1)

        self.critic_model1 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, logdir=self.logdir)
        self.target_critic1 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, logdir=self.logdir)

        # Run through model once to initialize variables
        self.critic_model1(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_actions]))
        self.target_critic1(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_actions]))

        self.critic_model1.save_cfg()

        time.sleep(1 / 10)
        seed2 = time.time_ns()
        str_seed2 = str(seed2)
        seed2 = int(str_seed2[9:-3])
        sac_log.debug(f'seed2:{seed2}')
        tf.random.set_seed(seed2)

        self.critic_model2 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, logdir=self.logdir)
        self.target_critic2 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, logdir=self.logdir)

        # Run through model once to initialize variables
        self.critic_model2(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_actions]))
        self.target_critic2(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_actions]))

        self.target_critic1.set_weights(self.critic_model1.get_weights())
        self.target_critic2.set_weights(self.critic_model2.get_weights())

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones, weights):
        
        next_actions, log_pi = self.actor_model(next_states, training=False)

        target_q1 = self.target_critic1(
            next_states, next_actions, training=False)
        target_q2 = self.target_critic2(
            next_states, next_actions, training=False)
        target_q = tf.math.minimum(target_q1, target_q2)

        # Add the entropy term to get soft Q target
        target_q = target_q - self.alpha * log_pi

        # Bellman equation for the q value
        q_targets = rewards + self.gamma * target_q * (1.0 - dones)

        # Critic 1 and 2
        with tf.GradientTape() as tape:
        
            q_values1 = self.critic_model1(states, actions, training=True)
            q_values2 = self.critic_model2(states, actions, training=True)

            td_errors1 = q_values1 - q_targets
            td_errors2 = q_values2 - q_targets

            critic_loss1 = self.mse_loss(
                q_values1, q_targets, sample_weight=weights)
            critic_loss2 = self.mse_loss(
                q_values2, q_targets, sample_weight=weights)
            
            critic_losses = critic_loss1 + critic_loss2

        gradients = tape.gradient(
            critic_losses, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(
            gradients, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables))

        td_errors_avg = (tf.abs(td_errors1) + tf.abs(td_errors2)) / 2

        return critic_loss1, critic_loss2, td_errors_avg

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            actions, log_pi = self.actor_model(states)
            q_value1 = self.critic_model1(states, actions, training=False)
            q_value2 = self.critic_model2(states, actions, training=False)
            q_value = tf.math.minimum(q_value1, q_value2)

            # Add the entropy term to get soft Q target
            q_value = q_value - self.alpha * log_pi

            loss = -tf.math.reduce_mean(q_value)

        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradient, self.actor_model.trainable_variables))
        return loss

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau +
                                 target_weight * (1.0 - self.tau))

    @tf.function   
    def update_alpha(self, states):
        if self.automatic_entropy_tuning:
            with tf.GradientTape() as tape:
                # Sample actions from the policy for current states
                actions, log_pi = self.actor_model(states, training=False)

                alpha_loss = tf.reduce_mean(- self.log_alpha*(log_pi +
                                                        self.target_entropy))

            variables = [self.log_alpha]
            grads = tape.gradient(alpha_loss, variables)
            self.alpha_optimizer.apply_gradients(zip(grads, variables))

            self.alpha = tf.exp(self.log_alpha).item()
        else:
            alpha_loss = 0.

        return alpha_loss

    def train(self):
        """ Method used to train """
        self.ntrain_calls += 1

        if self.buffer.size() > np.max([self.batch_size, self.warmup_size]):

            # Get samples
            states, actions, rewards, next_states, dones, weights = self.buffer.sample(
                self.batch_size)

            # Convert to tensors
            state_batch = tf.convert_to_tensor(states, dtype=tf.float32)
            action_batch = tf.convert_to_tensor(actions, dtype=tf.float32)
            reward_batch = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(next_states, dtype=tf.float32)
            done_batch = tf.convert_to_tensor(dones, dtype=tf.float32)
            weights_batch = tf.convert_to_tensor(weights, dtype=tf.float32)

            # Train critic
            critic_loss1, critic_loss2, td_errors = self.train_critic(state_batch, action_batch, reward_batch,
                                                                          next_state_batch, done_batch, weights_batch)

            tf.summary.scalar('Critic Loss 1', data=critic_loss1,
                              step=int(self.ntrain_calls))
            tf.summary.scalar('Critic Loss 2', data=critic_loss2,
                              step=int(self.ntrain_calls))

            # Update Priorities
            if "PER" in self.buffer_type:
                new_priorities = td_errors.numpy()
                self.buffer.update_priorities(new_priorities)

            if self.ntrain_calls % self.actor_update_freq == 0:
                actor_loss = self.train_actor(state_batch)
                tf.summary.scalar('Actor Loss', data=actor_loss,
                                  step=int(self.ntrain_calls))
                
                alpha_loss = self.update_alpha(state_batch)
                tf.summary.scalar('Alpha Loss', data=alpha_loss,
                                  step=int(self.ntrain_calls))


            if self.ntrain_calls % self.critic_update_freq == 0:
                self.soft_update(self.target_critic1.variables,
                                 self.critic_model1.variables)
                self.soft_update(self.target_critic2.variables,
                                 self.critic_model2.variables)

    def action(self, state, train=True, inference=False):
        """ Method used to provide the next action using the target model """
        # Warmup experience sample
        if (self.buffer.size() < np.max([self.batch_size, self.warmup_size])) and inference == False:
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
        # Warmup completed, sample from actor or run inference
        else:
            state = tf.expand_dims(state, 0)
            sampled_action, noise = self.actor_model(state)
            sampled_action = sampled_action.numpy().flatten()
            noise = noise.numpy().flatten()


        # Log the training action(s) taken
        if train:
            self.nactions = self.nactions + 1
            if self.num_actions == 0:
                tf.summary.scalar('Action', data=sampled_action,
                                  step=int(self.nactions))
            else:
                for i in range(self.num_actions):
                    tf.summary.scalar('Action #{}'.format(
                        i), data=sampled_action[i], step=int(self.nactions))
        if inference:
            self.inf_nactions = self.inf_nactions + 1
            if self.num_actions == 0:
                tf.summary.scalar('Inference Action', data=sampled_action,
                                  step=int(self.inf_nactions))
            else:
                for i in range(self.num_actions):
                    tf.summary.scalar('Inference Action #{}'.format(
                        i), data=sampled_action[i], step=int(self.inf_nactions))
        # Insure action output by actor is in legal environment range
        return sampled_action, noise

    def memory(self, obs_tuple):
        memory_with_default_priority = obs_tuple + (self.buffer.max_priority,)
        self.buffer.record(memory_with_default_priority)


    def load(self):
        """ Load the ML models """
        try:
            model_load_count = 0
            for file in os.listdir(self.model_load_path):
                if 'actor_model' in file and file.endswith('.h5'):
                    self.actor_model.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif 'critic_model1' in file and file.endswith('.h5'):
                    self.critic_model1.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif 'target_critic1' in file and file.endswith('.h5'):
                    self.target_critic1.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif 'critic_model2' in file and file.endswith('.h5'):
                    self.critic_model2.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif 'target_critic2' in file and file.endswith('.h5'):
                    self.target_critic2.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
            if model_load_count == 5:
                sac_log.info('Models loaded successfully')
            else:
                sac_log.error('Models not loaded properly, please check model save directory')
        except:
            sac_log.error("Error while loading models, initializing new models...")

    def save(self, post_fix="test"):
        """ Save the ML models """
        try:
            destination_file_path = os.path.join(self.logdir, 'models/')
            destination_file_path = os.path.join(destination_file_path, post_fix + '/')
            os.makedirs(destination_file_path, exist_ok=True)

            self.actor_model.save_weights(
                join(destination_file_path, "actor_model_" + post_fix + ".h5"))
            self.critic_model1.save_weights(
                join(destination_file_path, "critic_model1_" + post_fix + ".h5"))
            self.target_critic1.save_weights(
                join(destination_file_path, "target_critic1_" + post_fix + ".h5"))
            self.critic_model2.save_weights(
                join(destination_file_path, "critic_model2_" + post_fix + ".h5"))
            self.target_critic2.save_weights(
                join(destination_file_path, "target_critic2_" + post_fix + ".h5"))
            sac_log.info('Agent models saved successfully')
        except:
            sac_log.error("Error in saving the models...")

    def save_cfg(self):
        """ Save the actor cfg """
        try:
            destination_file_path = os.path.join(self.logdir, 'cfgs/')
            os.makedirs(destination_file_path, exist_ok=True)
            destination_file_path = os.path.join(
                destination_file_path, os.path.basename(self.pfn_json_file))
            shutil.copy(self.pfn_json_file, destination_file_path)
            sac_log.info('Agent config saved successfully')
        except:
            sac_log.error("Error in saving the agent cfg...")