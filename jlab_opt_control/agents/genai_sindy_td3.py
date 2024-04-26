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
from jlab_opt_control.utils.sindy_utils import PolynomialLibrary

import tensorflow as tf
import numpy as np
import os
from os.path import join
import time
import json
import platform
import sys
import shutil
processor = platform.processor()

td3_log = logging.getLogger("TD3-Agent")
td3_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class GenAISINDyTD3(jlab_opt_control.Agent):

    def __init__(self, env, logdir, buffer_type=None, buffer_size=None, cfg='keras_td3.json'):
        """ Define all key variables required for all agent """

        # Get env info
        self.target_critic2 = None
        self.critic_model2 = None
        self.target_critic1 = None
        self.critic_model1 = None
        self.target_actor = None
        self.actor_model = None
        td3_log.info('Running KerasTD3 __init__')

        # Environment setup
        self.env = env
        try:
            assert "Box" in str(type(env.action_space)), 'Invalid action space'
            self.num_states = env.observation_space.shape[0]
            self.num_actions = env.action_space.shape[0]
            self.upper_bound = env.action_space.high
            self.lower_bound = env.action_space.low
            td3_log.info(f'Action upper bound: {self.upper_bound}')
            td3_log.info(
                f'Action upper bound: {float(env.action_space.high[0])}')
            td3_log.info(f'Action lower bound: {self.lower_bound}')
            td3_log.info(
                f'Action lower bound: {float(env.action_space.low[0])}')
            self.range = self.upper_bound - self.lower_bound
            td3_log.info(f'Action range: {self.range}')
        except:
            td3_log.error('Action space not valid for this agent.')
            sys.exit(0)

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)
        td3_log.debug(f'pfn_json_file:{self.pfn_json_file}')
        with open(self.pfn_json_file) as json_file:
            data = json.load(json_file)
        self.warmup_size = int(cfg_utils.cfg_get(data, 'warmup_size', 10000))
        self.batch_size = int(cfg_utils.cfg_get(data, 'batch_size', 100))
        self.model_load_path = cfg_utils.cfg_get(data, 'load_model', None)

        self.actor_model_type = cfg_utils.cfg_get(
            data, 'actor_model', "actor_fcnn-v0")
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

        # Setup Optimizers
        self.critic_lr = float(cfg_utils.cfg_get(
            data, 'critic_learning_rate', 5e-4))
        self.actor_lr = float(cfg_utils.cfg_get(
            data, 'actor_learning_rate', 1e-4))

        if processor == 'arm':
            td3_log.info('Using legacy Adam')
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(
                self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(
                self.actor_lr, epsilon=1e-08)
        else:
            self.critic_optimizer = tf.keras.optimizers.Adam(
                self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.Adam(
                self.actor_lr, epsilon=1e-08)

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
            td3_log.warning(error)
        file_writer = tf.summary.create_file_writer(self.logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = 0

        # SINDy library
        num_poly = 3
        self.library = PolynomialLibrary(degree=num_poly, include_bias=False)
        rng = np.random.default_rng(1)
        init_states = tf.convert_to_tensor(rng.normal(loc=0., scale=1., size=[self.batch_size, self.num_states]),
                                           dtype=tf.float32)
        #print(f'init_states: {init_states.shape}')
        self.library.fit(init_states)
        lib_batch = self.library(init_states)
        td3_log.debug(f'lib shape:{lib_batch.shape}')

        # SINDy critic
        # self.sindy_critic_model = jlab_opt_control.models.make('uqsindy_network-v0',
        #                                                        num_features_in=self.library.output_dim_,
        #                                                        num_features_out=1,
        #                                                        min_action = env.action_space.low
        #                                                        max_action = env.action_space.high
        #                                                        # assuming a scalar reward
        #                                                        batch_size=self.batch_size,
        #                                                        logdir=self.logdir + '/sindy_test/')
        # self.sindy_critic_optimizer = tf.keras.optimizers.legacy.Adam(self.critic_lr, epsilon=1e-08)
        # self.sindy_critic_model(lib_batch)

        # SINDy Actor
        # self.sindy_actor_model = jlab_opt_control.models.make('sindy_network-v0',
        #     num_features_in=self.library.output_dim_, num_features_out=self.num_actions, logdir=self.logdir+'/sindy_test/')
        # self.sindy_actor_optimizer = tf.keras.optimizers.legacy.Adam(self.actor_lr, epsilon=1e-08)
        # self.sindy_actor_model(lib_batch)
        self.sindy_actor_model = jlab_opt_control.models.make('uqsindy_network-v0',
                                                              num_features_in=self.library.output_dim_,
                                                              num_features_out=self.num_actions,
                                                              min_action=env.action_space.low,
                                                              max_action = env.action_space.high,
                                                              batch_size=self.batch_size,
                                                              logdir=self.logdir+'/sindy_test/')
        self.sindy_actor_optimizer = tf.keras.optimizers.legacy.Adam(self.actor_lr, epsilon=1e-08)
        self.sindy_actor_model(lib_batch)

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        td3_log.info('Running KerasTD3 initialize_new_models()')

        self.actor_model = jlab_opt_control.models.make(
            self.actor_model_type, state_dim=self.num_states, action_dim=self.num_actions, min_action=self.lower_bound, max_action=self.upper_bound, logdir=self.logdir)
        self.target_actor = jlab_opt_control.models.make(
            self.actor_model_type, state_dim=self.num_states, action_dim=self.num_actions, min_action=self.lower_bound, max_action=self.upper_bound, logdir=self.logdir)

        # Run through model once to initialize variables
        self.actor_model(tf.zeros([1, self.num_states]))
        self.target_actor(tf.zeros([1, self.num_states]))

        self.actor_model.save_cfg()

        seed1 = time.time_ns()
        str_seed1 = str(seed1)
        seed1 = int(str_seed1[9:-3])
        td3_log.debug(f'seed1:{seed1}')
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
        td3_log.debug(f'seed2:{seed2}')
        tf.random.set_seed(seed2)

        self.critic_model2 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, logdir=self.logdir)
        self.target_critic2 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, logdir=self.logdir)

        # Run through model once to initialize variables
        self.critic_model2(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_actions]))
        self.target_critic2(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_actions]))

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic1.set_weights(self.critic_model1.get_weights())
        self.target_critic2.set_weights(self.critic_model2.get_weights())

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones, weights):
        # Generate the proper noise
        noise = (tf.random.normal(tf.shape(actions), dtype=tf.float32) * 0.2)
        noise_clipped = tf.clip_by_value(
            noise, -self.noise_clip, self.noise_clip) * self.target_actor.action_scale
        next_actions = tf.clip_by_value(self.target_actor(
            next_states, training=False) + noise_clipped, self.lower_bound, self.upper_bound)

        target_q1 = self.target_critic1(
            next_states, next_actions, training=False)
        target_q2 = self.target_critic2(
            next_states, next_actions, training=False)
        target_q = tf.math.minimum(target_q1, target_q2)

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

    # @tf.function
    # def train_sindy_critic(self, states, actions):
    #     lib_batch = self.library(states)
    #     with tf.GradientTape() as tape:
    #         q_values = self.critic_model1(states, actions)
    #         neg_log_p_Xy = self.sindy_critic_model.negative_log_likelihood(lib_batch, q_values)
    #         kld = self.sindy_critic_model.kld()
    #         loss = tf.reduce_mean(neg_log_p_Xy) + kld
    #     gradients = tape.gradient(loss, self.sindy_critic_model.trainable_variables)
    #     self.sindy_critic_optimizer.apply_gradients(zip(gradients, self.sindy_critic_model.trainable_variables))
    #     sindy_q_values = self.sindy_critic_model(lib_batch)
    #     rel_mean_diff = tf.abs( tf.reduce_mean(sindy_q_values)-tf.reduce_mean(q_values) )/ tf.abs(tf.reduce_mean(q_values))
    #     return loss, rel_mean_diff

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, training=True)
            q_value = self.critic_model1(states, actions, training=False)
            td3_loss = -tf.math.reduce_mean(q_value)

            # Include SINdy loss from action distance
            lib_batch = self.library(states)
            sindy_actions = self.sindy_actor_model(lib_batch)
            #print(f'sindy_action min/max: {tf.math.reduce_max(sindy_actions)} / {tf.math.reduce_min(sindy_actions)}')
            # print(f'actions: {actions.shape}')
            # print(f'sindy_actions: {sindy_actions.shape}')
            actions_dist = 0
            sindy_actions_dist = 0
            for a in range(self.num_actions):
                sub_actions = tf.reshape(actions[:, a], [-1])
                sub_actions_dist = tf.math.squared_difference(
                    tf.expand_dims(sub_actions, axis=1), tf.expand_dims(sub_actions, axis=0))
                actions_dist += sub_actions_dist
                #
                sub_sindy_actions = tf.reshape(sindy_actions[:, a], [-1])
                sub_sindy_actions_dist = tf.math.squared_difference(
                    tf.expand_dims(sub_sindy_actions, axis=1), tf.expand_dims(sub_sindy_actions, axis=0))
                sindy_actions_dist += sub_sindy_actions_dist

            actions_dist = tf.reshape(actions_dist,shape=(-1,1))
            sindy_actions_dist = tf.reshape(sindy_actions_dist,shape=(-1,1))
            #print(f'actions_dist: {actions_dist.shape}')
            #print(f'sindy_actions_dist: {sindy_actions_dist.shape}')
            actions_dist = tf.sort(actions_dist)
            sindy_actions_dist = tf.sort(sindy_actions_dist)
            # Simple linear distance loss term for now
            extra_loss = tf.reduce_mean(tf.abs(actions_dist - sindy_actions_dist))
            loss = td3_loss + extra_loss

        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
        return loss, td3_loss, extra_loss

    def train_sindy_actor(self, states):
        # Use Critic 1
        lib_batch = self.library(states)
        with tf.GradientTape() as tape:
            actions = self.sindy_actor_model(lib_batch)
            print(f'actions: {actions.shape}')
            q_value = self.critic_model1(states, actions, training=False)
            print(f'q_value: {q_value.shape}')
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.sindy_actor_model.trainable_variables)
        self.sindy_actor_optimizer.apply_gradients(zip(gradient, self.sindy_actor_model.trainable_variables))
        return loss

    @tf.function
    def train_uq_sindy_actor(self, states):
        # Use Critic 1
        nsamples = 25
        #print(f'train states: {states.shape}')
        repeated_states = tf.repeat(states, nsamples, axis=0)
        #print(f'train repeated_states: {repeated_states.shape}')
        lib_batch = self.library(repeated_states)
        #print(f'train lib_batch: {lib_batch.shape}')

        with tf.GradientTape() as tape:
            actions = self.sindy_actor_model(lib_batch, nsamples=1)
            #actions = tf.reshape(actions, shape=(-1, self.num_actions))
            q_value = self.critic_model1(repeated_states, actions, training=False)
            loss = -tf.math.reduce_mean(q_value) #+ self.sindy_actor_model.kld()
        gradient = tape.gradient(loss, self.sindy_actor_model.trainable_variables)
        self.sindy_actor_optimizer.apply_gradients(zip(gradient, self.sindy_actor_model.trainable_variables))
        return loss


    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau +
                                 target_weight * (1.0 - self.tau))

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
            # # SINDy Critic
            # sindy_loss, sindy_rel_loss_diff = self.train_sindy_critic(state_batch, action_batch)
            # tf.summary.scalar('Critic SINDy Loss', data=sindy_loss,
            #                   step=int(self.ntrain_calls))
            # tf.summary.scalar('Critic SINDy Rel Loss Diff', data=sindy_rel_loss_diff,
            #                   step=int(self.ntrain_calls))

            # SINDy Actor
            sindy_actor_loss = self.train_uq_sindy_actor(state_batch)
            tf.summary.scalar('Actor SINDy Loss', data=sindy_actor_loss, step=int(self.ntrain_calls))

            # Update Priorities
            if "PER" in self.buffer_type:
                new_priorities = td_errors.numpy()
                self.buffer.update_priorities(new_priorities)

            if self.ntrain_calls % self.actor_update_freq == 0:
                actor_loss, td3_loss, extra_loss = self.train_actor(state_batch)
                tf.summary.scalar('Actor Loss', data=actor_loss,
                                  step=int(self.ntrain_calls))
                tf.summary.scalar('Actor TD3 Loss', data=td3_loss,
                                  step=int(self.ntrain_calls))
                tf.summary.scalar('Actor SINDy Loss', data=extra_loss,
                                  step=int(self.ntrain_calls))
                self.soft_update(self.target_actor.variables,
                                 self.actor_model.variables)

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
            sindy_action_mu = np.zeros(self.num_actions)
            sindy_action_std = np.zeros(self.num_actions)
        # Warmup completed, sample from actor or run inference
        else:
            state = tf.expand_dims(state, 0)
            sampled_action = (self.actor_model(state)).numpy()
            lib = self.library(tf.repeat(state,1,axis=0))
            #print(f'lib: {lib.shape}')
            sindy_actions = self.sindy_actor_model(lib, nsamples=1)
            #print(f'sindy_actions: {sindy_actions.shape}')
            #sindy_action_mu = np.mean(sindy_actions)
            #sindy_action_std = np.std(sindy_actions)
            # print(f'sindy_actions: {sindy_actions.shape}')
            # print(f'sampled_action: {sampled_action.shape}')
            # print(f'sindy_action_mu: {sindy_action_mu.shape}')

            if train:
                noise = (tf.random.normal(shape=(self.num_actions,), mean=0,
                         stddev=self.actor_model.action_scale * 0.1, dtype=tf.float32)).numpy()
                noise = np.zeros(self.num_actions)
                sampled_action = np.clip(
                    sampled_action + noise, self.lower_bound, self.upper_bound)
            else:
                noise = np.zeros(self.num_actions)

            sampled_action = sampled_action.flatten()
            noise = noise.flatten()
            assert sampled_action.shape == self.num_actions or sampled_action.shape == (self.num_actions,), \
                f"Sampled action shape is incorrect... {sampled_action.shape}"

            # sindy_action_mu = sindy_action.flatten()
            # sindy_action_std = sindy_action_std.flatten()
            # noise = (tf.random.normal(shape=(self.num_actions,), mean=0,
            #          stddev=sindy_action_std, dtype=tf.float32)).numpy()
            sampled_action = np.reshape(sindy_actions,-1) #np.clip( sindy_action_mu + noise, self.lower_bound, self.upper_bound)
        #print(f'sampled_action: {sampled_action.shape}')


        # Log the training action(s) taken
        if train:
            self.nactions = self.nactions + 1
            if self.num_actions == 0:
                tf.summary.scalar('Action', data=sampled_action,
                                  step=int(self.nactions))
                # tf.summary.scalar('SINDy Action Mean', data=sindy_action_mu,
                #                       step=int(self.nactions))
                # tf.summary.scalar('SINDy Action STD', data=sindy_action_std,
                #                       step=int(self.nactions))
            else:
                for i in range(self.num_actions):
                    tf.summary.scalar('Action #{}'.format(
                        i), data=sampled_action[i], step=int(self.nactions))
                    # tf.summary.scalar('SINDy Action Mean', data=sindy_action_mu[i],
                    #                   step=int(self.nactions))
                    # tf.summary.scalar('SINDy Action STD', data=sindy_action_std[i],
                    #                   step=int(self.nactions))
\
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
                elif 'target_actor' in file and file.endswith('.h5'):
                    self.target_actor.load_weights(join(self.model_load_path, file))
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
            if model_load_count == 6:
                td3_log.info('Models loaded successfully')
            else:
                td3_log.error('Models not loaded properly, please check model save directory')
        except:
            td3_log.error("Error while loading models, initializing new models...")

    def save(self, post_fix="test"):
        """ Save the ML models """
        try:
            destination_file_path = os.path.join(self.logdir, 'models/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)

            destination_file_path = os.path.join(destination_file_path, post_fix + '/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)

            self.actor_model.save_weights(
                join(destination_file_path, "actor_model_" + post_fix + ".h5"))
            self.target_actor.save_weights(
                join(destination_file_path, "target_actor_" + post_fix + ".h5"))
            self.critic_model1.save_weights(
                join(destination_file_path, "critic_model1_" + post_fix + ".h5"))
            self.target_critic1.save_weights(
                join(destination_file_path, "target_critic1_" + post_fix + ".h5"))
            self.critic_model2.save_weights(
                join(destination_file_path, "critic_model2_" + post_fix + ".h5"))
            self.target_critic2.save_weights(
                join(destination_file_path, "target_critic2_" + post_fix + ".h5"))
            td3_log.info('Agent models saved successfully')
        except:
            td3_log.error("Error in saving the models...")

    def save_cfg(self):
        """ Save the actor cfg """
        try:
            destination_file_path = os.path.join(self.logdir, 'cfgs/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)
            destination_file_path = os.path.join(
                destination_file_path, os.path.basename(self.pfn_json_file))
            shutil.copy(self.pfn_json_file, destination_file_path)
            td3_log.info('Agent config saved successfully')
        except:
            td3_log.error("Error in saving the agent cfg...")
