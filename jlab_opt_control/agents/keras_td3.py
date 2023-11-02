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

import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
import tensorflow as tf
import numpy as np
import os
from os.path import join
import time
import json
import platform
import sys
processor = platform.processor()

import logging

td3_log = logging.getLogger("TD3-Agent")
td3_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class KerasTD3(jlab_opt_control.Agent):

    def __init__(self, env, logdir, cfg='keras_td3.json'):
        """ Define all key variables required for all agent """

        # Get env info
        # super().__init__(**kwargs)
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
            td3_log.info(f'Action lower bound: {self.lower_bound}')
        except:
            td3_log.error('Action space not valid for this agent.')
            sys.exit(0)



        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        pfn_json_file = os.path.join(full_path, cfg)
        td3_log.debug('pfn_json_file:', pfn_json_file)
        with open(pfn_json_file) as json_file:
            data = json.load(json_file)
        self.buffer_counter = 0
        self.warmup_size = int(cfg_utils.cfg_get(data, 'warmup_size', 1000))
        self.min_buffer_counter = self.warmup_size
        self.buffer_capacity = int(cfg_utils.cfg_get(data, 'buffer_capacity', 5000000))
        self.batch_size = int(cfg_utils.cfg_get(data, 'batch_size', 1000))

        self.model_load_path = cfg_utils.cfg_get(data, 'load_model', None)
        self.model_save_path = cfg_utils.cfg_get(data, 'save_model', None)

        self.logdir = logdir

        # Buffer

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.priority_buffer = np.ones((self.buffer_capacity, 1))
        self.action_type_buffer = np.ones((self.buffer_capacity, 1))
        self.batch_indices = None
        self.use_priority = 0

        # Used to update target networks
        self.tau = 0.01
        self.gamma = 0.99

        # Setup Optimizers
        self.critic_lr = 5e-3
        self.actor_lr = 1e-3

        if processor == 'arm':
            td3_log.info('Using legacy Adam')
            self.critic_optimizer1 = tf.keras.optimizers.legacy.Adam(self.critic_lr, epsilon=1e-08)
            self.critic_optimizer2 = tf.keras.optimizers.legacy.Adam(self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(self.actor_lr, epsilon=1e-08)
        else:
            self.critic_optimizer1 = tf.keras.optimizers.Adam(self.critic_lr, epsilon=1e-08)
            self.critic_optimizer2 = tf.keras.optimizers.Adam(self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr, epsilon=1e-08)

        self.hidden_size = 256
        self.layer_std = 1.0 / np.sqrt(self.num_actions)
        self.ncritic_layers = 4

        self.initialize_new_models()
        # Load models for retraining
        if self.model_load_path is not None:
            self.load()

        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq = 2
        self.critic_update_freq = 2

        try:
            os.mkdir(self.logdir)
        except OSError as error:
            td3_log.warning(error)
        file_writer = tf.summary.create_file_writer(self.logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = 0

    def get_critic(self):
        # State as input
        state_input = tf.keras.layers.Input(shape=self.num_states)
        # Action as input
        action_input = tf.keras.layers.Input(shape=self.num_actions)
        state_action = tf.keras.layers.Concatenate()([state_input, action_input])
        for _ in range(self.ncritic_layers):
            state_action = tf.keras.layers.Dense(self.hidden_size, activation="relu")(state_action)
        outputs = tf.keras.layers.Dense(1)(state_action)
        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def get_actor(self):

        inputs = tf.keras.layers.Input(shape=self.num_states)
        #
        out = tf.keras.layers.Dense(self.hidden_size)(inputs)
        out = tf.keras.layers.Activation(tf.nn.relu)(out)
        #
        out = tf.keras.layers.Dense(self.hidden_size)(out)
        out = tf.keras.layers.Activation(tf.nn.relu)(out)
        #
        out = tf.keras.layers.Dense(self.num_actions)(out)
        out = tf.keras.layers.Activation(tf.nn.tanh)(out)
        #
        outputs = out
        # Rescale for tanh [-1,1]
        outputs = tf.keras.layers.Lambda(
            lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(outputs)

        model = tf.keras.Model(inputs, outputs)
        return model

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        td3_log.info('Running KerasTD3 initialize_new_models()')

        self.actor_model = self.get_actor()
        self.target_actor = self.get_actor()
        self.target_actor.set_weights(self.actor_model.get_weights())

        seed1 = time.time_ns()

        td3_log.debug('seed1:', seed1)
        tf.random.set_seed(seed1)
        self.critic_model1 = self.get_critic()
        self.target_critic1 = self.get_critic()
        self.target_critic1.set_weights(self.critic_model1.get_weights())
        time.sleep(1 / 10)
        seed2 = time.time_ns()
        td3_log.debug('seed2:', seed2)
        tf.random.set_seed(seed2)
        self.critic_model2 = self.get_critic()
        self.target_critic2 = self.get_critic()
        self.target_critic2.set_weights(self.critic_model2.get_weights())

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        next_actions = self.target_actor(next_states, training=False)
        noises = tf.random.normal(next_actions.shape, 0, 0.2)
        noises = tf.clip_by_value(noises, -0.5, 0.5)
        next_actions = next_actions + noises
        new_q1 = self.target_critic1([next_states, next_actions], training=False)
        new_q2 = self.target_critic2([next_states, next_actions], training=False)
        new_q = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        q_targets = rewards + self.gamma * new_q * (1.0 - dones)
        # Critic 1
        with tf.GradientTape() as tape:
            q_values1 = self.critic_model1([states, actions], training=False)
            td_errors1 = q_values1 - q_targets
            critic_loss1 = tf.reduce_mean(tf.math.square(td_errors1))
        gradient1 = tape.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(gradient1, self.critic_model1.trainable_variables))

        # Critic 2
        with tf.GradientTape() as tape:
            q_values2 = self.critic_model2([states, actions], training=False)
            td_errors2 = q_values2 - q_targets
            critic_loss2 = tf.reduce_mean(tf.math.square(td_errors2))
        gradient2 = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(gradient2, self.critic_model2.trainable_variables))

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, training=True)
            q_value = self.critic_model1([states, actions], training=False)
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.train_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.train_actor(state_batch)

    def train(self):
        """ Method used to train """
        self.ntrain_calls += 1

        if self.buffer_counter >= self.batch_size:
            # Get sampling range
            record_range = min(self.buffer_counter, self.buffer_capacity)
            self.batch_indices = np.random.choice(record_range, self.batch_size)

            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[self.batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[self.batch_indices])
            reward_batch = tf.convert_to_tensor(self.reward_buffer[self.batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[self.batch_indices])
            done_batch = tf.convert_to_tensor(self.done_buffer[self.batch_indices])
            done_batch = tf.cast(done_batch, dtype=tf.float32)

            self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            if self.ntrain_calls % self.actor_update_freq == 0:
                self.soft_update(self.target_actor.variables, self.actor_model.variables)
            if self.ntrain_calls % self.critic_update_freq == 0:
                self.soft_update(self.target_critic1.variables, self.critic_model1.variables)
                self.soft_update(self.target_critic2.variables, self.critic_model2.variables)

    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """
        self.nactions = self.nactions + 1

        if self.buffer_counter < np.max([self.batch_size, self.min_buffer_counter]):
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
        else:
            state = tf.expand_dims(state, 0)

            # else:
            sampled_action = (self.actor_model(state)).numpy()
            if train:
                noise = (tf.random.normal(sampled_action.shape, 0, 0.1)).numpy()
                sampled_action = sampled_action + noise
            else:
                noise = np.zeros(self.num_actions)

            sampled_action = sampled_action.flatten()
            noise = noise.flatten()
            assert sampled_action.shape == self.num_actions or sampled_action.shape == (self.num_actions,), \
                f"Sampled action shape is incorrect... {sampled_action.shape}"

        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))

        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return legal_action, noise

    def memory(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1

    def load(self):
        """ Load the ML models """
        try:
            self.actor_model.load_weights(join(self.model_load_path, "actor_model.h5"))
            self.target_actor.load_weights(join(self.model_load_path, "target_actor.h5"))
            self.critic_model1.load_weights(join(self.model_load_path, "critic_model1.h5"))
            self.target_critic1.load_weights(join(self.model_load_path, "target_critic1.h5"))
            self.critic_model2.load_weights(join(self.model_load_path, "critic_model2.h5"))
            self.target_critic2.load_weights(join(self.model_load_path, "target_critic2.h5"))
        except:
            print("Error while loading models, initializing new models...")

    def save(self):
        """ Save the ML models """
        try:
            self.actor_model.save_weights(join(self.model_save_path, "actor_model.h5"))
            self.target_actor.save_weights(join(self.model_save_path, "target_actor.h5"))
            self.critic_model1.save_weights(join(self.model_save_path, "critic_model1.h5"))
            self.target_critic1.save_weights(join(self.model_save_path, "target_critic1.h5"))
            self.critic_model2.save_weights(join(self.model_save_path, "critic_model2.h5"))
            self.target_critic2.save_weights(join(self.model_save_path, "target_critic2.h5"))
        except:
            td3_log.error("Error in saving the models...")
