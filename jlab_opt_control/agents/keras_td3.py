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
from tensorflow.keras import layers
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
td3_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        super().__init__()

        # Actor Architecture
        self.l1 = layers.Dense(256, activation="relu", input_shape=(state_dim,))
        self.l2 = layers.Dense(256, activation="relu")
        self.l3 = layers.Dense(action_dim, activation='tanh')

        self.action_scale = tf.constant((max_action - min_action) / 2, dtype=tf.float32)
        self.action_bias = tf.constant((max_action + min_action) / 2, dtype=tf.float32)

        self.max_action = max_action

    def call(self, state, training=False):
        a = self.l1(state)
        a = self.l2(a)
        a = self.l3(a)
        return a * self.action_scale + self.action_bias

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q network Architecture
        self.l1 = layers.Dense(256, activation="relu", input_shape=(state_dim + action_dim,))
        self.l2 = layers.Dense(256, activation="relu")
        self.l3 = layers.Dense(1)

    def call(self, state, action, training=False):
        x = tf.concat([state, action], axis=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x

class KerasTD3(jlab_opt_control.Agent):

    def __init__(self, env, logdir, buffer_type=None, buffer_size=None, cfg='keras_td3.json'):
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
            td3_log.info(f'Action upper bound: {float(env.action_space.high[0])}')
            td3_log.info(f'Action lower bound: {self.lower_bound}')
            td3_log.info(f'Action lower bound: {float(env.action_space.low[0])}')
            self.range = self.upper_bound - self.lower_bound
            td3_log.info(f'Action range: {self.range}')
        except:
            td3_log.error('Action space not valid for this agent.')
            sys.exit(0)



        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        pfn_json_file = os.path.join(full_path, cfg)
        td3_log.debug(f'pfn_json_file:{pfn_json_file}')
        with open(pfn_json_file) as json_file:
            data = json.load(json_file)
        self.buffer_counter = 0
        self.warmup_size = int(cfg_utils.cfg_get(data, 'warmup_size', 10000))
        self.min_buffer_counter = self.warmup_size
        self.buffer_capacity = int(cfg_utils.cfg_get(data, 'buffer_capacity', 1000000))
        self.batch_size = int(cfg_utils.cfg_get(data, 'batch_size', 100))

        self.model_load_path = cfg_utils.cfg_get(data, 'load_model', None)
        self.model_save_path = cfg_utils.cfg_get(data, 'save_model', None)

        self.logdir = logdir

        self.mse_loss = tf.keras.losses.MeanSquaredError()

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
        self.tau = float(cfg_utils.cfg_get(data, 'tau', 0.005))
        self.gamma = float(cfg_utils.cfg_get(data, 'discount', 0.99))

        # Setup Optimizers
        self.critic_lr = float(cfg_utils.cfg_get(data, 'critic_learning_rate', 5e-4))
        self.actor_lr = float(cfg_utils.cfg_get(data, 'actor_learning_rate', 1e-4))

        if processor == 'arm':
            td3_log.info('Using legacy Adam')
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(self.actor_lr, epsilon=1e-08)
        else:
            self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr, epsilon=1e-08)

        self.hidden_size = 400
        self.ncritic_layers = 2

        # self.initialize_new_models()
        self.actor_model = Actor(self.num_states, self.num_actions, self.lower_bound, self.upper_bound)
        self.target_actor = Actor(self.num_states, self.num_actions, self.lower_bound, self.upper_bound)

        self.critic_model1 = Critic(self.num_states, self.num_actions)
        self.target_critic1 = Critic(self.num_states, self.num_actions)

        self.critic_model2 = Critic(self.num_states, self.num_actions)
        self.target_critic2 = Critic(self.num_states, self.num_actions)

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic1.set_weights(self.critic_model1.get_weights())
        self.target_critic2.set_weights(self.critic_model2.get_weights())

        # Load models for retraining
        if self.model_load_path is not None:
            self.load()

        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq = int(cfg_utils.cfg_get(data, 'actor_update_freq', 2))
        self.critic_update_freq = int(cfg_utils.cfg_get(data, 'critic_update_freq', 2))

        self.noise_clip = 0.2

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
            state_action = tf.keras.layers.Dense(self.hidden_size, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(state_action)
        outputs = tf.keras.layers.Dense(1, activation="linear")(state_action)
        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def get_actor(self):
        inputs = tf.keras.layers.Input(shape=self.num_states)
        #
        out = tf.keras.layers.Dense(self.hidden_size, kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
        out = tf.keras.layers.Activation(tf.nn.relu)(out)
        #
        out = tf.keras.layers.Dense(self.hidden_size, kernel_initializer=tf.keras.initializers.HeNormal())(out)
        out = tf.keras.layers.Activation(tf.nn.relu)(out)
        #
        out = tf.keras.layers.Dense(self.num_actions, kernel_initializer=tf.keras.initializers.HeNormal())(out)
        out = tf.keras.layers.Activation(tf.nn.tanh)(out)
        #
        outputs = self.upper_bound * out

        # Rescale for tanh [-1,1]
        # outputs = tf.keras.layers.Lambda(
        #     lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(outputs)

        model = tf.keras.Model(inputs, outputs)
        return model

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        td3_log.info('Running KerasTD3 initialize_new_models()')

        self.actor_model = self.get_actor()
        self.target_actor = self.get_actor()
        self.target_actor.set_weights(self.actor_model.get_weights())

        seed1 = time.time_ns()
        str_seed1 = str(seed1)
        seed1 = int(str_seed1[9:-3])
        td3_log.debug(f'seed1:{seed1}')
        tf.random.set_seed(seed1)
        self.critic_model1 = self.get_critic()
        self.target_critic1 = self.get_critic()
        self.target_critic1.set_weights(self.critic_model1.get_weights())
        time.sleep(1 / 10)
        seed2 = time.time_ns()
        str_seed2 = str(seed2)
        seed2 = int(str_seed2[9:-3])
        td3_log.debug(f'seed2:{seed2}')
        tf.random.set_seed(seed2)
        self.critic_model2 = self.get_critic()
        self.target_critic2 = self.get_critic()
        self.target_critic2.set_weights(self.critic_model2.get_weights())

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        # Generate the proper noise
        noise = (tf.random.normal(tf.shape(actions), dtype=tf.float32) * 0.2)
        noise_clipped = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip) * self.target_actor.action_scale
        next_actions = tf.clip_by_value(self.target_actor(next_states, training=False) + noise_clipped, self.lower_bound, self.upper_bound)

        target_q1 = self.target_critic1(next_states, next_actions, training=False)
        target_q2 = self.target_critic2(next_states, next_actions, training=False)
        target_q = tf.math.minimum(target_q1, target_q2)

        # Bellman equation for the q value
        q_targets = rewards + self.gamma * target_q * (1.0 - dones)

        # Critic 1 and 2
        with tf.GradientTape() as tape:
            q_values1 = self.critic_model1(states, actions, training=True)
            q_values2 = self.critic_model2(states, actions, training=True)
            td_errors1 = q_values1 - q_targets
            td_errors2 = q_values2 - q_targets
            critic_loss1 = self.mse_loss(q_values1, q_targets)
            critic_loss2 = self.mse_loss(q_values2, q_targets)
            critic_losses =  critic_loss1 + critic_loss2
        gradients = tape.gradient(critic_losses, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables))

        return critic_loss1, critic_loss2

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, training=True)
            q_value = self.critic_model1(states, actions, training=False)
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
        return loss

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def train(self):
        """ Method used to train """
        if self.buffer_counter > np.max([self.batch_size, self.warmup_size]):
            self.ntrain_calls += 1
            # Get sampling range
            record_range = min(self.buffer_counter, self.buffer_capacity)
            batch_indices = np.random.choice(record_range, self.batch_size)
            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
            action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
            reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
            done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])
            done_batch = tf.cast(done_batch, dtype=tf.float32)

            # Train critic
            critic_loss1, critic_loss2 = self.train_critic(state_batch, action_batch, reward_batch,
                                                           next_state_batch,done_batch)
            tf.summary.scalar('Critic Loss 1', data=critic_loss1, step=int(self.ntrain_calls))
            tf.summary.scalar('Critic Loss 2', data=critic_loss2, step=int(self.ntrain_calls))

            if self.ntrain_calls % self.actor_update_freq == 0:
                actor_loss = self.train_actor(state_batch)
                tf.summary.scalar('Actor Loss', data=actor_loss, step=int(self.ntrain_calls))
                self.soft_update(self.target_actor.variables, self.actor_model.variables)

            if self.ntrain_calls % self.critic_update_freq == 0:
                self.soft_update(self.target_critic1.variables, self.critic_model1.variables)
                self.soft_update(self.target_critic2.variables, self.critic_model2.variables)

    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """            
        # Warmup experience sample
        if self.buffer_counter < np.max([self.batch_size, self.warmup_size]):
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
        # Warmup completed, sample from actor
        else:
            state = tf.expand_dims(state, 0)
            sampled_action = (self.actor_model(state)).numpy()
            if train:
                noise = (tf.random.normal(shape=(self.num_actions,), mean=0, stddev = self.actor_model.action_scale * 0.1, dtype=tf.float32)).numpy()
                sampled_action = np.clip(sampled_action + noise, self.lower_bound, self.upper_bound)
            else:
                noise = np.zeros(self.num_actions)

            sampled_action = sampled_action.flatten()
            noise = noise.flatten()
            assert sampled_action.shape == self.num_actions or sampled_action.shape == (self.num_actions,), \
                f"Sampled action shape is incorrect... {sampled_action.shape}"

        # Log the training action(s) taken
        if train:
            self.nactions = self.nactions + 1
            if self.num_actions == 0:
                tf.summary.scalar('Action', data=sampled_action, step=int(self.nactions))
            else:
                for i in range(self.num_actions):
                    tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))

        # Insure action output by actor is in legal environment range
        return sampled_action, noise

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
