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

import jlab_rl as jlab_rl
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
import numpy as np
import os
from os.path import join
import time
import random
import matplotlib.pyplot as plt

import platform
processor = platform.processor()
# if processor == 'arm':
#     import tensorflow.keras.optimizers.legacy.Adam as Adam
#     print('Using legacy Adam')
# else:
#from tensorflow.keras.optimizers import Adam
#import tf.keras.optimizers.legacy.Adam

import copy

class KerasTD3(jlab_rl.Agent):

    def __init__(self, env, warmup_size, nrff=0, logdir=None, model_load_path=None, model_save_path=None, **kwargs):
        """ Define all key variables required for all agent """

        # Get env info
        super().__init__(**kwargs)
        print('Running KerasTD3 __init__')
        self.logdir = logdir
        self.env = env
        self.model_load_path = model_load_path
        self.model_save_path = model_save_path
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low
        print('upper_bound: ',self.upper_bound)
        print('lower_bound: ',self.lower_bound)
        self.action_width = (self.upper_bound+self.lower_bound)/2.0

        # Buffer
        self.min_buffer_counter = warmup_size
        self.buffer_counter = 0
        self.buffer_capacity = 5000000
        self.batch_size = 1024
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.priority_buffer = np.ones((self.buffer_capacity, 1))
        self.batch_indices = None
        self.use_priority = 0

        # Used to update target networks
        self.tau = 0.01
        self.gamma = 0.99

        # Setup Optimizers
        critic_lr = 5e-3
        actor_lr = 1e-3

        if processor == 'arm':
            print('Using legacy Adam')
            self.critic_optimizer1 = tf.keras.optimizers.legacy.Adam(critic_lr, epsilon=1e-08)
            self.critic_optimizer2 = tf.keras.optimizers.legacy.Adam(critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(actor_lr, epsilon=1e-08)
        else:
            self.critic_optimizer1 = Adam(critic_lr, epsilon=1e-08)
            self.critic_optimizer2 = Adam(critic_lr, epsilon=1e-08)
            self.actor_optimizer = Adam(actor_lr, epsilon=1e-08)

        self.hidden_size = 256
        self.layer_std = 1.0 / np.sqrt(self.num_actions)
        self.ncritic_layers = 5

        self.initialize_new_models()
        # Load models for retraining
        if model_load_path is not None:
            self.load()

        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq=2
        self.critic_update_freq=2

        try:
            os.mkdir(self.logdir)
        except OSError as error:
            print(error)
        file_writer = tf.summary.create_file_writer(self.logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = tf.Variable(0)

    #@tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        next_actions = self.target_actor(next_states, training=False)
        # print('states:',states[0])
        # print('rewards:',rewards[0])
        # print('next_states:',next_states[0])
        # print('dones:',dones[0])
        # print('next_actions:',next_actions[0])
        # print('next_actions 1:',next_actions.shape)
        # Add a little noise
        noises = tf.random.normal(next_actions.shape, 0, 0.2)
        # print('noise:',noises[0])
        # print('noise 2:',noises.shape)
        #noise = np.random.normal(0, 0.2, self.num_actions)
        noises = np.clip(noises, -0.5, 0.5)
        next_actions = next_actions+noises
        # print('next_actions 2:',next_actions[0])
        new_q1 = self.target_critic1([next_states, next_actions], training=False)
        new_q2 = self.target_critic2([next_states, next_actions], training=False)
        new_q = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        q_targets = rewards + self.gamma * new_q * (1.0-dones)
        # Critic 1
        priority_buffer1 = None
        with tf.GradientTape() as tape:
            q_values1 = self.critic_model1([states, actions], training=False)
            td_errors1 = q_values1-q_targets
            #self.priority_buffer1 = tf.math.abs(td_errors1)
            priority_buffer1 = np.abs(td_errors1.numpy()+1e-8)
            critic_loss1 = tf.reduce_mean(tf.math.square(td_errors1))
        gradient1 = tape.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(gradient1, self.critic_model1.trainable_variables))

        # Critic 2
        priority_buffer2 = None
        with tf.GradientTape() as tape:
            q_values2 = self.critic_model2([states, actions], training=False)
            td_errors2 = q_values2-q_targets
            #self.priority_buffer2 = tf.math.abs(td_errors2)
            priority_buffer2 = np.abs(td_errors2.numpy()+1e-8)
            critic_loss2 = tf.reduce_mean(tf.math.square(td_errors2))
        gradient2 = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(gradient2, self.critic_model2.trainable_variables))

        self.priority_buffer[self.batch_indices] = (priority_buffer1+priority_buffer2)/2

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, training=True)
            q_value1 = self.critic_model1([states, actions], training=False)
            q_value2 = self.critic_model2([states, actions], training=False)
            q_value = tf.keras.layers.Average()([q_value1, q_value2])
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

    def get_critic(self):

        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states))
        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))
        state_action = tf.keras.layers.Concatenate()([state_input, action_input])
        for _ in range(self.ncritic_layers):
            state_action = tf.keras.layers.Dense(self.hidden_size, activation="relu")(state_action)
        #state_action2 = tf.keras.layers.Dense(self.hidden_size, activation="relu")(state_action1)
        outputs = tf.keras.layers.Dense(1)(state_action)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        #print('Critic model:',model.summary())
        return model

    def get_actor(self):

        inputs = tf.keras.layers.Input(shape=(self.num_states))
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

        # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # Input
        # inputs = tf.keras.layers.Input(shape=(self.num_states,))
        #
        # # Layer 1
        # out = tf.keras.layers.Dense(self.hidden_size,
        #                             kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
        #                             bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))(inputs)
        # out = tf.keras.layers.Activation(tf.nn.leaky_relu)(out)
        #
        # # Layer 2
        # out = tf.keras.layers.Dense(self.hidden_size,
        #                             kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
        #                             bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))(out)
        # out = tf.keras.layers.Activation(tf.nn.leaky_relu)(out)

        # Output
        # outputs = tf.keras.layers.Dense(self.num_actions, activation="tanh",
        #                                 kernel_initializer=last_init,
        #                                 use_bias=True)(out)

        # Rescale for tanh [-1,1]
        outputs = tf.keras.layers.Lambda(
            lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(outputs)

        model = tf.keras.Model(inputs, outputs)
        return model

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

        if self.buffer_counter>self.batch_size:
            # Get sampling range
            record_range = min(self.buffer_counter, self.buffer_capacity)

            #print('np.max(self.priority_buffer): ',np.max(self.priority_buffer))
            #self.priority_buffer = self.priority_buffer/np.max(self.priority_buffer)
            # print('record_range:{}\n'.format(range(record_range)))
            # print('range(len(self.priority_buffer):{}\n'.format(range(len(self.priority_buffer))))
            if self.use_priority == 1:
                # Normalize priority
                sum_priority_buffer = np.sum(self.priority_buffer)
                current_prob = self.priority_buffer / sum_priority_buffer
                current_weights = 1.0/current_prob
                max_weight = np.max(current_weights)
                current_is = (1.0/current_prob)/max_weight
                # Sample based on loss contribution
                self.batch_indices = random.choices(range(record_range),
                                                    k=self.batch_size,
                                                    weights=current_is[range(record_range)])
    #                                                weights=current_priority_buffer[range(record_range)])
                #self.priority_buffer[range(record_range)])
            else:
                # Randomly sample indices (priority = 0)
                self.batch_indices = np.random.choice(record_range, self.batch_size)

            # fig = plt.figure()
            if self.ntrain_calls%100==0:
                fig = plt.figure()
                plt.hist(self.priority_buffer[np.random.choice(record_range, self.batch_size)], bins=25, color='black',range=[0,1])
                plt.hist(self.priority_buffer[self.batch_indices], color='red', bins=25, range=[0,1])
                plt.savefig(self.logdir+'/priority_{}.png'.format(self.ntrain_calls))

            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[self.batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[self.batch_indices])
            reward_batch = tf.convert_to_tensor(self.reward_buffer[self.batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[self.batch_indices])
            done_batch = tf.convert_to_tensor(self.done_buffer[self.batch_indices])
            done_batch = tf.cast(done_batch, dtype=tf.float32)

            # print('### agent.train() ####')
            # print('state_batch', state_batch.shape)
            # print('action_batch', action_batch.shape)
            # print('reward_batch', reward_batch.shape)
            # print('next_state_batch', next_state_batch.shape)
            # print('done_batch', done_batch.shape)
            #
            self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            if self.ntrain_calls%self.actor_update_freq == 0:
                self.soft_update(self.target_actor.variables, self.actor_model.variables)
            if self.ntrain_calls%self.critic_update_freq == 0:
                self.soft_update(self.target_critic1.variables, self.critic_model1.variables)
                self.soft_update(self.target_critic2.variables, self.critic_model2.variables)

    #@tf.function
    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """
        state = tf.expand_dims(state, 0)

        # if train==False:
        #     sampled_action = self.actor_model.predict_on_batch(state)
        #     noise = tf.zeros(sampled_action.shape)
        #     legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        #     return [np.squeeze(legal_action)], [np.squeeze(noise)]

        self.nactions.assign(self.nactions + 1)
        # TD3 version
        if self.buffer_counter < self.min_buffer_counter:
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
            #noise = tf.zeros(sampled_action.shape)
        else:
            sampled_action = self.actor_model.predict_on_batch(state)
            #noise = tf.random.normal(sampled_action.shape, 0, 0.1)
            noise = np.random.normal(0, 0.1, self.num_actions)

        sampled_action = sampled_action.flatten()
        noise = noise.flatten()
        # print('sampled_action', sampled_action)
        # print('noise:', noise)
        #sampled_action = np.squeeze(sampled_action)
        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))
        #q_pred = self.critic_model1([state, np.expand_dims(sampled_action, 0)])
        #tf.summary.scalar('Critic Prediction', data=np.squeeze(q_pred), step=int(self.nactions))
        if train == True:
            sampled_action = sampled_action + noise
            #print('sampled_action w/ noise', sampled_action)

        return sampled_action, noise
        #legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        #return [np.squeeze(legal_action)], [np.squeeze(noise)]

    def memory(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
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

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        print('Running KerasTD3 initialize_new_models()')

        self.actor_model = self.get_actor()
        self.target_actor = self.get_actor()
        self.target_actor.set_weights(self.actor_model.get_weights())

        seed1 = time.time_ns()
        print('seed1:',seed1)
        tf.random.set_seed(seed1)
        self.critic_model1 = self.get_critic()
        self.target_critic1 = self.get_critic()
        self.target_critic1.set_weights(self.critic_model1.get_weights())

        seed2 = time.time_ns()
        print('seed2:',seed2)
        tf.random.set_seed(seed2)
        self.critic_model2 = self.get_critic()
        self.target_critic2 = self.get_critic()
        self.target_critic2.set_weights(self.critic_model2.get_weights())

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
            print("Error in saving the models...")

