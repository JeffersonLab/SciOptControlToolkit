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
from jlab_rl.models.constraint_circle_generator import ConstraintCircleGenerator
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import os
from os.path import join
import time

class KerasTD3(jlab_rl.Agent):

    def __init__(self, env, warmup_size, nrff=0, logdir=None, model_load_path=None, model_save_path=None, **kwargs):
        """ Define all key variables required for all agent """

        # Get env info
        super().__init__(**kwargs)
        print('Running KerasTD3 __init__')
        self.env = env
        self.model_load_path = model_load_path
        self.model_save_path = model_save_path
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low
        print('upper_bound: ', self.upper_bound)
        print('lower_bound: ', self.lower_bound)
        self.action_width = (self.upper_bound+self.lower_bound)/2.0

        # Buffer
        self.min_buffer_counter = warmup_size
        self.buffer_counter = 0
        self.buffer_capacity = 5000000
        self.batch_size = 512 #1024
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.per_buffer = np.ones((self.buffer_capacity, 1))

        # Used to update target networks
        self.tau = 0.005
        self.gamma = 0.99

        # Setup Optimizers
        critic_lr = 3e-4
        actor_lr = 3e-4
        self.critic_optimizer1 = Adam(critic_lr, epsilon=1e-08)
        self.critic_optimizer2 = Adam(critic_lr, epsilon=1e-08)
        self.actor_optimizer = Adam(actor_lr, epsilon=1e-08)

        self.hidden_size = 256
        self.layer_std = 1.0 / np.sqrt(self.num_actions)

        self.initialize_new_models()
        # Load models for retraining
        if model_load_path is not None:
            self.load()

        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq=2
        self.critic_update_freq=2

        try:
            os.mkdir(logdir)
        except OSError as error:
            print(error)
        file_writer = tf.summary.create_file_writer(logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = tf.Variable(0)

    # @tf.function
    def train_critic(self, states, actions, rewards, next_states):
        next_rdm_gaus = np.array([np.random.normal(0, 1, self.num_actions + 2) for state in states])
        next_actions = self.target_actor(next_rdm_gaus, training=False)
        # next_actions = self.target_actor(next_states, training=False)
        # # Add a little noise
        # noise = np.random.normal(0, 0.2, self.num_actions)
        # noise = np.clip(noise, -0.5, 0.5)
        # next_actions = next_actions+noise
        new_q1 = self.target_critic1([next_states, next_actions], training=False)
        new_q2 = self.target_critic2([next_states, next_actions], training=False)
        new_q = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        q_targets = rewards + self.gamma * new_q
        # Critic 1
        with tf.GradientTape() as tape:
            q_values1 = self.critic_model1([states, actions], training=False)
            td_errors1 = q_values1-q_targets
            critic_loss1 = tf.reduce_mean(tf.math.square(td_errors1))
        gradient1 = tape.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(gradient1, self.critic_model1.trainable_variables))

        # Critic 2
        with tf.GradientTape() as tape:
            q_values2 = self.critic_model2([states, actions], training=False)
            td_errors2 = q_values2-q_targets
            critic_loss2 = tf.reduce_mean(tf.math.square(td_errors2))
        gradient2 = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(gradient2, self.critic_model2.trainable_variables))

    #@tf.function
    def train_actor(self, states):
        # Use Critic 1
        next_rdm_gaus = np.array([np.random.normal(0, 1, self.num_actions + 2) for state in states])
        with tf.GradientTape() as tape:
            actions = self.actor_model(next_rdm_gaus, training=True)
            q_value = self.critic_model1([states, actions], training=False)
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

    def get_critic(self):

        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states))
        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))
        state_action = tf.keras.layers.Concatenate()([state_input, action_input])
        state_action1 = tf.keras.layers.Dense(self.hidden_size, activation="relu")(state_action)
        state_action2 = tf.keras.layers.Dense(self.hidden_size, activation="relu")(state_action1)
        outputs = tf.keras.layers.Dense(1)(state_action2)
        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        #print('Critic model:', model.summary())
        return model

    def get_actor(self):

        model = ConstraintCircleGenerator(ndims=2, nlayers=5,
                                          lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return model

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        self.train_critic(state_batch, action_batch, reward_batch, next_state_batch)
        self.train_actor(state_batch)


    def train(self):
        """ Method used to train """
        self.ntrain_calls += 1

        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        #
        self.update(state_batch, action_batch, reward_batch, next_state_batch)
        if self.ntrain_calls%self.actor_update_freq == 0:
            self.soft_update(self.target_actor.variables, self.actor_model.variables)
        if self.ntrain_calls%self.critic_update_freq == 0:
            self.soft_update(self.target_critic1.variables, self.critic_model1.variables)
            self.soft_update(self.target_critic2.variables, self.critic_model2.variables)

    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """
        state = np.expand_dims(state, 0)

        rdm_norms = np.random.normal(0, 1, self.num_actions + 2)
        rdm_norms = np.expand_dims(rdm_norms, 0)
        sampled_action = self.actor_model(rdm_norms)
        noise = tf.zeros(sampled_action.shape)
        self.nactions.assign(self.nactions + 1)
        # if train==False:
        #     noise = tf.zeros(sampled_action.shape)
        #     legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        #     return [np.squeeze(legal_action)], [np.squeeze(noise)]
        #
        # self.nactions.assign(self.nactions + 1)
        # # TD3 version
        # if self.buffer_counter < self.min_buffer_counter:
        #     sampled_action = self.env.action_space.sample()
        #     noise = np.zeros(self.num_actions)
        # else:
        #     sampled_action = self.actor_model.predict_on_batch(state)
        #     noise = np.random.normal(0, 0.1, self.num_actions)

        sampled_action = np.squeeze(sampled_action)
        #print(sampled_action)
        for i in range(self.num_actions):
            if self.num_actions > 1:
                #print(sampled_action[i])
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))
        #q_pred = self.critic_model1([state, np.expand_dims(sampled_action, 0)])
        #tf.summary.scalar('Critic Prediction', data=np.squeeze(q_pred), step=int(self.nactions))
        # if train == True:
        #     sampled_action = sampled_action + noise

        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)], [np.squeeze(noise)]

    def memory(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

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

