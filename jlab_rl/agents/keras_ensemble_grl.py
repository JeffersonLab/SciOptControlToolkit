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
from jlab_rl.models.state_generator import Generator
from jlab_rl.agents.keras_td3 import KerasTD3

#from tensorflow.keras.initializers import RandomUniform
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import os
from os.path import join
import time
import random

from tensorflow.keras.optimizers.legacy import Adam

class KerasEnsembleGenerativeTD3(KerasTD3):

    def __init__(self, env, warmup_size, nrff=0, logdir=None, model_load_path=None, model_save_path=None, **kwargs):
        """ Define all key variables required for all agent """

        # Get env info
        super().__init__(env, warmup_size, nrff, logdir, model_load_path, model_save_path, **kwargs)
        print('Running KerasGenerativeDynamicModelBased __init__')

        self.nrdm_inputs = 100
        self.nactors = 7
        self.actor_models = []
        self.target_actors = []
        self.actor_optimizers = []
        self.batch_size = 1000

        # Setup Optimizers
        # if processor == 'arm':
        #     print('Using legacy Adam')
        #     self.actor_optimizers = [tf.keras.optimizers.legacy.Adam(self.actor_lr, epsilon=1e-08) for _ in range(self.nactors)]
        # else:
        self.actor_optimizers = [Adam(self.actor_lr, epsilon=1e-08) for _ in range(self.nactors)]

        self.actor_models = [self.get_actor() for _ in range(self.nactors)]
        self.target_actors = [self.get_actor() for _ in range(self.nactors)]
        for i in range(self.nactors):
            self.target_actors[i].set_weights(self.actor_models[i].get_weights())
        print('Actor summary:', self.actor_models[0].summary())

    def get_actor(self):
        seed = time.time_ns()
        tf.random.set_seed(seed)
        model = Generator(ndims=self.num_actions, nlayers=5, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return model

    #@tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):

        # Take the average
        q_targets = 0
        for i in range(self.nactors):
            next_rdm_gaus = tf.random.normal([next_states.shape[0], self.nrdm_inputs], 0, 1, tf.float32, seed=1)
            next_actions = self.target_actors[i]([next_states, next_rdm_gaus], training=False)
            new_q1 = self.target_critic1([next_states, next_actions], training=False)
            new_q2 = self.target_critic2([next_states, next_actions], training=False)
            new_q = tf.math.minimum(new_q1, new_q2)
            # Bellman equation for the q value
            this_q_targets = rewards + self.gamma * new_q * (1.0-dones)
            q_targets += this_q_targets

        q_targets = q_targets/self.nactors

        # Critic 1
        with tf.GradientTape() as tape:
            q_values1 = self.critic_model1([states, actions], training=False)
            td_errors1 = q_values1-q_targets
            #priority_buffer1 = np.abs(td_errors1.numpy()+1e-8)
            #self.priority_buffer1 = tf.math.abs(td_errors1)
            critic_loss1 = tf.reduce_mean(tf.math.square(td_errors1))
        gradient1 = tape.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(gradient1, self.critic_model1.trainable_variables))

        # Critic 2
        with tf.GradientTape() as tape:
            q_values2 = self.critic_model2([states, actions], training=False)
            td_errors2 = q_values2-q_targets
            #priority_buffer2 = np.abs(td_errors2.numpy()+1e-8)
            #self.priority_buffer2 = tf.math.abs(td_errors2)
            critic_loss2 = tf.reduce_mean(tf.math.square(td_errors2))
        gradient2 = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(gradient2, self.critic_model2.trainable_variables))

        # average_priority_buffer = (priority_buffer1 + priority_buffer2) / 2
        # max_average_priority_buffer = np.max(average_priority_buffer)
        # self.priority_buffer[self.batch_indices] = average_priority_buffer / max_average_priority_buffer

        # Update the priority buffer
        #self.priority_buffer[self.batch_indices] = (priority_buffer1+priority_buffer2)/2

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        for i in range(self.nactors):
            next_rdm_gaus = tf.random.normal([states.shape[0], self.nrdm_inputs], 0, 1, tf.float32, seed=1)
            with tf.GradientTape() as tape:
                actions = self.actor_models[i]([states, next_rdm_gaus], training=True)
                q_value = self.critic_model1([states, actions], training=False)
                #q_value2 = self.critic_model2([states, actions], training=False)
                #q_value = tf.keras.layers.Average()([q_value1, q_value2])
                loss = -tf.math.reduce_mean(q_value)
            gradient = tape.gradient(loss, self.actor_model.trainable_variables)
            self.actor_optimizers[i].apply_gradients(zip(gradient, self.actor_models[i].trainable_variables))
            self.soft_update(self.target_actors[i].variables, self.actor_models[i].variables)

    #    @tf.function
    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """

        self.nactions.assign(self.nactions + 1)

        if self.buffer_counter < self.batch_size:
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
            return sampled_action, noise

        # Single try
        state = np.expand_dims(state, 0)

        # Try multiple times
        nrepeats = 250
        states = tf.repeat(state, nrepeats, axis=0)

        # Loop over models
        max_reward = -999999
        max_sampled_action = None
        for i in range(self.nactors):
            rdm_norms = tf.random.normal([nrepeats, self.nrdm_inputs], 0, 1, tf.float32, seed=1)
            sampled_actions = self.actor_models[i]([states, rdm_norms])
            new_q1 = self.target_critic1([states, sampled_actions])
            new_q2 = self.target_critic2([states, sampled_actions])
            rewards = tf.math.maximum(new_q1, new_q2)
            ireward = np.argmax(rewards)
            this_sampled_action = sampled_actions[ireward]
            noise = tf.zeros(this_sampled_action.shape)
            if max_reward<rewards[ireward]:
                sampled_action = this_sampled_action
                max_reward = rewards[ireward]

        sampled_action = np.squeeze(sampled_action)
        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))

        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)], [np.squeeze(noise)]
