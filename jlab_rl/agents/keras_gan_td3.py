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

class KerasGenerativeTD3(KerasTD3):

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        #
        next_rdm_gaus = tf.random.normal([next_states.shape[0], self.num_actions + self.num_states], 0, 1, tf.float32, seed=1)
        next_actions = self.target_actor([next_states, next_rdm_gaus], training=False)
        #
        new_q1 = self.target_critic1([next_states, next_actions], training=False)
        new_q2 = self.target_critic2([next_states, next_actions], training=False)
        new_q = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        q_targets = rewards + self.gamma * new_q * (1.0-dones)
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

    @tf.function
    def train_actor(self, states):
        # Use Critic 1
        next_rdm_gaus = tf.random.normal([states.shape[0], self.num_actions + self.num_states], 0, 1, tf.float32, seed=1)
        with tf.GradientTape() as tape:
            actions = self.actor_model([states, next_rdm_gaus], training=True)
            q_value = self.critic_model1([states, actions], training=False)
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

    def get_actor(self):
        model = Generator(ndims=self.num_actions, nlayers=4, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return model

#    @tf.function
    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """

        # Single try
        state = np.expand_dims(state, 0)
        # rdm_norms = tf.random.normal([1, self.num_actions + self.num_states], 0, 1, tf.float32, seed=1)
        # sampled_action = self.actor_model([state, rdm_norms])

        # Try multiple times
        nrepeats = 100
        #print('state:', state.shape)
        states = tf.repeat(state, nrepeats, axis=0)
        #print('states:', states.shape)
        # states = np.reshape(states, (state.shape[0], state.shape[1], nrepeats))
        # print('states:', states.shape)
        rdm_norms = tf.random.normal([nrepeats, self.num_actions + self.num_states], 0, 1, tf.float32, seed=1)
        sampled_actions = self.actor_model([states, rdm_norms])
        new_q1 = self.target_critic1([states, sampled_actions])
        new_q2 = self.target_critic2([states, sampled_actions])
        rewards = tf.math.maximum(new_q1, new_q2)
        #rewards = tf.math.minimum(new_q1, new_q2)
        ireward = np.argmax(rewards)
        sampled_action = sampled_actions[ireward]

        #print(states)
        # rdm_norms = np.random.normal(0, 1, (nrepeats, self.num_actions + 2))
        # print('states:', states.shape)
        # print('rdm gauss', rdm_norms.shape)
        # sampled_actions = self.actor_model([states, rdm_norms])
        # print('gen actions:', sampled_actions.shape)
        # # rewards1 = self.target_critic1([states, sampled_actions])
        # # rewards2 = self.target_critic2([states, sampled_actions])
        # #rewards = (rewards1 + rewards2)/2.0
        # rewards = self.critic_model1([states, sampled_actions])
        # print('gen rewards:', rewards.shape)
        # ireward = np.argmax(rewards)
        # print('max reward:', rewards[ireward].numpy)
        # sampled_action = sampled_actions[ireward]

        # rdm_norms = rdm_norms[:,0]
        # rdm_norms = np.expand_dims(rdm_norms, 0)
        # sampled_action = self.actor_model([state, rdm_norms ])
        #sampled_action = self.actor_model(rdm_norms)
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
        return [np.squeeze(sampled_action)], [np.squeeze(noise)]
#        return [np.squeeze(legal_action)], [np.squeeze(noise)]

