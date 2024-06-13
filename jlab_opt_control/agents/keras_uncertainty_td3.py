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
from jlab_opt_control.agents.keras_td3 import KerasTD3
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

td3_log = logging.getLogger("TD3-Uncertainty-Agent")
td3_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class KerasUncertaintyTD3(KerasTD3):

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones, weights):
        # Generate the proper noise
        noise = (tf.random.normal(tf.shape(actions), dtype=tf.float32) * 0.2)
        noise_clipped = tf.clip_by_value(
            noise, -self.noise_clip, self.noise_clip) * self.target_actor.action_scale
        next_actions = tf.clip_by_value(self.target_actor(
            next_states, training=False) + noise_clipped, self.lower_bound, self.upper_bound)

        target_q1, _ = self.target_critic1(next_states, next_actions, training=False)
        target_q2, _ = self.target_critic2(next_states, next_actions, training=False)
        target_q = tf.math.minimum(target_q1, target_q2)

        # Bellman equation for the q value
        q_targets = rewards + self.gamma * target_q * (1.0 - dones)

        # Critic 1 and 2
        with tf.GradientTape() as tape:
        
            q_values1, q_logvar1 = self.critic_model1(states, actions, training=True)
            q_values2, q_logvar2 = self.critic_model2(states, actions, training=True)

            q_values1 = q_values1 + tf.random.normal(tf.shape(q_values1), dtype=tf.float32) * tf.exp(0.5 * q_logvar1)
            q_values2 = q_values2 + tf.random.normal(tf.shape(q_values2), dtype=tf.float32) * tf.exp(0.5 * q_logvar2)

            td_errors1 = q_values1 - q_targets
            td_errors2 = q_values2 - q_targets

            beta = 1e-5
            critic_loss1 = self.mse_loss(q_values1, q_targets, sample_weight=weights) \
                + beta * 0.5 * tf.reduce_sum(q_values1*q_values1 + tf.exp(q_logvar1) - q_logvar1 - 1)
            critic_loss2 = self.mse_loss(q_values2, q_targets, sample_weight=weights) \
                + beta * 0.5 * tf.reduce_sum(q_values2*q_values2 + tf.exp(q_logvar2) - q_logvar2 - 1)
            
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
            actions = self.actor_model(states, training=True)
            q_value, _ = self.critic_model1(states, actions, training=False)
            loss = -tf.math.reduce_mean(q_value)
            with tf.GradientTape(persistent=True) as tape2:
                dq_da, dq_ds = tape2.gradient(q_value, [actions, states])
                print(f'dq_da: {dq_da}')
                print(f'dq_ds: {dq_ds}')

        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradient, self.actor_model.trainable_variables))
        return loss

    def action(self, state, train=True, inference=False):
        """ Method used to provide the next action using the target model """
        # Warmup experience sample
        if (self.buffer.size() < np.max([self.batch_size, self.warmup_size])) and inference == False:
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
        # Warmup completed, sample from actor or run inference
        else:
            state = tf.expand_dims(state, 0)
            sampled_action = (self.actor_model(state)).numpy()
            if train:
                if False:
                    noise = (tf.random.normal(shape=(self.num_actions,), mean=0,
                            stddev=self.actor_model.action_scale * 0.1, dtype=tf.float32)).numpy()
                    sampled_action = np.clip(
                        sampled_action + noise, self.lower_bound, self.upper_bound)
                else:
                    N = 256
                    sampled_actions = (tf.random.uniform(
                        shape=(N, self.num_actions), 
                        minval=self.lower_bound, 
                        maxval=self.upper_bound, 
                        dtype=tf.float32)).numpy()
                    sampled_states = np.repeat(state, N, axis=0)
                    _, q_logvar = self.critic_model1(sampled_states, sampled_actions, training=False)

                    # Choose the action with the highest uncertainty
                    action_idx = np.argmax(np.exp(0.5 * q_logvar))
                    sampled_action = sampled_actions[action_idx, None]
                    noise = np.zeros(self.num_actions)
                                          
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
                tf.summary.scalar('Action', data=sampled_action,
                                  step=int(self.nactions))
            else:
                for i in range(self.num_actions):
                    tf.summary.scalar('Action #{}'.format(
                        i), data=sampled_action[i], step=int(self.nactions))

        # Insure action output by actor is in legal environment range
        return sampled_action, noise