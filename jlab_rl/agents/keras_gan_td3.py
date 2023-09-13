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

import sys
#import jlab_rl as jlab_rl
import tensorflow as tf
from jlab_rl.models.state_generator import Generator_v3 as Generator
from jlab_rl.agents.keras_td3 import KerasTD3
#from scipy.stats import wasserstein_distance

#from tensorflow.keras.initializers import RandomUniform
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import os
from os.path import join
import time

class KerasGenerativeTD3(KerasTD3):
    """ Define all key variables required for all agent """


    def __init__(self, env, warmup_size, nrff=0, logdir=None, model_load_path=None, model_save_path=None, **kwargs):
        """ Define all key variables required for all agent """

        self.rdm_intputs = 100
        self.nactor_layers = 6 # (was 4)
        self.ncritic_layers = 6

        # Get env info
        super().__init__(env, warmup_size, nrff, logdir, model_load_path, model_save_path, **kwargs)
        print('Running KerasGenerativeTD3 __init__')
        self.batch_size = 512
        self.ntrain_actor_calls = 0
        # Re-init models
        self.initialize_new_models()

    def get_actor(self):
        model = Generator(ndims=self.num_actions, nlayers=self.nactor_layers, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return model

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.train_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        if self.buffer_counter >= self.min_buffer_counter:
            self.ntrain_actor_calls += 1
            td_loss, kl_loss = self.train_actor(state_batch)
            tf.summary.scalar('Actor TD Loss', data=td_loss, step=int(self.ntrain_actor_calls))
            tf.summary.scalar('Actor KL Loss', data=kl_loss, step=int(self.ntrain_actor_calls))
            tf.summary.scalar('Actor Total Loss', data=td_loss + kl_loss, step=int(self.ntrain_actor_calls))

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        #
        next_rdm_gaus = tf.random.normal([next_states.shape[0], self.rdm_intputs], 0, 1, tf.float32, seed=time.time_ns())
        next_actions = self.target_actor([next_states, next_rdm_gaus], training=False)
        # Do we need this noise ?
        # noises = tf.random.normal(next_actions.shape, 0, 0.2)
        # noises = tf.clip_by_value(noises, -0.5, 0.5)
        # next_actions = next_actions+noises
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
        next_rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, 1, tf.float32, seed=time.time_ns())
        with tf.GradientTape() as tape:
            actions = self.actor_model([states, next_rdm_gaus], training=True)
            q_value = self.critic_model1([states, actions], training=False)
            #q_value2 = self.critic_model2([states, actions], training=False)
            #q_value = tf.keras.layers.Average()([q_value1, q_value2])
            td_loss = -tf.math.reduce_mean(q_value)
        #gradient = tape.gradient(td_loss, self.actor_model.trainable_variables)
        #self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

        # Add KL-div using top 5% of the warmup samples
        w_rewards = self.reward_buffer[0:self.min_buffer_counter]
        w_states = self.state_buffer[0:self.min_buffer_counter]
        w_actions = self.action_buffer[0:self.min_buffer_counter]
        isort_reward = np.argsort(np.squeeze(w_rewards))
        idx_thr = int(0.90 * self.min_buffer_counter)
        isort_top_reward = isort_reward[idx_thr:]
        top_states = w_states[isort_top_reward]
        top_actions = w_actions[isort_top_reward]

        top_next_rdm_gaus = tf.random.normal([top_states.shape[0], self.rdm_intputs], 0, 1, tf.float32, seed=time.time_ns())
        with tf.GradientTape() as tape:
            this_actions = self.actor_model([top_states, top_next_rdm_gaus], training=True)
            #wd_loss = wasserstein_distance(top_actions,this_actions)
            kl_loss = tf.math.reduce_sum(tf.keras.losses.kl_divergence(top_actions, this_actions))
            # print('top_actions ', top_actions)
            # print('this_actions ', this_actions)
            #print('wd_loss ', wd_loss)
            #
            # sys.exit()
        gradient = tape.gradient(kl_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
        return td_loss, kl_loss

    #    @tf.function
    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """

        self.nactions.assign(self.nactions + 1)

        if self.buffer_counter < np.max([self.batch_size, self.min_buffer_counter]):
            #true_params = [0.72916667, 0.25, 0.6, 0.36458333, 0.25, 0.8]
            #sampled_action = np.random.normal(true_params, 0.25)
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
            # return sampled_action, noise

        else:
            # Single try
            state = np.expand_dims(state, 0)

            rdm_norms = tf.random.normal([1, self.rdm_intputs], 0, 1, tf.float32, seed=time.time_ns())
            sampled_action = self.actor_model([state, rdm_norms])

            # Try multiple times
            nrepeats = 100
            states = tf.repeat(state, nrepeats, axis=0)
            rdm_norms = tf.random.normal([nrepeats, self.rdm_intputs], 0, 1, tf.float32, seed=time.time_ns())
            sampled_actions = self.actor_model([states, rdm_norms])
            #
            if train:
               sampled_actions = np.random.normal(sampled_actions, 0.5, sampled_actions.shape)

            new_q1 = self.target_critic1([states, sampled_actions])
            new_q2 = self.target_critic2([states, sampled_actions])
            rewards = tf.math.maximum(new_q1, new_q2)
            rewards = np.squeeze(rewards)


            # isort_reward = np.argsort(rewards)
            # isort_reward_sub = isort_reward[-25:]
            # rdm_idx = isort_reward_sub[np.random.randint(0,24)]
            # # print(rdm_idx)
            # # print(isort_reward_sub)
            # # print(rewards[isort_reward_sub])
            # # print(rewards[rdm_idx])
            # # sys.exit()
            # sampled_action = sampled_actions[rdm_idx]
            # noise = tf.zeros(sampled_action.shape)

            ireward = np.argmax(rewards)
            sampled_action = sampled_actions[ireward]
            noise = tf.zeros(sampled_action.shape)

            sampled_action = np.squeeze(sampled_action)#sampled_action = sampled_action.flatten()
            noise = np.squeeze(noise)#noise = noise.flatten()

            # if train:
            #     noise = np.random.normal(0, 0.1, self.num_actions)
            #     sampled_action = sampled_action + noise

        #sampled_action = np.squeeze(sampled_action)
        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))

        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)], [np.squeeze(noise)]
