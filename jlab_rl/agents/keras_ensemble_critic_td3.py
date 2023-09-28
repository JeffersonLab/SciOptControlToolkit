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

import sys, random
#import jlab_rl as jlab_rl
import tensorflow as tf
from jlab_rl.models.state_generator import Generator_v3 as Generator
from jlab_rl.agents.keras_td3 import KerasTD3
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import os
from os.path import join
import time
from jlab_rl.utils.score import get_score, get_score_1d

class KerasECGTD3(KerasTD3):
    """ Define all key variables required for all agent """


    def __init__(self, env, warmup_size, nrff=0, logdir=None, model_load_path=None, model_save_path=None, dynamic_ref=True, **kwargs):
        """ Define all key variables required for all agent """

        self.rdm_intputs = 100
        self.norm_sdt = 1
        self.nactor_layers = 3
        self.ncritic_layers = 3
        self.hidden_size = 256
        self.dynamic_ref = dynamic_ref
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.best_qvalue = -9999
        self.decay_epsilon = 0.999

        #
        self.ncritics = 7
        self.critic_models = []
        self.target_critics = []
        self.critic_optimizers = []

        # Get env info
        super().__init__(env, warmup_size, nrff, logdir, model_load_path, model_save_path, **kwargs)
        print('Running KerasECGTD3 __init__')
        self.batch_size = 500
        self.ntrain_actor_calls = 0

        self.top_states = None
        self.top_actions = None
        self.top_rewards = None
        self.n_top = warmup_size
        self.max_size = np.max([self.batch_size, self.min_buffer_counter])

        # Re-init models
        #self.initialize_new_models()

    def get_critic(self):

        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states))
        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))
        state_action = tf.keras.layers.Concatenate()([state_input, action_input])
        for _ in range(self.ncritic_layers):
            state_action = tf.keras.layers.Dense(self.hidden_size, activation=tf.keras.activations.selu)(state_action)
        outputs = tf.keras.layers.Dense(1, activation='linear')(state_action)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def get_actor(self):
        model = Generator(ndims=self.num_actions, nlayers=self.nactor_layers, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return model

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        critic_losses = self.train_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        for i in range(self.ncritics):
            tf.summary.scalar(f'Critic #{i} Loss', data=critic_losses[i], step=int(self.buffer_counter))

        if self.buffer_counter >= np.max([self.batch_size, self.min_buffer_counter]) and self.top_actions is not None:
            self.ntrain_actor_calls += 1
            # Train
            #print('state_batch:', state_batch)
            td_loss, kl_loss = self.train_actor(state_batch)
            #td_loss, kl_loss = self.train_actor(state_batch)
            tf.summary.scalar('Actor TD-error Loss', data=td_loss, step=int(self.ntrain_actor_calls))
            tf.summary.scalar('Actor Distance Loss', data=kl_loss, step=int(self.ntrain_actor_calls))
            tf.summary.scalar('Actor Total Loss', data=td_loss + kl_loss, step=int(self.ntrain_actor_calls))

    def get_ensemble_critic_predict(self, critics, states, actions):
        q_list = []
        for i in range(self.ncritics):
            q_list.append(critics[i]([states, actions], training=False))
        q_mean = tf.reduce_mean(q_list, axis=0)
        q_std = tf.math.reduce_std(q_list, axis=0)
        return q_mean, q_std

    #@tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        #
        next_rdm_gaus = tf.random.normal([next_states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32, seed=time.time_ns())
        next_actions = self.target_actor([next_states, next_rdm_gaus], training=False)
        # Do we need this noise ?
        noises = tf.random.normal(next_actions.shape, 0, 0.2)
        noises = tf.clip_by_value(noises, -0.5, 0.5)
        next_actions = next_actions+noises
        #
        q_mean, q_std = self.get_ensemble_critic_predict(self.critic_models, next_states, next_actions)

        # Bellman equation for the q value
        q_targets = rewards + self.gamma * q_mean * (1.0-dones)
        #
        critic_losses = []
        for i in range(self.ncritics):
            with tf.GradientTape() as tape:
                q_values = self.critic_models[i]([states, actions], training=False)
                td_errors = q_values-q_targets
                critic_loss = tf.reduce_mean(tf.math.square(td_errors))
                critic_losses.append(critic_loss)
            gradient = tape.gradient(critic_loss, self.critic_models[i].trainable_variables)
            self.critic_optimizers[i].apply_gradients(zip(gradient, self.critic_models[i].trainable_variables))

        return critic_losses

    def train_actor(self, states):
        next_rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32, seed=time.time_ns())
        with tf.GradientTape() as tape:
            actions = self.actor_model([states, next_rdm_gaus], training=True)
            q_mean, q_std = self.get_ensemble_critic_predict(self.critic_models, states, actions)
            # q_list = []
            # for i in range(self.ncritics):
            #     q_list.append(self.critic_models[i]([states, actions], training=False))
            # q_mean = tf.reduce_mean(q_list, axis=0)
            # #q_std = np.mean(q_list, axis=0)
            # TODO: should include the STD
            td_loss = -tf.math.reduce_mean(q_mean)
        #try:
        gradient = tape.gradient(td_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
        # except:
        #     print(q_mean)
        #     print(q_std)
        #     sys.exit()

        top_next_rdm_gaus = tf.random.normal([self.top_states.shape[0],
                                              self.rdm_intputs], 0, self.norm_sdt, tf.float32, seed=time.time_ns())
        with tf.GradientTape() as tape:
            this_actions = self.actor_model([self.top_states, top_next_rdm_gaus], training=True)
            this_actions = tf.cast(this_actions, dtype=tf.float32)
            top_actions = tf.cast(self.top_actions, dtype=tf.float32)
            if top_actions.shape[1] > 1:
                score, score1, score2 = get_score(this_actions, top_actions) # For ND problems
            else:
                score, score1, score2 = get_score_1d(this_actions,top_actions) # For 1D problems

        gradient = tape.gradient(score, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

        return td_loss, score

    def get_critic_qvalue(self, state):
        nrepeats = 100
        states = tf.repeat(state, nrepeats, axis=0)
        rdm_actions = tf.random.uniform([nrepeats, self.num_actions], \
                                        self.lower_bound, self.upper_bound, tf.float32, seed=time.time_ns())
        q_mean, q_std = self.get_ensemble_critic_predict(self.target_critics, states, rdm_actions)
        q_ucb = q_mean + 5.0 * q_std
        q_ucb = np.squeeze(q_ucb)

        q_threshold = np.quantile(q_ucb, 1 - self.epsilon)
        percentile_xyz, top_ucb_actions = [], []
        for i, val in enumerate(zip(rdm_actions, q_ucb)):
            this_action, this_ucb = val
            if this_ucb >= q_threshold:
                percentile_xyz.append((this_action, this_ucb))
                top_ucb_actions.append(this_action)
        rdm_action_q_ucb = random.choice(percentile_xyz)
        return rdm_action_q_ucb

    def get_policy_qvalue(self, state):
        nrepeats = 100
        states = tf.repeat(state, nrepeats, axis=0)
        rdm_norms = tf.random.normal([nrepeats, self.rdm_intputs], 0, self.norm_sdt, tf.float32, seed=time.time_ns())
        sampled_actions = self.actor_model([states, rdm_norms])
        q_list = []
        for i in range(self.ncritics):
            q_list.append(self.target_critics[i]([states, sampled_actions], training=False))
        q_mean = tf.reduce_mean(q_list, axis=0)
        q_std = tf.math.reduce_std(q_list, axis=0)
        # TODO: How do we use STD
        ireward = np.argmax(q_mean)
        sampled_action = sampled_actions[ireward]
        return sampled_action, q_mean[ireward]


    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """

        self.nactions.assign(self.nactions + 1)
        state = np.expand_dims(state, 0)
        #sampled_action = np.zeros(self.num_actions)
        #noise = np.zeros(self.num_actions)
        action_type = 0
        if self.buffer_counter <= self.max_size:
            sampled_action = self.env.action_space.sample()
        else:
            # Calculate q-value from critic sampling
            rdm_action_q_ucb = self.get_critic_qvalue(state)
            policy_action_q_ucb = self.get_policy_qvalue(state)
            sampled_action = rdm_action_q_ucb[0].numpy()
            if policy_action_q_ucb[1]>rdm_action_q_ucb[1]:
                self.epsilon = self.epsilon*self.decay_epsilon # Need to add annealing
                self.epsilon = self.epsilon if self.epsilon>self.min_epsilon else self.min_epsilon
                sampled_action = policy_action_q_ucb[0].numpy()
                action_type = 1

        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))

        tf.summary.scalar('Annealing Term', data=self.epsilon, step=int(self.nactions))
        tf.summary.scalar('Action Type', data=action_type, step=int(self.nactions))
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)], [action_type]

    def memory(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]
        action_type = obs_tuple[5][0]
        #print('action_type: ',action_type)
        #print('self.buffer_counter:', self.buffer_counter)
        self.buffer_counter += 1

        if (self.buffer_counter >= np.max([self.batch_size, self.min_buffer_counter])):

            if self.top_actions is None or self.top_actions is None:
                # Add KL-div using top N% of the warmup samples
                w_rewards = self.reward_buffer[0:self.min_buffer_counter]
                w_states = self.state_buffer[0:self.min_buffer_counter]
                w_actions = self.action_buffer[0:self.min_buffer_counter]
                #print('w_actions: ', w_actions.shape)
                isort_reward = np.argsort(np.squeeze(w_rewards))
                #self.n_top = int(0.25 * self.min_buffer_counter)
                isort_top_reward = isort_reward[-self.n_top:]
                # print('total/filtered: ', w_rewards.shape, self.n_top)
                # sys.exit()
                self.top_states = w_states[isort_top_reward]
                self.top_actions = w_actions[isort_top_reward]
                #print('top_actions: ', self.top_actions.shape)

                self.top_rewards = np.squeeze(w_rewards[isort_top_reward])
            # ============ Dynamic Reference ==============================
            elif (self.dynamic_ref and action_type==0):
                action = obs_tuple[1]
                action = np.expand_dims(obs_tuple[1], axis=0)
                if self.num_actions==1:
                    action = np.expand_dims(action, axis=0)
                merged_top_states = np.concatenate([self.top_states, np.expand_dims(obs_tuple[0], axis=0)])
                merged_top_actions = np.concatenate([self.top_actions, action])
                merged_top_reward = np.concatenate([self.top_rewards, np.expand_dims(obs_tuple[2], axis=0)])
                isort_reward = np.argsort(np.squeeze(merged_top_reward))
                isort_top_reward = isort_reward[-self.n_top:]
                self.top_states = merged_top_states[isort_top_reward]
                self.top_actions = merged_top_actions[isort_top_reward]
                self.top_rewards = merged_top_reward[isort_top_reward]

        # ========================================================================

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        print('Running KerasECGTD3 initialize_new_models()')

        self.actor_model = self.get_actor()
        self.target_actor = self.get_actor()
        self.target_actor.set_weights(self.actor_model.get_weights())

        print(f'Creating {self.ncritics} critics')
        for i in range(self.ncritics):
            seed = time.time_ns()
            tf.random.set_seed(seed)
            self.critic_models.append(self.get_critic())
            self.target_critics.append(self.get_critic())
            self.target_critics[i].set_weights(self.critic_models[i].get_weights())
            self.critic_optimizers.append(tf.keras.optimizers.legacy.Adam(self.critic_lr, epsilon=1e-08))
            time.sleep(1 / 10)

    def train(self):
        """ Method used to train """
        self.ntrain_calls += 1

        if self.buffer_counter>=self.batch_size:
            # Get sampling range
            record_range = min(self.buffer_counter, self.buffer_capacity)
            self.batch_indices = np.random.choice(record_range, self.batch_size)
            #print('batch_indices: ', self.batch_indices)
            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[self.batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[self.batch_indices])
            reward_batch = tf.convert_to_tensor(self.reward_buffer[self.batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[self.batch_indices])
            done_batch = tf.convert_to_tensor(self.done_buffer[self.batch_indices])
            done_batch = tf.cast(done_batch, dtype=tf.float32)
            # print('train action_batch:', action_batch)
            # sys.exit()

            self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            if self.ntrain_calls%self.actor_update_freq == 0:
                self.soft_update(self.target_actor.variables, self.actor_model.variables)
            if self.ntrain_calls%self.critic_update_freq == 0:
                for i in range(self.ncritics):
                    self.soft_update(self.target_critics[i].variables, self.critic_models[i].variables)
