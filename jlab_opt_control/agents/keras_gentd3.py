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

import random
import tensorflow as tf
from jlab_opt_control.models.generative_actor import GenerativeActor
from jlab_opt_control.agents.keras_td3 import KerasTD3
import numpy as np
import time

import logging

gentd3_log = logging.getLogger("GenTD3-Agent")
gentd3_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

class KerasGenerativeTD3(KerasTD3):
    """ Define all key variables required for all agent """

    def __init__(self, env, logdir, cfg='keras_td3.json', **kwargs):

        # Standard TD3 setup
        self.layer_nodes = [256, 256, 256, 256]
        self.ncritic_layers = 4
        self.hidden_size = 256
        self.batch_size = 1000
        self.ntrain_actor_calls = 0

        # Get env info
        super().__init__(self, env, logdir, cfg='keras_td3.json', **kwargs)
        gentd3_log.info('Running KerasGenerativeTD3 __init__')

        # Used for random samples
        self.rdm_intputs = 100
        self.norm_sdt = 2.0

        # For quantile annealing
        self.epsilon = 1.0
        self.min_epsilon = 0.001
        self.best_qvalue = -9999
        self.decay_epsilon = 0.9995

        # Reference distribution
        self.dynamic_ref = dynamic_ref
        self.top_states = None
        self.top_actions = None
        self.top_rewards = None

        # Score
        self.scores = [0] * self.num_actions

        self.max_size = np.max([self.batch_size, self.min_buffer_counter])
        gentd3_log.info('max_size:', self.max_size)

        # Re-init models
        self.initialize_new_models()

    def get_actor(self):
        model = GenerativeActor(nactions=self.num_actions, layer_nodes=self.layer_nodes,
                                lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return model

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # Check the shapes of the training data
        assert state_batch.shape == (self.batch_size, self.num_states), "State_batch shape incorrect in update function"
        assert action_batch.shape == (
        self.batch_size, self.num_actions), "action_batch shape incorrect in update function"
        assert reward_batch.shape == (self.batch_size, 1), "reward_batch shape incorrect in update function"
        assert next_state_batch.shape == (
        self.batch_size, self.num_states), "next_state_batch shape incorrect in update function"
        assert done_batch.shape == (self.batch_size, 1), "done_batch shape incorrect in update function"

        # Train Critic
        critic_loss1, critic_loss2 = self.train_critic(state_batch, action_batch, reward_batch, next_state_batch,
                                                       done_batch)
        tf.summary.scalar('Critic #1 Loss', data=critic_loss1, step=int(self.buffer_counter))
        tf.summary.scalar('Critic #2 Loss', data=critic_loss2, step=int(self.buffer_counter))

        # Train actor
        if self.buffer_counter >= self.max_size:
            self.ntrain_actor_calls += 1
            # Train
            td_loss, kl_loss = self.train_actor(state_batch)  # self.top_states)
            tf.summary.scalar('Actor TD-error Loss', data=td_loss, step=int(self.ntrain_actor_calls))
            tf.summary.scalar('Actor Distance Loss', data=kl_loss, step=int(self.ntrain_actor_calls))

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):

        next_rdm_gaus = tf.random.normal([next_states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32,
                                         seed=time.time_ns())
        next_actions = self.target_actor([next_states, next_rdm_gaus], training=False)

        # Do we need this noise ?
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

        return critic_loss1, critic_loss2

    # @tf.function
    def train_actor(self, states):

        chosen_indices = np.random.choice(np.arange(self.max_size), size=self.batch_size)
        top_actions = self.top_actions[chosen_indices]
        top_actions = tf.cast(top_actions, dtype=tf.float32)
        if self.num_actions == 2:
            top_actions0, top_actions1 = tf.split(top_actions, num_or_size_splits=self.num_actions, axis=1)
            top_actions0, top_actions1 = np.squeeze(top_actions0), np.squeeze(top_actions1)
            angles = np.arctan2(top_actions1, top_actions0)
            sorted_indices = np.argsort(angles)
            top_actions0 = top_actions0[sorted_indices]
            top_actions1 = top_actions1[sorted_indices]
            top_actions = [top_actions0, top_actions1]
            # define the loss function to be 2d
            # To be used in the gradient tape
            loss_function = get_score_2d
        elif self.num_actions == 1:
            top_actions = np.squeeze(top_actions)
            top_actions = np.sort(top_actions)
            # define the loss function to be 1d
            # To be used in the gradient tape
            loss_function = get_score  # get_score_1d

        # top_actions = random.choice(top_actions,k=self.batch_size)
        with tf.GradientTape() as tape:
            next_rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32,
                                             seed=time.time_ns())
            self.training_actions = self.actor_model([states, next_rdm_gaus], training=True)
            self.training_actions = tf.squeeze(self.training_actions)
            # score = tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(self.training_actions, top_actions)))
            # training_actions0, training_actions1 = tf.squeeze(training_actions0), tf.squeeze(training_actions1)
            # score, score1, score2 = loss_function(self.training_actions, top_actions)
            score = 0  # score*100.0 # in order for it to be on the same scale as the q_value scale
            # print('score:', score)
            # score = tf.math.reduce_mean(tf.losses.kl_divergence(self.top_actions, self.training_actions))
            q_value = self.critic_model1([states, self.training_actions], training=False)
            td_loss = -tf.math.reduce_mean(q_value)

            # print('self.top_actions.shape:', self.top_actions.shape)
            # print('self.training_actions.shape:', self.training_actions.shape)
            # score = tf.math.reduce_mean(tf.losses.kl_divergence(self.top_actions, self.training_actions))
            # print(score)
            # sys.exit()
            # for i in range(self.num_actions):
            #     self.scores[i] =  score
            #         # tf.math.reduce_mean(
            #         # tf.losses.kl_divergence(tf.convert_to_tensor(self.top_actions[:,i]),
            #         #                         tf.convert_to_tensor(self.training_actions[:,i])))
            #     # top_sorted = tf.sort(self.top_actions[:,i])
            #     # print('top_sorted', top_sorted.shape)
            #     # current_sorted = tf.sort(self.training_actions[:,i])
            #     # print('current_sorted', current_sorted.shape)
            #     # self.scores[i] = tf.compat.v1.losses.absolute_difference(top_sorted, current_sorted)
            #     # #tf.reduce_sum(tf.math.squared_difference(top_sorted-current_sorted))
            #     score += self.scores[i]

            # tf_top_actions = tf.convert_to_tensor(self.top_actions[:,i], dtype=float)
            # print('type:', tf_top_actions)
            # # print('type:', actions[:,i])
            # score += get_score_1d(tf_top_actions, actions[:,i])
            # Binary
            # print('actions:', actions.shape)
            # print('self.top_actions:', self.top_actions.shape)
            # score = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(self.top_actions, actions))
            # score = tf.math.reduce_mean(tf.keras.losses.mse(self.top_actions, actions))
            # print('td_loss:', td_loss.shape)
            # print('score:', score.shape)
            # sys.exit()
            # Add distance
            # top_next_rdm_gaus = tf.random.normal([self.top_states.shape[0],
            #                                       self.rdm_intputs], 0, self.norm_sdt, tf.float32,seed=time.time_ns())
            # this_actions = self.actor_model([self.top_states, top_next_rdm_gaus], training=True)
            # top_actions = tf.cast(self.top_actions, dtype=tf.float32)
            # training_actions = tf.cast(self.training_actions, dtype=tf.float32)
            # if self.top_actions.shape[1] > 1:
            #     score, score1, score2 = get_score(training_actions, top_actions) # For ND problems
            # else:
            #     score, score1, score2 = get_score_1d(training_actions, top_actions) # For 1D problems
            # for i in range(self.num_actions):
            #     top_actions = tf.cast(self.top_actions[:, i], dtype=tf.float32)
            #     training_actions = tf.cast(self.training_actions[:, i], dtype=tf.float32)
            #     self.scores[i] = tf.reduce_mean(tf.math.square(tf.sort(training_actions) - tf.sort(top_actions)))
            # score = tf.math.reduce_sum(self.scores)
            # print(score)
            total_loss = score + td_loss  # + score

        gradient = tape.gradient(total_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
        return td_loss, score

    def get_critic_qvalue(self, state, nrepeats=100):
        states = tf.repeat(state, nrepeats, axis=0)
        rdm_actions = tf.random.uniform([nrepeats, self.num_actions], \
                                        self.lower_bound, self.upper_bound, tf.float32, seed=time.time_ns())
        new_q1 = self.target_critic1.predict_on_batch([states, rdm_actions])
        new_q2 = self.target_critic2.predict_on_batch([states, rdm_actions])
        q_mean = np.mean([new_q1, new_q2], axis=0)
        q_std = np.std([new_q1, new_q2], axis=0)
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
        rdm_action = rdm_action_q_ucb[0]
        rdm_ucb = rdm_action_q_ucb[1]
        return rdm_action, rdm_ucb

    def get_policy_qvalue(self, state, nrepeats=100):
        states = tf.repeat(state, nrepeats, axis=0)
        rdm_norms = tf.random.normal([nrepeats, self.rdm_intputs], 0, self.norm_sdt, tf.float32, seed=time.time_ns())
        sampled_actions = self.actor_model.predict_on_batch([states, rdm_norms])
        new_q1 = self.target_critic1.predict_on_batch([states, sampled_actions])
        new_q2 = self.target_critic2.predict_on_batch([states, sampled_actions])
        new_q = tf.math.maximum(new_q1, new_q2)
        new_q = tf.squeeze(new_q)

        q_threshold = np.quantile(new_q, 1 - self.epsilon)
        percentile_xyz = []
        for i, val in enumerate(zip(sampled_actions, new_q)):
            this_action, this_q = val
            if this_q >= q_threshold:
                percentile_xyz.append((this_action, this_q))
        policy_action_q = random.choice(percentile_xyz)
        sampled_action = policy_action_q[0]
        sampled_q = policy_action_q[1]

        return sampled_action, sampled_q

    def action_inference(self, states):

        rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32,
                                    seed=time.time_ns())
        actions = self.actor_model([states, rdm_gaus])
        rewards = []
        for a in actions:
            self.env.reset()
            _, reward, _, _, _ = self.env.step(a)
            rewards.append(reward)
        return np.squeeze(actions), np.squeeze(rewards)

    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """

        assert state.shape == self.num_states, f'Shape of the input state {state.shape} to action method is not correct...'

        self.nactions.assign(self.nactions + 1)
        state = tf.expand_dims(state, 0)
        action_type = 0
        if self.buffer_counter <= self.max_size:
            sampled_action = self.env.action_space.sample()
        else:
            # Calculate q-value from critic sampling
            rdm_action_q_ucb = self.get_critic_qvalue(state, 100)
            policy_action_q_ucb = self.get_policy_qvalue(state, 100)

            sampled_action = rdm_action_q_ucb[0].numpy()
            if policy_action_q_ucb[1] > rdm_action_q_ucb[1]:
                self.epsilon = self.epsilon * self.decay_epsilon  # Need to add annealing
                self.epsilon = self.epsilon if self.epsilon > self.min_epsilon else self.min_epsilon
                sampled_action = policy_action_q_ucb[0]
                action_type = 1

        sampled_action = sampled_action.flatten()
        assert sampled_action.shape == self.num_actions or sampled_action.shape == (
        self.num_actions,), "Sampled action shape is incorrect..."

        # sampled_action = np.squeeze(sampled_action)
        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))

        tf.summary.scalar('Annealing Term', data=self.epsilon, step=int(self.nactions))
        tf.summary.scalar('Action Type', data=action_type, step=int(self.nactions))
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(legal_action), action_type

    def memory(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]
        self.action_type_buffer[index] = obs_tuple[5]
        self.buffer_counter += 1

        if (self.buffer_counter >= self.max_size):  # np.max([self.batch_size, self.min_buffer_counter])):

            if self.top_actions is None:
                self.top_states = self.state_buffer[0:self.max_size]
                self.top_actions = self.action_buffer[0:self.max_size]
                self.top_rewards = self.reward_buffer[0:self.max_size]
            # ============ Dynamic Reference ==============================
            elif (self.dynamic_ref and obs_tuple[5] == 0):
                state = np.expand_dims(obs_tuple[0], axis=0)
                action = np.expand_dims(obs_tuple[1], axis=0)
                reward = np.expand_dims(obs_tuple[2], axis=0)
                reward = np.expand_dims(reward, axis=0)
                if self.num_actions == 1:
                    action = np.expand_dims(action, axis=0)
                merged_top_states = np.concatenate([self.top_states, state])
                merged_top_actions = np.concatenate([self.top_actions, action])
                merged_top_reward = np.concatenate([self.top_rewards, reward])
                isort_reward = np.argsort(np.squeeze(merged_top_reward))
                isort_top_reward = isort_reward[-self.max_size:]
                self.top_states = merged_top_states[isort_top_reward]
                self.top_actions = merged_top_actions[isort_top_reward]
                self.top_rewards = merged_top_reward[isort_top_reward]

        # ========================================================================
