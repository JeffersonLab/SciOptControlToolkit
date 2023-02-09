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

import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
import numpy as np
from os.path import join
import time
from jlab_rl.agents.keras_td3 import KerasTD3
from jlab_rl.models.keras_dgpa_actor import Keras_Actor_DGPA
from jlab_rl.models.keras_dgpa_actor import euclidean_dist


class KerasTD3ActorDGPA(KerasTD3):

    @tf.function
    def train_critic(self, states, actions, rewards, next_states):
        next_actions, next_actions_std = self.target_actor_dpga_model(next_states, training=False)
        next_actions_std = next_actions_std / self.num_states
        noise = tf.random.normal(next_actions.shape,
                                 mean=np.zeros(next_actions.shape),
                                 stddev=5 * next_actions_std)
        next_legal_actions = next_actions + noise
        new_q1 = self.target_critic1([next_states, next_legal_actions], training=False)
        new_q2 = self.target_critic2([next_states, next_legal_actions], training=False)
        new_q = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        q_targets = rewards + self.gamma * new_q

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

    def soft_update_dgpa(self):
        # Update weights
        self.soft_update(self.target_actor_dpga_model.variables, self.actor_dpga_model.variables)
        # # Update cov, etc.
        self.target_actor_dpga_model.cov = self.actor_dpga_model.cov
        self.target_actor_dpga_model.W = self.actor_dpga_model.W
        self.target_actor_dpga_model.b = self.actor_dpga_model.b
        self.target_actor_dpga_model.scale = self.actor_dpga_model.scale
        self.target_actor_dpga_model.counts = self.actor_dpga_model.counts
        self.target_actor_dpga_model.calib = self.actor_dpga_model.calib

    # TODO: this function is slow!!!
    @tf.function
    def train_actor_dpga(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            actions, fourier_pred, hidden_x, input_x, y_std = self.actor_dpga_model(states, training=True)
            q_value = self.critic_model1([states, actions], training=False)
            loss_mu = -tf.math.reduce_mean(q_value)
            # Bi-Lip
            hidden_x = (hidden_x - tf.reduce_min(hidden_x)) / (tf.reduce_max(hidden_x) - tf.reduce_min(hidden_x))
            l1, l2 = self.actor_dpga_model.l_bounds
            inp_dist = tf.numpy_function(euclidean_dist, [input_x], tf.float32)
            hidden_dist = tf.numpy_function(euclidean_dist, [hidden_x], tf.float32)
            c1 = l1 * inp_dist - hidden_dist
            c2 = hidden_dist - l2 * inp_dist
            loss1 = tf.reduce_mean(tf.nn.relu(c1))
            loss2 = tf.reduce_mean(tf.nn.relu(c2))
            distance_loss =  (loss1 + loss2) / 2.0
            # original total loss
            loss = loss_mu + distance_loss

            # Testing to include error term
            # s_pred = tf.math.log(tf.math.square(y_std))
            # loss_term1 = tf.math.exp(-s_pred) * q_value * q_value
            # loss_term2 = s_pred
            # loss = 0.5 * tf.math.reduce_mean(loss_term1 + loss_term2) + distance_loss

        gradient = tape.gradient(loss, self.actor_dpga_model.trainable_variables)
        del tape

        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_dpga_model.trainable_variables))
        # Predictive Covariance update
        phi = tf.stop_gradient(fourier_pred)
        self.actor_dpga_model.update_cov(phi)
        # P = tf.linalg.matmul(phi, tf.transpose(phi))
        # S = tf.eye(self.actor_dpga_model.fourier_dim) - \
        #     tf.linalg.matmul(
        #         tf.linalg.inv(P + (self.actor_dpga_model.scale ** 2) * tf.eye(self.actor_dpga_model.fourier_dim)), P)
        # # Bug is here
        # if self.actor_dpga_model.counts > 1:
        #     self.actor_dpga_model.cov = 0.99 * self.actor_dpga_model.cov + 0.01 * tf.linalg.matmul(P, S)
        # #
        #tf.summary.scalar('Action loss', data=loss, step=int(self.actor_dpga_model.counts))

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        self.train_critic(state_batch, action_batch, reward_batch, next_state_batch)
        self.train_actor_dpga(state_batch)
        self.actor_dpga_model.counts += 1

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
        if self.ntrain_calls % self.actor_update_freq == 0:
            self.soft_update_dgpa()
        if self.ntrain_calls % self.critic_update_freq == 0:
            self.soft_update(self.target_critic1.variables, self.critic_model1.variables)
            self.soft_update(self.target_critic2.variables, self.critic_model2.variables)

    def action(self, state, train=True, monitoring=False):
        """ Method used to provide the next action using the target model """
        self.nactions = self.nactions + 1
        # TD3 version
        if self.buffer_counter < self.min_buffer_counter:
            sampled_actions = self.env.action_space.sample()
            noise = 0
        else:
            # Normal actor
            state = np.expand_dims(state, 0)
            # DPGA actor
            sampled_dgpa_actions, sampled_dgpa_actions_std = self.actor_dpga_model(state)
            sampled_dgpa_actions_std = sampled_dgpa_actions_std / self.num_states
            # sampled_actions = sampled_dgpa_actions
            noise = tf.random.normal(sampled_dgpa_actions.shape,
                                     mean=np.zeros(sampled_dgpa_actions.shape),
                                     stddev=5 * sampled_dgpa_actions_std)
            sampled_actions = sampled_dgpa_actions + noise
            if monitoring:
                for i in range(self.num_actions):
                    tf.summary.scalar('Action_{} mean'.format(i), data=sampled_dgpa_actions[0][i],
                                      step=int(self.nactions))
                    tf.summary.scalar('Action_{} noise'.format(i), data=noise[0][i], step=int(self.nactions))
                    if sampled_dgpa_actions[0][i] != 0:
                        rel_std = float(sampled_dgpa_actions_std[0] / np.abs(sampled_dgpa_actions[0][i]))
                        tf.summary.scalar('Action_{} std'.format(i), data=rel_std, step=int(self.nactions))
            legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)], [np.squeeze(noise)]

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
        rff = 256
        self.actor_dpga_model = Keras_Actor_DGPA(hidden_size=self.hidden_size,
                                                 num_inputs=self.num_states,
                                                 num_outputs=self.num_actions,
                                                 fourier_dim=rff,
                                                 upper_bound=self.upper_bound,
                                                 lower_bound=self.lower_bound)
        # print(self.actor_dpga_model.cov)
        self.target_actor_dpga_model = Keras_Actor_DGPA(hidden_size=self.hidden_size,
                                                        num_inputs=self.num_states,
                                                        num_outputs=self.num_actions,
                                                        fourier_dim=rff,
                                                        upper_bound=self.upper_bound,
                                                        lower_bound=self.lower_bound)

        self.soft_update_dgpa()

        seed1 = time.time_ns()
        print('seed1:', seed1)
        tf.random.set_seed(seed1)
        self.critic_model1 = self.get_critic()
        self.target_critic1 = self.get_critic()
        self.target_critic1.set_weights(self.critic_model1.get_weights())

        seed2 = time.time_ns()
        print('seed2:', seed2)
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
