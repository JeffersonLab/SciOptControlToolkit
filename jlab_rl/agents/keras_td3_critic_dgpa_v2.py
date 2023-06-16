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
from jlab_rl.models.keras_dgpa_critic_v2 import KerasCriticDGPA
from tensorflow.keras.initializers import RandomUniform

class KerasTD3CriticDGPA(KerasTD3):

    # @tf.function
    # def train_actor(self, states):
    #     # Use Critic 1
    #     with tf.GradientTape() as tape:
    #         actions = self.actor_model(states, training=True)
    #         state_action = tf.keras.layers.Concatenate()([states, actions])
    #         q_value = self.critic_model1(state_action, training=False)
    #         loss = -tf.math.reduce_mean(q_value)
    #     gradient = tape.gradient(loss, self.actor_model.trainable_variables)
    #     self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

    @tf.function
    def train_critic(self, states, actions, rewards, next_states):
        next_actions = self.target_actor(next_states, training=False)
        # Add a little noise
        noise = np.random.normal(0, 0.2, self.num_actions)
        noise = np.clip(noise, -0.5, 0.5)
        next_actions = next_actions+noise
        next_state_action = tf.keras.layers.Concatenate()([next_states, next_actions])
        new_q1, _, _ = self.target_critic1(next_state_action, training=False)
        new_q2, _, _ = self.target_critic2(next_state_action, training=False)
        new_q = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        q_targets = rewards + self.gamma * new_q
        state_action = tf.keras.layers.Concatenate()([states, actions])
        # Critic 1
        with tf.GradientTape() as tape:
            q_values1, q_values_std1, _ = self.critic_model1(state_action, training=False)
            td_errors1 = q_values1-q_targets
            # Testing to include error term
            s_pred = tf.math.log(tf.math.square(q_values_std1))
            critic_loss1_term1 = tf.math.exp(-s_pred) * tf.math.square(td_errors1)
            critic_loss1_term2 = s_pred
            critic_loss1 = 0.5 * tf.math.reduce_mean(critic_loss1_term1 + critic_loss1_term2)

        gradient1 = tape.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(gradient1, self.critic_model1.trainable_variables))

        # Critic 2
        with tf.GradientTape() as tape:
            q_values2, q_values_std2, _ = self.critic_model2(state_action, training=False)
            td_errors2 = q_values2-q_targets
            # Testing to include error term
            s_pred = tf.math.log(tf.math.square(q_values_std2))
            critic_loss2_term1 = tf.math.exp(-s_pred) * tf.math.square(td_errors2)
            critic_loss2_term2 = s_pred
            critic_loss2 = 0.5 * tf.math.reduce_mean(critic_loss2_term1 + critic_loss2_term2)

        gradient2 = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(gradient2, self.critic_model2.trainable_variables))

    def soft_update_dgpa(self):
        # Update weights
        self.soft_update(self.target_critic1.trainable_variables, self.critic_model1.trainable_variables)
        # # Update cov, etc.
        self.target_critic1.mean = self.critic_model1.mean
        self.target_critic1.var = self.critic_model1.var
        self.target_critic1.old_var = self.critic_model1.old_var
        self.target_critic1.k = self.critic_model1.k
        # Update weights
        self.soft_update(self.target_critic2.trainable_variables, self.critic_model2.trainable_variables)
        # # Update cov, etc.
        self.target_critic2.mean = self.critic_model2.mean
        self.target_critic2.var = self.critic_model2.var
        self.target_critic2.old_var = self.critic_model2.old_var
        self.target_critic2.k = self.critic_model2.k

    #@tf.function
    def train_actor(self, states):
        # Use Critic 1
        with tf.GradientTape() as tape:
            pred_actions = self.actor_model(states, training=True)
            pred_state_action = tf.keras.layers.Concatenate()([states, pred_actions])
            q_value, q_value_std, _  = self.critic_model1(pred_state_action, training=False)
            loss = -tf.math.reduce_mean(q_value+5.0*q_value_std)

        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

    def update_prior(self):
        momentum = 0.99
        # Critic 1
        self.critic_model1.model_prior = (1 - momentum) * self.critic_model1.gp.prior \
                                            + momentum * self.critic_model1.model_prior
        self.critic_model1.gp.update_cov(self.critic_model1.model_prior)
        # Critic 2
        self.critic_model2.model_prior = (1 - momentum) * self.critic_model2.gp.prior \
                                            + momentum * self.critic_model2.model_prior
        self.critic_model2.gp.update_cov(self.critic_model2.model_prior)

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
        if self.ntrain_calls % self.actor_update_freq == 0:
            self.soft_update(self.target_actor.variables, self.actor_model.variables)
        if self.ntrain_calls % self.critic_update_freq == 0:
            self.soft_update_dgpa()
        if self.ntrain_calls % 1000 == 0:
            self.update_prior()

    def action(self, state, train=True, monitoring=False):
        monitoring = True
        """ Method used to provide the next action using the target model """
        state = np.expand_dims(state, 0)

        self.nactions = self.nactions + 1
        # TD3 version
        if self.buffer_counter < self.min_buffer_counter:
            action = self.env.action_space.sample()
            action = np.expand_dims(action, 0)
            noise = tf.zeros(action.shape)
        else:
            action = self.actor_model(state, training=False)
            noise = tf.zeros(action.shape)

            if monitoring:
                for i in range(self.num_actions):
                    tf.summary.scalar('Action_{} mean'.format(i), data=action[0][i],
                                      step=int(self.nactions))

        legal_action = np.clip(action, self.lower_bound, self.upper_bound)
        val_state_action = tf.keras.layers.Concatenate()([state, legal_action])
        # Critic 1
        q_value1, q_value_std1, _ = self.critic_model1(val_state_action, training=False)
        q_value1 = np.squeeze(q_value1)
        q_value_std1 = np.squeeze(q_value_std1)
        tf.summary.scalar('Q-value 1 mean', data=q_value1, step=int(self.nactions))
        tf.summary.scalar('Q-value 1 std', data=q_value_std1, step=int(self.nactions))
        if q_value1!=0:
            tf.summary.scalar('Q-value 1 rel', data=tf.math.abs(q_value_std1/q_value1), step=int(self.nactions))
        # Critic 2
        q_value2, q_value_std2, _ = self.critic_model2(val_state_action, training=False)
        q_value2 = np.squeeze(q_value2)
        q_value_std2 = np.squeeze(q_value_std2)
        tf.summary.scalar('Q-value 2 mean', data=q_value2, step=int(self.nactions))
        tf.summary.scalar('Q-value 2 std', data=q_value_std2, step=int(self.nactions))
        if q_value2!=0:
            tf.summary.scalar('Q-value 2 rel', data=tf.math.abs(q_value_std2/q_value2), step=int(self.nactions))

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

    def get_critic(self):
        rff=256
        model = KerasCriticDGPA(hidden_size=self.hidden_size,
                                num_inputs=(self.num_states+self.num_actions),
                                num_outputs=1,
                                fourier_dim=rff
        )
        return model

    # def get_actor(self):
    #
    #     last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    #     # Input
    #     inputs = tf.keras.layers.Input(shape=(self.num_states,))
    #
    #     # Layer 1
    #     out = tf.keras.layers.Dense(self.hidden_size,
    #                                 kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
    #                                 bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))(inputs)
    #     out = tf.keras.layers.Activation(tf.nn.leaky_relu)(out)
    #
    #     # Layer 2
    #     out = tf.keras.layers.Dense(self.hidden_size,
    #                                 kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
    #                                 bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))(out)
    #     out = tf.keras.layers.Activation(tf.nn.leaky_relu)(out)
    #
    #     # Output
    #     outputs = tf.keras.layers.Dense(self.num_actions, activation="tanh",
    #                                     kernel_initializer=last_init,
    #                                     use_bias=True)(out)
    #
    #     # Rescale for tanh [-1,1]
    #     outputs = tf.keras.layers.Lambda(
    #         lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(outputs)
    #
    #     model = tf.keras.Model(inputs, outputs)
    #     return model

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        self.actor_model = KerasTD3.get_actor(self)
        self.target_actor = KerasTD3.get_actor(self)
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_actor.summary()

        # Critics
        self.critic_model1 = self.get_critic()
        self.target_critic1 = self.get_critic()
        self.target_critic1.set_weights(self.critic_model1.get_weights())
        self.target_critic1.set_variables(self.critic_model1.mean,self.critic_model1.var,
                                          self.critic_model1.old_var, self.critic_model1.k)

        self.critic_model2 = self.get_critic()
        self.target_critic2 = self.get_critic()
        self.target_critic2.set_weights(self.critic_model2.get_weights())
        self.target_critic2.set_variables(self.critic_model2.mean,self.critic_model2.var,
                                          self.critic_model2.old_var, self.critic_model2.k)
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
