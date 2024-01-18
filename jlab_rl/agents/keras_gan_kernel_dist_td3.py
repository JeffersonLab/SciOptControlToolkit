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

import tensorflow as tf
from jlab_rl.models.state_generator import Generator_v3 as Generator
from jlab_rl.agents.keras_td3 import KerasTD3
from tensorflow.keras.optimizers.legacy import Adam, RMSprop
import numpy as np
import time

class GCRMSprop(RMSprop):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= ops.mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

class KerasKernelDistGenerativeTD3(KerasTD3):
    """ Define all key variables required for all agent """

    def __init__(self, env, warmup_size, nrff=0, logdir=None, model_load_path=None, model_save_path=None, dynamic_ref=True, **kwargs):
        """ Define all key variables required for all agent """

        self.ntrain_actor_calls = 0
        self.nactor_layers = 2
        self.ncritic_layers = 3

        # Get env info
        super().__init__(env, warmup_size, nrff, logdir, model_load_path, model_save_path, **kwargs)
        print('Running KerasKernelGenerativeTD3 __init__')

        # Standard TD3 setup
        self.hidden_size = 256
        self.batch_size = 128# 512

        # Used for random samples
        self.rdm_intputs = 77
        self.norm_sdt = 1.0

        self.max_size = np.max([self.batch_size, self.min_buffer_counter])
        self.nmatrix = self.batch_size*self.batch_size
        print('max_size:', self.max_size)

        self.critic_optimizer1 = GCRMSprop(learning_rate=self.critic_lr)
        self.critic_optimizer2 = GCRMSprop(learning_rate=self.critic_lr)
        self.actor_optimizer = GCRMSprop(learning_rate=self.actor_lr)


        print(f'lower_bound: {self.lower_bound}, {type(self.lower_bound)}')
        self.action_diff_range = tf.math.sqrt(tf.math.squared_difference(self.lower_bound, self.upper_bound))._numpy()
        #self.action_diff_range = np.linalg.norm(self.upper_bound-self.lower_bound)
        print(f'action_diff_range: {self.action_diff_range}, {type(self.action_diff_range)}')
        # Re-init models
        self.initialize_new_models()

    def get_critic(self):

        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states))
        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))
        # Concatenate state + rdm
        state_action = tf.keras.layers.Concatenate()([state_input, action_input])
        # Build layers
        for _ in range(self.ncritic_layers):
            state_action = tf.keras.layers.Dense(self.hidden_size, activation=tf.keras.activations.relu)(state_action)
        outputs = tf.keras.layers.Dense(1, activation='linear')(state_action)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def get_actor(self):
        model = Generator(ndims=self.num_actions, nlayers=self.nactor_layers,
                          lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return model

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # Check the shapes of the training data
        assert state_batch.shape == (self.batch_size, self.num_states), "State_batch shape incorrect in update function" 
        assert action_batch.shape == (self.batch_size, self.num_actions), "action_batch shape incorrect in update function" 
        assert reward_batch.shape == (self.batch_size, 1), "reward_batch shape incorrect in update function" 
        assert next_state_batch.shape == (self.batch_size, self.num_states), "next_state_batch shape incorrect in update function" 
        assert done_batch.shape == (self.batch_size, 1), "done_batch shape incorrect in update function" 

        # Train Critic
        critic_loss1, critic_loss2 = self.train_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        tf.summary.scalar('Critic #1 Loss', data=critic_loss1, step=int(self.buffer_counter))
        tf.summary.scalar('Critic #2 Loss', data=critic_loss2, step=int(self.buffer_counter))

        # Train actor
        if self.buffer_counter >= self.batch_size:

            self.ntrain_actor_calls += 1
            # Train
            td_loss, extra_loss = self.train_actor(state_batch)#(self.buffer_counter >= self.max_size))
            tf.summary.scalar('Actor TD-error Loss', data=td_loss, step=int(self.ntrain_actor_calls))
            tf.summary.scalar('Actor Kernel Loss', data=extra_loss, step=int(self.ntrain_actor_calls))
            tf.summary.scalar('Actor Total Loss', data=(td_loss+extra_loss), step=int(self.ntrain_actor_calls))

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        
        next_rdm_gaus = tf.random.normal([next_states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32, seed=time.time_ns())
        next_actions = self.target_actor([next_states, next_rdm_gaus], training=False)
        
        # Do we need this noise ?
        noises = tf.random.normal(next_actions.shape, 0, 0.2)
        noises = tf.clip_by_value(noises, -0.5, 0.5)
        next_actions = next_actions+noises
        
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

        return critic_loss1, critic_loss2

    #@tf.function
    def train_actor(self, states):
        next_rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32,
                                         seed=time.time_ns())
        with tf.GradientTape() as tape:
            #print(f'states: {states.shape}')
            #print(f'next_rdm_gaus: {next_rdm_gaus.shape}')
            training_actions = self.actor_model([states, next_rdm_gaus], training=True)
            #print(f'training_actions.shape: {training_actions.shape}')
            # training_actions = tf.squeeze(training_actions)
            # print(f'training_actions.shape: {training_actions.shape}')
            training_actions = tf.clip_by_value(training_actions, self.lower_bound, self.upper_bound)
            q_values = self.critic_model1([states, training_actions], training=False)

            # Calculate the original TD3 loss
            td_loss = -tf.math.reduce_mean(q_values)

            # Calculate the q_value combinations
            q_values_comb = tf.math.add(tf.expand_dims(q_values, axis=1), tf.expand_dims(q_values, axis=0))
            # Calculate the action distance combinations
            # Dissipative term - should optimize code
            action_distance_comb = 0
            reduced_q_action = 0
            if self.num_actions == 1:
                total_reshaped_a_sd = tf.math.squared_difference(
                    tf.expand_dims(training_actions, axis=1),
                    tf.expand_dims(training_actions, axis=0))
                action_distance_comb = action_distance_comb / self.action_diff_range[a]
            else:
                for a in range(self.num_actions):
                    ra = tf.reshape(training_actions[:, a], [-1])
                    ra_sd = tf.math.squared_difference(
                        tf.expand_dims(ra, axis=1), tf.expand_dims(ra, axis=0))
                    ra_sd = ra_sd/self.action_diff_range[a]
                    #reduced_dist_action = tf.reduce_mean(ra_sd)
                    #ra_sd = tf.exp(-ra_sd)
                    #ra_sd = 1 - ra_sd + tf.eye(self.batch_size)
                    ra_sd = ra_sd + tf.eye(self.batch_size)
                    q_ra_matrix = tf.multiply(q_values_comb, ra_sd)
                    reduced_q_action = -tf.reduce_mean(q_ra_matrix)
                    print(f'reduced_q_action #{a}: {reduced_q_action}')

                    #print(f'actions distance #{a}: {ra_sd}')
                    #action_distance_comb += reduced_q_action
                    action_distance_comb += reduced_q_action

            #extra_loss = reduced_q_action
            extra_loss = action_distance_comb/self.num_actions
            #print(f'action_distance_comb: {action_distance_comb}')
            #action_distance_comb = action_distance_comb + tf.eye(self.batch_size)
            #print(f'action_distance_comb: {action_distance_comb}')
            #q_action_matrix = tf.multiply(q_values_comb, action_distance_comb)
            #print(f'q_action_matrix: {q_action_matrix[0]}')

            #extra_loss = -tf.math.reduce_mean(q_action_matrix)#/self.batch_size)
            # print(f'extra_loss: {extra_loss}')
            # sys.exit()
            # Add both losses
            #total_loss = td_loss + extra_loss
            total_loss = td_loss + extra_loss
            print(f'total_loss: {total_loss}')
            #total_loss = extra_loss

        gradient = tape.gradient(total_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

        return td_loss, extra_loss

    # def train_actor(self, batch_states):
    #
    #     train_vars = self.actor_model.trainable_variables
    #     accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
    #     for state in batch_states:
    #         state = tf.expand_dims(state, 0)
    #         #print(state.shape)
    #         states = tf.repeat(state, 25, axis=0)
    #         #print(states.shape)
    #         next_rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32,
    #                                          seed=time.time_ns())
    #         #print(next_rdm_gaus.shape)
    #         with tf.GradientTape() as tape:
    #             training_actions = self.actor_model([states, next_rdm_gaus], training=True)
    #             training_actions = tf.squeeze(training_actions)
    #             training_actions = tf.clip_by_value(training_actions, self.lower_bound, self.upper_bound)
    #             q_values = self.critic_model1([states, training_actions], training=False)
    #             # Calculate the original TD3 loss
    #             td_loss = -tf.math.reduce_mean(q_values)
    #             # Distance range
    #             # Dissipative term - should optimize code
    #             #print(f'training_actions: {training_actions}')
    #             total_reshaped_a_sd = 0
    #             if self.num_actions == 1:
    #                 total_reshaped_a_sd = tf.math.squared_difference(
    #                     tf.expand_dims(training_actions, axis=1),
    #                     tf.expand_dims(training_actions, axis=0))
    #             else:
    #                 for a in range(self.num_actions):
    #                     ra = tf.reshape(training_actions[:, a], [-1])
    #                     ra_sd = tf.math.squared_difference(
    #                         tf.expand_dims(ra, axis=1), tf.expand_dims(ra, axis=0))
    #                     #print(f'ra_sd: {ra_sd}')
    #                     ra_sd = tf.abs(tf.sqrt(ra_sd))
    #                     #print(f'ra_sd sqrt: {ra_sd}')
    #                     ra_sd = ra_sd/self.action_diff_range[a]
    #                     #print(f'ra_sd w/ range: {ra_sd}')
    #                     total_reshaped_a_sd += ra_sd
    #
    #             # Normalize for the number of actions
    #             total_reshaped_a_sd = total_reshaped_a_sd/self.num_actions
    #             #print(f'total_reshaped_a_sd / actions: {total_reshaped_a_sd}')
    #             # Normalize for the number of matrix
    #             total_reshaped_a_sd = total_reshaped_a_sd/self.nmatrix
    #             extra_loss = -tf.math.reduce_sum(total_reshaped_a_sd)
    #             #print(f'total_reshaped_a_sd / matrix: {total_reshaped_a_sd}')
    #             #total_reshaped_a_sd += tf.eye(total_reshaped_a_sd.shape[0])
    #             #print(f'total_reshaped_a_sd w/ eye: {total_reshaped_a_sd}')
    #             #print(f'total_reshaped_a_sd: {total_reshaped_a_sd.shape}')
    #             #print(f'q_values: {q_values.shape}')
    #             #matrix_vec = tf.linalg.matvec(total_reshaped_a_sd, tf.squeeze(q_values))
    #             #matrix_vec = tf.expand_dims(matrix_vec,axis=1)
    #             #print(f'matrix_vec: {matrix_vec.shape}')
    #             #extra_loss = -tf.reduce_sum(matrix_vec)
    #             #print(f'extra_loss: {extra_loss.shape}')
    #             #print(f'td_loss: {td_loss.shape}')
    #
    #             #print(f'extra_loss: {extra_loss}')
    #             #sys.exit()
    #             # RBF (length scale depends on the range anf number of actions, and maybe other things)
    #             # length_scale = 1.0*self.num_actions
    #             # rbf = tf.exp(-total_reshaped_a_sd /length_scale)
    #             # rbf = rbf - tf.eye(total_reshaped_a_sd.shape[0])
    #             # #
    #             # # # Normalize based on the problem size
    #             # extra_loss = tf.math.reduce_sum(rbf)/(self.nmatrix-self.batch_size)
    #
    #             # Divide by two since we are double counting the upper and lower part of the matrix
    #             #extra_loss = extra_loss/2.0
    #
    #             # Add both losses
    #             total_loss = td_loss + extra_loss
    #             #total_loss = extra_loss
    #
    #
    #         gradient = tape.gradient(total_loss, self.actor_model.trainable_variables)
    #         accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, gradient)]
    #     self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
    #
    #     return td_loss, extra_loss

    def get_best_qvalue_action(self, states, sampled_actions):
        new_q1 = self.target_critic1([states, sampled_actions])
        new_q2 = self.target_critic2([states, sampled_actions])
        q_mean = tf.math.minimum(new_q1, new_q2)
        max_q_idx = tf.argmax(q_mean)
        return max_q_idx

    def action_inference(self, nrepeats=10000):
        prev_state, _ = self.env.reset()
        prev_state = tf.expand_dims(prev_state, 0)
        states = tf.repeat(prev_state, nrepeats, axis=0)
        rdm_norms = tf.random.normal([nrepeats, self.rdm_intputs], 0, self.norm_sdt, tf.float32)
        actions = self.actor_model.predict_on_batch([states, rdm_norms])
        rewards = []
        for a in actions:
            prev_state, _ = self.env.reset()
            _, reward, _, _, _ = self.env.step(a)
            rewards.append( reward )
        rewards = np.array(rewards)
        return np.squeeze(actions), np.squeeze(rewards)


    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """
        
        assert state.shape == self.num_states, "Shape of the input state to action method is not correct..."

        self.nactions.assign(self.nactions + 1)
        state = tf.expand_dims(state, 0)
        action_type = 0
        # Random samples (not very efficient)
        if self.buffer_counter <= self.max_size:
            sampled_action = self.env.action_space.sample()
        # Use the policy
        else:
            nrepeats = 1
            states = tf.repeat(state, nrepeats, axis=0)
            rdm_norms = tf.random.normal([nrepeats, self.rdm_intputs], 0, self.norm_sdt, tf.float32)
            sampled_actions = self.actor_model.predict_on_batch([states, rdm_norms])
            # Use critic models to steer action selection
            max_id = self.get_best_qvalue_action(states, sampled_actions)
            sampled_actions = sampled_actions[max_id]
            # Apply TD3 noise
            if train:
                noise = tf.random.normal(sampled_actions.shape, 0, 0.1)
                noise = tf.clip_by_value(noise, -0.25, 0.25)
                sampled_actions = sampled_actions + noise.numpy()
            sampled_action = sampled_actions[0]

        sampled_action = sampled_action.flatten()
        assert sampled_action.shape == self.num_actions or sampled_action.shape == (self.num_actions,), "Sampled action shape is incorrect..."

        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))
            else:
                tf.summary.scalar('Action #0', data=sampled_action[0], step=int(self.nactions))

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
        #self.action_type_buffer[index] = obs_tuple[5]
        self.buffer_counter += 1
