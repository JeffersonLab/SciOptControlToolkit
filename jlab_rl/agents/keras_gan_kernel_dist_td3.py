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
from jlab_rl.models.state_generator import Generator_v3 as Generator
from jlab_rl.agents.keras_td3 import KerasTD3
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import time

class KerasKernelDistGenerativeTD3(KerasTD3):
    """ Define all key variables required for all agent """

    def __init__(self, env, warmup_size, nrff=0, logdir=None, model_load_path=None, model_save_path=None, dynamic_ref=True, **kwargs):
        """ Define all key variables required for all agent """

        # Standard TD3 setup
        self.nactor_layers = 5
        self.ncritic_layers = 2
        self.hidden_size = 128
        self.batch_size = 128
        self.ntrain_actor_calls = 0

        # Get env info
        super().__init__(env, warmup_size, nrff, logdir, model_load_path, model_save_path, **kwargs)
        print('Running KerasGenerativeTD3 __init__')

        # Used for random samples
        self.rdm_intputs = 77
        self.norm_sdt = 2.0

        self.max_size = np.max([self.batch_size, self.min_buffer_counter])
        self.nmatrix = self.batch_size*self.batch_size
        print('max_size:', self.max_size)

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
            td_loss, rsad_loss = self.train_actor(state_batch, train_tde=(self.buffer_counter >= self.max_size))
            tf.summary.scalar('Actor TD-error Loss (Opt)', data=td_loss, step=int(self.ntrain_actor_calls))
            tf.summary.scalar('Actor Distance Loss (Disperse)', data=rsad_loss, step=int(self.ntrain_actor_calls))

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

    def train_actor(self, states, train_tde):
        next_rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32,
                                         seed=time.time_ns())
        with tf.GradientTape() as tape:
            training_actions = self.actor_model([states, next_rdm_gaus], training=True)
            training_actions = tf.squeeze(training_actions)
            training_actions = tf.clip_by_value(training_actions, self.lower_bound, self.upper_bound)
            # Calculate the original TD3 loss
            td_loss = 0
            if train_tde:
                q_values = self.critic_model1([states, training_actions], training=False)
                td_loss = -tf.math.reduce_mean(q_values)/10.0
            #

            # Calculate the distance between rewards
            # action_diff = tf.math.squared_difference(tf.expand_dims(training_actions, axis=2),
            #                                             tf.expand_dims(training_actions, axis=1))
            # action_rbf = tf.exp(-action_diff)


            # # Flatten action space
            a1 = tf.reshape(training_actions[:, 0], [-1])
            # Calculate the distance between actions
            all_diff1 = tf.math.squared_difference(tf.expand_dims(a1, axis=1),
                                                   tf.expand_dims(a1, axis=0))
            #inv_diff1 = 1.0/(all_diff1+1e-7)
            #all_loss1 = tf.math.reduce_sum(inv_diff1)/self.nmatrix
            #all_loss1 = -tf.math.reduce_sum(all_diff1)/self.nmatrix
            # nnzeros1 = tf.math.count_nonzero(all_diff1)
            # nzeros1 = self.nmatrix-nnzeros1-self.batch_size
            # all_loss1 = 1/rsad1 if rsad1 > 0 else 9e+8
            # all_loss1 += 2.0*tf.cast(nzeros1, tf.float32)
            # RBF
            # rbf_diff1 = tf.exp(-all_diff1)/self.nmatrix
            # all_loss1 = tf.math.reduce_sum(rbf_diff1)
            # all_loss1 += tf.cast(nzeros1, tf.float32)

            # if nzeros1>0:
            #     print(f'nmatrix1: {self.nmatrix} nnzeros1: {nnzeros1} - nzeros1: {nzeros1}- loss1: {all_loss1}')
            # rsad1 = tf.math.reduce_sum(all_diff1)/self.nmatrix
            # all_loss1 = 1/rsad1 if rsad1 > 0 else 9e+8
            # all_loss1 += tf.cast(nzeros1, tf.float32)
            #all_loss1 = -tf.math.reduce_sum(all_diff1)


            a2 = tf.reshape(training_actions[:, 1], [-1])
            # Calculate the distance between actions
            all_diff2 = tf.math.squared_difference(tf.expand_dims(a2, axis=1),
                                                   tf.expand_dims(a2, axis=0))
            #inv_diff2 = 1.0 / (all_diff2 + 1e-7)

            all_diff = all_diff1 + all_diff2
            inv_diff = 1.0 / (all_diff + 1e-7)
            rs_inv_diff = tf.math.reduce_sum(inv_diff) / self.nmatrix / self.batch_size
            #all_loss2 = tf.math.reduce_sum(inv_diff2) / self.nmatrix
            # all_diff2 = 1.0/(all_diff2+1e-7)
            #all_loss2 = tf.math.reduce_sum(1.0/(all_diff2+1e-7))/self.nmatrix
            #all_loss2 = -tf.math.reduce_sum(all_diff2)/self.nmatrix

            # nnzeros2 = tf.math.count_nonzero(all_diff2)
            # nzeros2 = tf.cast(self.nmatrix-nnzeros2-self.batch_size,tf.float32)
            # # RBF
            # # rbf_diff2 = tf.exp(all_diff2)/self.nmatrix
            # # all_loss2 = tf.math.reduce_sum(rbf_diff2)
            # # all_loss2 += tf.cast(nzeros2, tf.float32)
            # OLD WAY --
            # rsad2 = tf.math.reduce_sum(all_diff2)/self.nmatrix
            #
            # all_loss2 = 1/rsad2 if rsad2 > 0 else 9e+8
            # all_loss2 += 2.0*tf.cast(nzeros1, tf.float32)
            #all_loss2 += tf.abs(tf.random.normal(nzeros2, 1.0))
            # if nzeros2>0:
            #     print(f'all_diff2: {all_diff2.shape} - nzeros2: {nzeros2} - loss2: {all_loss2}')


            #all_loss = inv_diff
            # # Sum all per
            # rbf_diff2 = tf.exp(-all_diff2)
            # rsad_loss2 = tf.math.reduce_sum(rbf_diff2)
            # all_diff = all_diff1 + all_diff2
            # matrix_id = tf.eye(all_diff1.shape[0])
            # all_rbf = tf.exp(-all_diff) - matrix_id
            #rsad_loss = rsad_loss1 + rsad_loss2 #tf.math.reduce_sum(all_rbf) #(rsad_loss1 + rsad_loss2)#/self.batch_size
            # Calculate the loss
            # 1) td_loss: the same as in TD3
            # 2) sad_loss: reduce sum of all the distance between action --> encourage broader possible actions
            total_loss = rs_inv_diff + td_loss

        gradient = tape.gradient(total_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
        return td_loss, rs_inv_diff

    #@tf.function
    # def train_actor(self, states):
    #
    #     td_losses, rsad_losses = [], []
    #     nrepeats = 10
    #     for state in (states):
    #         states_per_action = tf.repeat([state], nrepeats, axis=0)
    #         #print(f'states_per_action: {states_per_action}')
    #         with tf.GradientTape() as tape:
    #             next_rdm_gaus = tf.random.normal([states_per_action.shape[0], self.rdm_intputs], 0, self.norm_sdt,
    #                                              tf.float32, seed=time.time_ns())
    #             training_actions = self.actor_model([states_per_action, next_rdm_gaus], training=True)
    #             training_actions = tf.squeeze(training_actions)
    #
    #             # Maximize difference
    #             all_diff = tf.math.squared_difference(tf.expand_dims(training_actions, axis=2),
    #                                                   tf.expand_dims(training_actions, axis=1))
    #             # print(f'all_diff: {all_diff.shape}')
    #             # print(f'all_diff: {all_diff}')
    #             rbf_all_df = tf.exp(-all_diff)
    #             #print(f'all_diff: {all_diff}')
    #             rsad_loss = tf.math.reduce_sum(rbf_all_df)
    #             total_loss = rsad_loss
    #
    #         gradient = tape.gradient(total_loss, self.actor_model.trainable_variables)
    #         self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
    #     return rsad_loss, rsad_loss

    # def action_inference(self, inference_states):
    #     actions, rewards = [], []
    #     for state in inference_states:
    #         self.env.reset()
    #         nrepeats = 5000
    #         states = tf.repeat([state], nrepeats, axis=0)
    #         rdm_norms = tf.random.normal([nrepeats, self.rdm_intputs], 0, self.norm_sdt, tf.float32)
    #         sampled_actions = self.actor_model.predict_on_batch([states, rdm_norms])
    #         new_q1 = self.target_critic1([states, sampled_actions])
    #         new_q2 = self.target_critic2([states, sampled_actions])
    #         q_mean = tf.math.minimum(new_q1, new_q2)
    #         max_q_idx = tf.argmax(q_mean)
    #         sampled_action = sampled_actions[max_q_idx]
    #         _, reward, _, _, _ = self.env.step(sampled_action)
    #         actions.append(sampled_action)
    #         rewards.append(reward)
    #     return np.squeeze(actions), np.squeeze(rewards)
    #     #     _, reward, _, _, _ = self.env.step(a)
    #     #     rewards.append( reward )
    #     # return np.squeeze(actions), np.squeeze(rewards)
    #     # rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32, seed=time.time_ns())
    #     # actions = self.actor_model([states, rdm_gaus])
    #     # rewards = []
    #     # for a in actions:
    #     #     self.env.reset()
    #     #     _, reward, _, _, _ = self.env.step(a)
    #     #     rewards.append( reward )
    #     # return np.squeeze(actions), np.squeeze(rewards)

    def action_inference(self, states):
        rdm_norms = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32)
        sampled_actions = self.actor_model([states, rdm_norms])
        # print(f'inf shapes: {states.shape}, {rdm_norms.shape}, {sampled_actions.shape}')
        # print(f'inf act: {sampled_actions}')

        #rdm_gaus = tf.random.normal([states.shape[0], self.rdm_intputs], 0, self.norm_sdt, tf.float32, seed=time.time_ns())
        #actions = self.actor_model([states, rdm_gaus])
        rewards = []
        for a in sampled_actions:
            self.env.reset()
            _, reward, _, _, _ = self.env.step(a)
            rewards.append( reward )
        return np.squeeze(sampled_actions), np.squeeze(rewards)

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
            # print(f'shapes: {states.shape}, {rdm_norms.shape}, {sampled_actions.shape}')
            # print(f' act: {sampled_actions}')
            # new_q1 = self.target_critic1([states, sampled_actions])
            # new_q2 = self.target_critic2([states, sampled_actions])
            # q_mean = tf.math.minimum(new_q1, new_q2)
            # max_q_idx = tf.argmax(q_mean)
            sampled_action = sampled_actions[0]

        #print(f'selected act: {sampled_action}')
        sampled_action = sampled_action.flatten()
        assert sampled_action.shape == self.num_actions or sampled_action.shape == (self.num_actions,), "Sampled action shape is incorrect..."

        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))
            else:
                tf.summary.scalar('Action #0', data=sampled_action, step=int(self.nactions))

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
