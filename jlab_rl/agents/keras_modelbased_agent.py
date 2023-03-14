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
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from os.path import join
from tqdm import tqdm
import time

class KerasGenericModelBasedAgent(jlab_rl.Agent):

    def __init__(self, env, warmup_size=1000, logdir=None, model_load_path=None, model_save_path=None, **kwargs):
        """ Define all key variables required for all agent """

        # Get env info
        super().__init__(**kwargs)
        print('Running KerasTD3 __init__')
        self.env = env
        self.nsteps = env._max_episode_steps
        print('Env steps:', self.nsteps)
        self.model_load_path = model_load_path
        self.model_save_path = model_save_path
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low
        print('upper_bound: ',self.upper_bound)
        print('lower_bound: ',self.lower_bound)
        self.action_width = (self.upper_bound+self.lower_bound)/2.0

        # Buffer
        self.batch_size = 1024
        self.min_buffer_counter = warmup_size#self.batch_size
        self.buffer_counter = 0
        self.buffer_capacity = 5000000
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.per_buffer = np.ones((self.buffer_capacity, 1))

        # Used to update target networks
        self.tau = 0.5#0.005
        self.gamma = 0.99

        # Setup Optimizers

        actor_lr = 3e-4
        self.actor_optimizer = Adam(actor_lr, epsilon=1e-08)

        self.hidden_size = 256
        self.layer_std = 1.0 / np.sqrt(self.num_actions)

        self.initialize_new_models()
        dynamic_lr = 3e-4
        self.dynamic_opt = Adam(dynamic_lr, epsilon=1e-08)
        self.dynamic_model_es = tf.keras.callbacks.EarlyStopping(monitor="loss")
        self.dynamic_model_rl = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss")
        self.dynamic_model_callbacks = [self.dynamic_model_es, self.dynamic_model_rl]
        self.dynamic_model.compile(self.dynamic_opt, loss="MSE")

        # Load models for retraining
        if model_load_path is not None:
            self.load()

        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq=2
        self.critic_update_freq=2

        try:
            os.mkdir(logdir)
        except OSError as error:
            print(error)
        file_writer = tf.summary.create_file_writer(logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = tf.Variable(0)
        self.nres = tf.Variable(0)

    def train_dynamic_model(self, states, actions, rewards, next_states):
        #print('Training dynamic model...')
        history = self.dynamic_model.fit(x=[states, actions], y=[next_states, rewards],
                                         callbacks = self.dynamic_model_callbacks,
                                         epochs=10, batch_size=self.batch_size, shuffle=True, verbose=0)
        # history = self.dynamic_model.fit(x=[self.state_buffer, self.action_buffer],
        #                                  y=[self.next_state_buffer, self.reward_buffer], epochs=250, verbose=0)
        #print("\nDynamic Model Loss: ", history.history['loss'][-1],"\n")
        tf.summary.scalar('Dynamic Model Loss', data=history.history['loss'][-1], step=int(self.ntrain_calls))

        # Plot results
        tf.summary.scalar('Dynamic Model Loss', data=history.history['loss'][-1], step=int(self.ntrain_calls))
        ns_pred, r_pred = self.dynamic_model([states, actions])
        # print(ns_pred.shape)
        # print(next_states.shape)
        tf.summary.histogram("Dynamic Model Reward Residual", r_pred-rewards, step=int(self.ntrain_calls))
        for s in range(ns_pred.shape[1]):
            tf.summary.histogram("Dynamic Model State {} Residual".format(s),
                                 (ns_pred.numpy())[:,s]-(next_states.numpy())[:,s], step=int(self.ntrain_calls))
        # for j in range(r_pred.shape[0]):
        #
        #     print(r_pred[j])
        #     resi = tf.cast(r_pred[j]-rewards[j], tf.float32)
        #     print(resi)
        #     tf.summary.scalar('Dynamic Model Reward Residual', data=resi, step=int(self.nres))
        #     self.nres+=1
        return history

    def train_actor(self, states):
        # Do rollout
        idxs = tf.range(tf.shape(states)[0])
        ridxs = tf.random.shuffle(idxs)[:128]
        states = tf.gather(states, ridxs)
        total_rewards = tf.zeros(shape=(states.shape[0],1))
        with tf.GradientTape() as tape:
            for step in range(25):# tried 25 <-140>, 100 <1200>
                actions = self.actor_model(states, training=True)
                next_s_preds, reward_preds = self.dynamic_model([states, actions])
                states = next_s_preds
                total_rewards = tf.add(total_rewards, reward_preds)
            loss = -tf.math.reduce_mean(total_rewards)
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
        tf.summary.scalar('Actor Model Loss', data=loss, step=int(self.ntrain_calls))

    # def train_actor(self, states):
    #     total_rewards = []
    #     for s in tqdm(states, desc='Training policy model - States'):
    #         state = tf.expand_dims(s, axis=0)
    #         total_reward = 0
    #         # Do rollout
    #         nsteps = 10 # env._max_episode_steps
    #         with tf.GradientTape() as tape:
    #             for step in range(nsteps):
    #                 action = self.actor_model(state, training=True)
    #                 #print('action:{}'.format(action))
    #                 next_s_pred, reward_pred = self.dynamic_model([state, action])
    #                 #print('next_s_pred, reward_pred:{}/{}'.format(next_s_pred, reward_pred))
    #                 state = next_s_pred
    #                 total_reward += reward_pred
    #             #total_rewards.append(total_reward)
    #             #print('total_reward:', total_reward)
    #             loss = -tf.math.reduce_mean(total_reward)
    #             #print('loss:', loss)
    #         gradient = tape.gradient(loss, self.actor_model.trainable_variables)
    #         self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
    #         tf.summary.histogram("Actor Model Loss".format(s), loss, step=int(self.ntrain_calls))
    #         tf.summary.scalar('Actor Model Loss', data=loss, step=int(self.ntrain_calls))

    def get_dynamic_model(self):

        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states))
        #state_input = tf.keras.layers.Dense(200, activation="tanh")(state_input)
        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))
        #action_input = tf.keras.layers.Dense(200, activation="tanh")(action_input)
        state_action = tf.keras.layers.Concatenate()([state_input, action_input])
        state_action1 = tf.keras.layers.Dense(200, activation="tanh")(state_action)
        state_action2 = tf.keras.layers.Dense(200, activation="tanh")(state_action1)
        state_action3 = tf.keras.layers.Dense(100, activation="tanh")(state_action2)
        # state_action1 = tf.keras.layers.Dense(self.hidden_size, activation="relu")(state_action)
        # state_action2 = tf.keras.layers.Dense(self.hidden_size, activation="relu")(state_action1)
        # Predict the next state and the reward
        next_states = tf.keras.layers.Dense(self.num_states, activation="linear")(state_action3)
        reward = tf.keras.layers.Dense(1, activation="linear")(state_action3)
        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], [next_states, reward])

        return model

    def get_actor(self):

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # Input
        inputs = tf.keras.layers.Input(shape=(self.num_states,))

        # Layer 1
        out = tf.keras.layers.Dense(self.hidden_size,
                                    kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
                                    bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))(inputs)
        out = tf.keras.layers.Activation(tf.nn.leaky_relu)(out)

        # Layer 2
        out = tf.keras.layers.Dense(self.hidden_size,
                                    kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
                                    bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))(out)
        out = tf.keras.layers.Activation(tf.nn.leaky_relu)(out)

        # Output
        outputs = tf.keras.layers.Dense(self.num_actions, activation="tanh",
                                        kernel_initializer=last_init,
                                        use_bias=True)(out)

        # Rescale for tanh [-1,1]
        outputs = tf.keras.layers.Lambda(
            lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(outputs)

        model = tf.keras.Model(inputs, outputs)
        return model

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # if self.buffer_counter % self.batch_size == 0:
        self.ntrain_calls += 1
        if self.buffer_counter % 2 == 0:
            self.train_dynamic_model(state_batch, action_batch, reward_batch, next_state_batch)
        #if self.buffer_counter > self.min_buffer_counter + 5*self.batch_size:
        #if self.buffer_counter > self.min_buffer_counter + 2*self.batch_size:
        #if self.buffer_counter % 20 == 0 and self.buffer_counter > 5000:
        if self.buffer_counter % 2 == 0 and self.buffer_counter > 4*self.batch_size:
            self.train_actor(state_batch)
        # if self.ntrain_calls%self.actor_update_freq == 0:
        #     self.soft_update(self.target_actor.variables, self.actor_model.variables)

    def train(self):
        """ Method used to train """
#        self.ntrain_calls += 1

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

        # Train dynamic model and actor
        if self.buffer_counter > self.batch_size:
            self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def action(self, state, train=True):
        """ Method used to provide the next action using the target model """
        state = np.expand_dims(state, 0)

        if train==False:
            sampled_action = self.actor_model.predict_on_batch(state)
            noise = tf.zeros(sampled_action.shape)
            legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
            return [np.squeeze(legal_action)], [np.squeeze(noise)]
        self.nactions.assign(self.nactions + 1)
        # TD3 version
        if self.buffer_counter < 5  * self.batch_size + self.min_buffer_counter:
        #if self.buffer_counter < self.min_buffer_counter:
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
        else:
            sampled_action = self.actor_model.predict_on_batch(state)
            noise = np.random.normal(0, 0.1, self.num_actions)
        # if train==True:
        #     sampled_action = self.env.action_space.sample()
        #     # print('env sample: ', sampled_action.shape)
        #     noise = np.zeros(self.num_actions)
        # else:
        #     sampled_action = self.actor_model.predict_on_batch(state)
        #     noise = np.random.normal(0, 0.1, self.num_actions)
        sampled_action = np.squeeze(sampled_action)
        # sampled_action = np.expand_dims(sampled_action, axis=0)
        # print('env sample 2: ', sampled_action.shape)

        # if self.num_actions == 1:
        #     sampled_action = np.expand_dims(sampled_action, axis=0)
        #     sampled_action = sampled_action.reshape(-1,1)
        # print(sampled_action.shape)

        for i in range(self.num_actions):
            if self.num_actions > 1:
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))
        if train == True:
            sampled_action = sampled_action + noise

        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)], [np.squeeze(noise)]

    def memory(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1
        #print('buffer_counter:', self.buffer_counter)

    def load(self):
        """ Load the ML models """
        try:
            self.actor_model.load_weights(join(self.model_load_path, "actor_model.h5"))
            self.target_actor.load_weights(join(self.model_load_path, "target_actor.h5"))
            self.dynamic_model.load_weights(join(self.model_load_path, "dynamic_model.h5"))
        except:
            print("Error while loading models, initializing new models...")

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        print('Running KerasTD3 initialize_new_models()')

        # Policy model
        self.actor_model = self.get_actor()
        self.target_actor = self.get_actor()
        self.target_actor.set_weights(self.actor_model.get_weights())

        # Dynamic model
        self.dynamic_model = self.get_dynamic_model()

    def save(self):
        """ Save the ML models """
        try:
            self.actor_model.save_weights(join(self.model_save_path, "actor_model.h5"))
            self.target_actor.save_weights(join(self.model_save_path, "target_actor.h5"))
            self.dynamic_model.save_weights(join(self.model_save_path, "dynamic_model.h5"))
        except:
            print("Error in saving the models...")

