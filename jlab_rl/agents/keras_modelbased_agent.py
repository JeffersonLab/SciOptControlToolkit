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

from jlab_rl.models.dynamic_model import DynamicModel


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
        print('upper_bound: ', self.upper_bound)
        print('lower_bound: ', self.lower_bound)
        self.action_width = (self.upper_bound+self.lower_bound)/2.0

        # Buffer
        self.batch_size = 1024
        self.min_buffer_counter = warmup_size
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
        self.nsamples = 5
        dynamic_lr = 3e-4
        self.dynamic_opt = Adam(dynamic_lr, epsilon=1e-08)
        self.dynamic_model_es = tf.keras.callbacks.EarlyStopping(monitor="loss")
        self.dynamic_model_rl = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss")
        self.dynamic_model_callbacks = [self.dynamic_model_es, self.dynamic_model_rl]
        self.dynamic_model.compile(self.dynamic_opt, loss="mse", metrics='loss')

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
        self.policy_training_started = False

    def train_dynamic_model(self, states, actions, rewards, next_states):
        state_actions = tf.keras.layers.Concatenate(axis=1)([states, actions])
        new_state_rewards = tf.keras.layers.Concatenate(axis=1)([next_states, rewards])
        history = self.dynamic_model.fit(x=state_actions, y=new_state_rewards,
                                         #callbacks = self.dynamic_model_callbacks,
                                         epochs=10, batch_size=self.batch_size, shuffle=True, verbose=0)
        print("Dynamic Model Loss: {}".format(history.history['loss'][-1]))
        tf.summary.scalar('Dynamic Model Loss', data=history.history['loss'][-1], step=int(self.ntrain_calls))

        # Plot results
        #tf.summary.scalar('Dynamic Model Loss', data=history.history['loss'][-1], step=int(self.ntrain_calls))
        #ns_pred, r_pred = self.dynamic_model([states, actions])
        pred_means, pred_std = self.dynamic_model.predict_uq(state_actions)
        ns_pred = pred_means[:,:-1]
        ns_pred_std = pred_std[:,:-1]
        r_pred = pred_means[:,-1]
        r_pred_std = pred_std[:,-1]
        tf.summary.histogram("Dynamic Model Reward Residual", r_pred-rewards, step=int(self.ntrain_calls))
        tf.summary.histogram("Dynamic Model Reward Error", r_pred_std, step=int(self.ntrain_calls))
        tf.summary.scalar('Dynamic Model Dropout', data=self.dynamic_model.drop_percent, step=int(self.ntrain_calls))

        for s in range(ns_pred.shape[1]):
            tf.summary.histogram("Dynamic Model State {} Residual".format(s),
                                 (ns_pred.numpy())[:,s]-(next_states.numpy())[:,s], step=int(self.ntrain_calls))
            tf.summary.histogram("Dynamic Model State {} Error".format(s),
                                 (ns_pred_std.numpy())[:,s], step=int(self.ntrain_calls))
        return history

    def train_actor(self, states):
        self.policy_training_started = True
        # Do rollout
        idxs = tf.range(tf.shape(states)[0])
        ridxs = tf.random.shuffle(idxs)[:128]
        states = tf.gather(states, ridxs)
        total_rewards = tf.zeros(shape=(states.shape[0],1))
        with tf.GradientTape() as tape:
            for step in range(10):# tried 25 <-140>, 100 <1200>
                actions = self.actor_model(states, training=True)
                state_actions = tf.keras.layers.Concatenate(axis=1)([states, actions])
                # No UQ
                predictions = self.dynamic_model(state_actions)
                next_s_preds = predictions[:,:-1]
                reward_preds = predictions[:,-1]
                total_rewards = tf.add(total_rewards, reward_preds)
                # W/ UQ
                # next_s_preds, _,  reward_preds, reward_pred_stds = self.dynamic_model.predict_uq([states, actions])
                # total_rewards = tf.add(total_rewards, reward_preds-reward_pred_stds)
                states = next_s_preds
            loss = -tf.math.reduce_mean(total_rewards)
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))
        tf.summary.scalar('Actor Model Loss', data=loss, step=int(self.ntrain_calls))

    def get_dynamic_model(self):
        model = DynamicModel(ndlayers=5, hidden_size=128, drop_value=0.001, output_size=self.num_states+1, nrff=128)
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
        self.train_dynamic_model(state_batch, action_batch, reward_batch, next_state_batch)
        if self.buffer_counter % 2 == 0:
            self.train_actor(state_batch)

    def train(self):
        """ Method used to train """
#        self.ntrain_calls += 1
        if self.buffer_counter>0:
            # Get sampling range
            record_range = min(self.buffer_counter, self.buffer_capacity)

            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)

            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
            action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
            reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)

            # Train dynamic model and actor
            if self.buffer_counter > self.batch_size:
                self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def run_dynamic_model_episode(self, env, nsteps):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        total_reward = 0
        for step in range(nsteps):
            action = self.actor_model(state)
            state_action = tf.keras.layers.Concatenate(axis=1)([state, action])
            predictions = self.dynamic_model(state_action)
            next_s_pred, reward_pred = predictions[:,:-1], predictions[:,-1]
            state = next_s_pred
            total_reward += tf.squeeze(reward_pred)
        #print("MBRL Dynamic Episodic Reward is ==> {}".format(total_reward))
        return total_reward

    def run_env_episode(self, env, nsteps):
        # theta, thetadot = env.state
        # intial_state = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        # intial_state = np.expand_dims(intial_state, axis=0)
        #print(intial_state.shape)
        # action = env.action_space.sample()
        # print(action.shape)
        # state = env.state()
        # print('Initial state: {}'.format(init_state))
        #
        # state = np.expand_dims(env.observation_space.sample(), axis=0)#env.state#initial_state
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        #state = tf.convert_to_tensor(state)
        total_reward = 0
        for _ in range(nsteps):
            #print('state shape {}'.format(state.shape))
            action = self.actor_model(state)
            action = np.reshape(action, -1)
            next_state, reward, done_old, done, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            state = next_state
            total_reward += float(reward)
        #print('Test total reward:{}'.format(total_reward))
        return total_reward

    def action(self, state, train=True, random_only=False):
        """ Method used to provide the next action using the target model """
        state = np.expand_dims(state, 0)
        sampled_action = self.actor_model.predict_on_batch(state)
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        noise = np.zeros(self.num_actions)
        return [np.squeeze(legal_action)], [noise]

        # if random_only:
        #     sampled_action = self.env.action_space.sample()
        #     noise = np.zeros(self.num_actions)
        #     return [np.squeeze(sampled_action)], [np.squeeze(noise)]
        #
        # if train==False:
        #     sampled_action = self.actor_model.predict_on_batch(state)
        #     noise = tf.zeros(sampled_action.shape)
        #     legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        #     return [np.squeeze(legal_action)], [np.squeeze(noise)]
        # self.nactions.assign(self.nactions + 1)
        # # TD3 version
        # if self.buffer_counter < 5  * self.batch_size + self.min_buffer_counter:
        # #if self.buffer_counter < self.min_buffer_counter:
        #     sampled_action = self.env.action_space.sample()
        #     noise = np.zeros(self.num_actions)
        # else:
        #     sampled_action = self.actor_model.predict_on_batch(state)
        #     noise = np.random.normal(0, 0.1, self.num_actions)
        # # if train==True:
        # #     sampled_action = self.env.action_space.sample()
        # #     # print('env sample: ', sampled_action.shape)
        # #     noise = np.zeros(self.num_actions)
        # # else:
        # #     sampled_action = self.actor_model.predict_on_batch(state)
        # #     noise = np.random.normal(0, 0.1, self.num_actions)
        # sampled_action = np.squeeze(sampled_action)
        # # sampled_action = np.expand_dims(sampled_action, axis=0)
        # # print('env sample 2: ', sampled_action.shape)
        #
        # # if self.num_actions == 1:
        # #     sampled_action = np.expand_dims(sampled_action, axis=0)
        # #     sampled_action = sampled_action.reshape(-1,1)
        # # print(sampled_action.shape)
        #
        # for i in range(self.num_actions):
        #     if self.num_actions > 1:
        #         tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))
        # if train == True:
        #     sampled_action = sampled_action + noise
        #
        # legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        # return [np.squeeze(legal_action)], [np.squeeze(noise)]

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

