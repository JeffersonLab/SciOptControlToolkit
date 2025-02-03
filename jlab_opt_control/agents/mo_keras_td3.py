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

import json
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import platform
import shutil
import sys
import time
from os.path import join

import numpy as np
import tensorflow as tf

import jlab_opt_control as jlab_opt_control
import jlab_opt_control.buffers
import jlab_opt_control.models
import jlab_opt_control.utils.cfg_utils as cfg_utils

processor = platform.processor()

td3_log = logging.getLogger("MO TD3-Agent")
td3_log.setLevel(logging.ERROR)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class GenEnergy(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #        self.input_layer = tf.keras.Input(shape = (None, energies.shape[1]))
        nactions = 197
        self.hidden1 = tf.keras.layers.Dense(2 * nactions, activation='leaky_relu')
        self.hidden2 = tf.keras.layers.Dense(4 * nactions, activation='leaky_relu')
        self.hidden3 = tf.keras.layers.Dense(8 * nactions, activation='leaky_relu')
        self.hidden4 = tf.keras.layers.Dense(4 * nactions, activation='leaky_relu')
        self.hidden5 = tf.keras.layers.Dense(2 * nactions, activation='leaky_relu')
        self.output_layer = tf.keras.layers.Dense(nactions, activation='tanh')

        # self.action_scale = tf.constant((max_action - min_action) / 2, dtype=tf.float32)
        # self.action_bias = tf.constant((max_action + min_action) / 2, dtype=tf.float32)

    def call(self, inputs):
        # x = self.input_layer(energy)
        x, noise = inputs
        x = tf.keras.layers.concatenate([x, noise])
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.output_layer(x)
        return x

class MO_KerasTD3(jlab_opt_control.Agent):

    def __init__(self, env, logdir, buffer_type=None, buffer_size=None, cfg='mo_keras_td3.json'):
        """ Define all key variables required for all agent """

        # Get env info
        self.target_critic2 = None
        self.critic_model2 = None
        self.target_critic1 = None
        self.critic_model1 = None
        self.target_actor = None
        self.actor_model = None
        td3_log.info('Running KerasMOTD3 __init__')

        # Environment setup
        self.env = env
        try:
            assert "Box" in str(type(env.action_space)), 'Invalid action space'
            self.num_states = env.observation_space.shape[0]
            self.num_actions = env.action_space.shape[0]
            self.num_rewards = env.reward_space.shape[0]
            self.upper_bound = env.action_space.high
            self.lower_bound = env.action_space.low
            td3_log.info(f'Action upper bound: {self.upper_bound}')
            td3_log.info(
                f'Action upper bound: {float(env.action_space.high[0])}')
            td3_log.info(f'Action lower bound: {self.lower_bound}')
            td3_log.info(
                f'Action lower bound: {float(env.action_space.low[0])}')
            self.range = self.upper_bound - self.lower_bound
            td3_log.info(f'Action range: {self.range}')
        except:
            td3_log.error('Action space not valid for this agent.')
            sys.exit(0)

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)
        td3_log.debug(f'pfn_json_file:{self.pfn_json_file}')
        with open(self.pfn_json_file) as json_file:
            data = json.load(json_file)
        self.warmup_size = int(cfg_utils.cfg_get(data, 'warmup_size', 10000))
        self.batch_size = int(cfg_utils.cfg_get(data, 'batch_size', 100))
        self.model_load_path = cfg_utils.cfg_get(data, 'load_model', None)
        #
        self.lr_decay_interval = cfg_utils.cfg_get(data, 'lr_decay_interval', 1000)
        self.min_lr = cfg_utils.cfg_get(data, 'min_lr', 1e-8)
        self.lr_decay_rate = cfg_utils.cfg_get(data, 'lr_decay_rate', 0.95)

        self.actor_model_type = cfg_utils.cfg_get(
            data, 'actor_model', "mo_actor_fcnn-v0")
        self.critic_model_type = cfg_utils.cfg_get(
            data, 'critic_model', "mo_critic_fcnn-v0")

        self.logdir = logdir

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

        # Buffer
        if buffer_type is None:
            self.buffer_type = cfg_utils.cfg_get(data, 'buffer_type', None)
        else:
            self.buffer_type = buffer_type

        self.buffer = jlab_opt_control.buffers.make(
            self.buffer_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, logdir=self.logdir, buffer_size=buffer_size)
        self.buffer.save_cfg()

        # Used to update target networks
        self.tau = float(cfg_utils.cfg_get(data, 'tau', 0.005))
        self.gamma = float(cfg_utils.cfg_get(data, 'discount', 0.99))

        # Setup Optimizers
        self.critic_lr = float(cfg_utils.cfg_get(
            data, 'critic_learning_rate', 5e-4))
        self.actor_lr = float(cfg_utils.cfg_get(
            data, 'actor_learning_rate', 1e-4))

        if processor == 'arm':
            td3_log.info('Using legacy Adam')
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(
                self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(
                self.actor_lr, epsilon=1e-08)
        else:
            self.critic_optimizer = tf.keras.optimizers.Adam(
                self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.Adam(
                self.actor_lr, epsilon=1e-08)

        self.initialize_new_models()

        # Load models for retraining
        if self.model_load_path is not None:
            self.load()

        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq = int(
            cfg_utils.cfg_get(data, 'actor_update_freq', 2))
        self.critic_update_freq = int(
            cfg_utils.cfg_get(data, 'critic_update_freq', 2))

        self.noise_clip = 0.5

        try:
            os.mkdir(self.logdir)
        except OSError as error:
            td3_log.warning(error)
        file_writer = tf.summary.create_file_writer(self.logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = 0

        # action noise parameters
        self.init_action_noise = 1e-2
        self.action_noise = self.init_action_noise
        self.action_noise_min = 1e-6
        self.action_decay = 0.95
        self.naction_for_noise_decay = 1000

        # model reset parameters
        self.max_action_reset = 4
        self.naction_reset = 0
        self.naction_for_reset = 5000

        ##
        # self.genai_cebaf_sampling_model = GenEnergy()
        # self.genai_cebaf_samples = self.genai_cebaf_sampling_v2(self.warmup_size)
        #print(self.genai_cebaf_samples[0:10])
        #sys.exit()


    # def alpha_alignment_model(self):

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        td3_log.info('Running KerasTD3 initialize_new_models()')

        self.actor_model = jlab_opt_control.models.make(
            self.actor_model_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, min_action=self.lower_bound, max_action=self.upper_bound, logdir=self.logdir)
        self.target_actor = jlab_opt_control.models.make(
            self.actor_model_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, min_action=self.lower_bound, max_action=self.upper_bound, logdir=self.logdir)

        self.actor_model.save_cfg()

        seed1 = time.time_ns()
        str_seed1 = str(seed1)
        seed1 = int(str_seed1[9:-3])
        td3_log.debug(f'seed1:{seed1}')
        tf.random.set_seed(seed1)

        self.critic_model1 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, logdir=self.logdir)
        self.target_critic1 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, logdir=self.logdir)

        self.critic_model1.save_cfg()

        time.sleep(1 / 10)
        seed2 = time.time_ns()
        str_seed2 = str(seed2)
        seed2 = int(str_seed2[9:-3])
        td3_log.debug(f'seed2:{seed2}')
        tf.random.set_seed(seed2)

        self.critic_model2 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, logdir=self.logdir)
        self.target_critic2 = jlab_opt_control.models.make(
            self.critic_model_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, logdir=self.logdir)

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic1.set_weights(self.critic_model1.get_weights())
        self.target_critic2.set_weights(self.critic_model2.get_weights())

    # def reset_actor(self):
    #
    #     time.sleep(1 / 10)
    #     seed = time.time_ns()
    #     str_seed = str(seed)
    #     seed = int(str_seed[9:-3])
    #     td3_log.debug(f'New actor seed:{seed}')
    #     tf.random.set_seed(seed)
    #
    #     self.action_noise = self.init_action_noise
    #
    #     # Delete
    #     self.actor_model = None
    #     self.target_actor = None
    #     self.actor_optimizer = None
    #
    #     self.actor_model = jlab_opt_control.models.make(
    #         self.actor_model_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, min_action=self.lower_bound, max_action=self.upper_bound, logdir=self.logdir)
    #     self.target_actor = jlab_opt_control.models.make(
    #         self.actor_model_type, state_dim=self.num_states, action_dim=self.num_actions, reward_dim=self.num_rewards, min_action=self.lower_bound, max_action=self.upper_bound, logdir=self.logdir)
    #     if processor == 'arm':
    #         td3_log.info('Using legacy Adam')
    #         self.actor_optimizer = tf.keras.optimizers.legacy.Adam(
    #             self.actor_lr, epsilon=1e-08)
    #     else:
    #         self.actor_optimizer = tf.keras.optimizers.Adam(
    #             self.actor_lr, epsilon=1e-08)
    #     td3_log.info(f'->Resetting actor model and optimizer #{self.naction_reset}')

    # @tf.function
    # def train_critic(self, states, actions, rewards, next_states, dones, weights, alphas):
    #
    #
    #     # Critic 1 and 2
    #     with tf.GradientTape() as tape:
    #         q_values1 = self.critic_model1(states, actions, training=True)
    #         q_values2 = self.critic_model2(states, actions, training=True)
    #         td_errors1 = q_values1 - rewards
    #         td_errors2 = q_values2 - rewards
    #
    #         if "PER" in self.buffer_type:
    #             critic_loss1 = self.mse_loss(
    #                 q_values1, rewards, sample_weight=weights)
    #             critic_loss2 = self.mse_loss(
    #                 q_values2, rewards, sample_weight=weights)
    #         else:
    #             critic_loss1 = self.mse_loss(q_values1, rewards)
    #             critic_loss2 = self.mse_loss(q_values2, rewards)
    #
    #         critic_losses = critic_loss1 + critic_loss2
    #
    #     gradients = tape.gradient(
    #         critic_losses, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables)
    #     self.critic_optimizer.apply_gradients(zip(
    #         gradients, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables))
    #
    #     td_errors_avg = (tf.abs(td_errors1) + tf.abs(td_errors2)) / 2
    #
    #     return critic_loss1, critic_loss2, td_errors_avg

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones, weights, alphas):
        # Generate the proper noise
        noise = (tf.random.normal(tf.shape(actions), dtype=tf.float32) * 0.2)
        noise_clipped = tf.clip_by_value(
            noise, -self.noise_clip, self.noise_clip) * self.target_actor.action_scale
        next_actions = tf.clip_by_value(self.target_actor(
            next_states, alphas, training=False) + noise_clipped, self.lower_bound, self.upper_bound)

        target_q1 = self.target_critic1(
            next_states, next_actions, training=False)
        target_q2 = self.target_critic2(
            next_states, next_actions, training=False)
        target_q = tf.math.minimum(target_q1, target_q2)

        # Bellman equation for the q value
        q_targets = rewards + self.gamma * target_q * (1.0 - dones)

        # Critic 1 and 2
        with tf.GradientTape() as tape:
            q_values1 = self.critic_model1(states, actions, training=True)
            q_values2 = self.critic_model2(states, actions, training=True)
            td_errors1 = q_values1 - q_targets
            td_errors2 = q_values2 - q_targets

            if "PER" in self.buffer_type:
                critic_loss1 = self.mse_loss(
                    q_values1, q_targets, sample_weight=weights)
                critic_loss2 = self.mse_loss(
                    q_values2, q_targets, sample_weight=weights)
            else:
                critic_loss1 = self.mse_loss(q_values1, q_targets)
                critic_loss2 = self.mse_loss(q_values2, q_targets)

            critic_losses = critic_loss1 + critic_loss2

        gradients = tape.gradient(
            critic_losses, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables)
        gradients = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients]
        self.critic_optimizer.apply_gradients(zip(
            gradients, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables))

        td_errors_avg = (tf.abs(td_errors1) + tf.abs(td_errors2)) / 2

        return critic_loss1, critic_loss2, td_errors_avg

    @tf.function
    def train_actor(self, states, alphas):

        # Use Critic 1
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, alphas, training=True)
            q_values1 = self.critic_model1(states, actions, training=False)
            q_values2 = self.critic_model2(states, actions, training=False)
            q_values = q_values1+q_values2
            q_values_alpha = q_values*alphas
            q_loss = -tf.math.reduce_mean(q_values_alpha)
            cosine_loss = self.cosine_loss(q_values, alphas)
            loss = q_loss

        gradients = tape.gradient(loss, self.actor_model.trainable_variables)
        gradients = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients]

        self.actor_optimizer.apply_gradients(
            zip(gradients, self.actor_model.trainable_variables))

        return loss, q_loss, cosine_loss

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau +
                                 target_weight * (1.0 - self.tau))

    def train(self):
        """ Method used to train """
        self.ntrain_calls += 1
        if self.ntrain_calls % self.lr_decay_interval == 0:
            current_lr = self.actor_optimizer.learning_rate.numpy()
            if current_lr > self.min_lr:
                new_lr = current_lr * self.lr_decay_rate
                self.actor_optimizer.learning_rate.assign(new_lr)
                td3_log.info(f'Updating actor learning rate: {new_lr}')
            current_lr = self.critic_optimizer.learning_rate.numpy()
            if current_lr > self.min_lr:
                new_lr = current_lr * self.lr_decay_rate
                self.critic_optimizer.learning_rate.assign(new_lr)
                td3_log.info(f'Updating critic 1 learning rate: {new_lr}')

        if self.buffer.size() >= np.min([self.batch_size, self.warmup_size]):
        #if self.buffer.size() >= np.max([self.batch_size, self.warmup_size]):
            # Get sampling range
            if "PER" in self.buffer_type:
                states, actions, rewards, next_states, dones, weights, alphas = self.buffer.sample(
                    self.batch_size)
                weights_batch = tf.convert_to_tensor(weights, dtype=tf.float32)
            elif "ER" in self.buffer_type: # CHANGE THIS TO USE THE ALPHAS
                states, actions, rewards, next_states, dones, _, alphas = self.buffer.sample(
                    self.batch_size)
            else:
                print("ERROR: Please check configuration of agent for buffer type.")

            # Convert to tensors
            state_batch = tf.convert_to_tensor(states, dtype=tf.float32)
            action_batch = tf.convert_to_tensor(actions, dtype=tf.float32)
            reward_batch = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(
                next_states, dtype=tf.float32)
            done_batch = tf.convert_to_tensor(dones, dtype=tf.float32)
            alpha_batch = tf.convert_to_tensor(alphas, dtype=tf.float32)

            # Train critic
            if "PER" in self.buffer_type:
                critic_loss1, critic_loss2, td_errors = self.train_critic(state_batch, action_batch, reward_batch,
                                                                          next_state_batch, done_batch, weights_batch, alpha_batch)
            elif "ER" in self.buffer_type:
                critic_loss1, critic_loss2, td_errors = self.train_critic(state_batch, action_batch, reward_batch,
                                                                          next_state_batch, done_batch, _, alpha_batch)

            tf.summary.scalar('Critic Loss 1', data=critic_loss1,
                              step=int(self.ntrain_calls))
            tf.summary.scalar('Critic Loss 2', data=critic_loss2,
                              step=int(self.ntrain_calls))

            # Update Priorities
            if "PER" in self.buffer_type:
                new_priorities = td_errors.numpy()
                # Take the sum of the td_errors 64x3 = 64x1
                # Want them to be independent (LOW TO DO)
                self.buffer.update_priorities(new_priorities)

            if self.buffer.size() >= np.max([self.batch_size, self.warmup_size]):
                actor_loss, q_loss, cosine_loss = self.train_actor(state_batch, alpha_batch)
                tf.summary.scalar('Actor Loss', data=actor_loss, step=int(self.ntrain_calls))
                tf.summary.scalar('Q-Loss', data=actor_loss, step=int(self.ntrain_calls))
                tf.summary.scalar('Cosine Loss', data=cosine_loss, step=int(self.ntrain_calls))

            if self.ntrain_calls % self.actor_update_freq == 0:
                    self.soft_update(self.target_actor.variables, self.actor_model.variables)

            if self.ntrain_calls % self.critic_update_freq == 0:
                self.soft_update(self.target_critic1.variables,
                                 self.critic_model1.variables)
                self.soft_update(self.target_critic2.variables,
                                 self.critic_model2.variables)

            # if self.ntrain_calls % 1000 == 0:
            #     current_lr = self.actor_optimizer.learning_rate.numpy()
            #     if current_lr > 1e-8:
            #         self.actor_optimizer.learning_rate.assign(current_lr * 0.7)
        #print('outside of train...')

    def cebaf_sampling(self):
        isValid = False
        nTrials = 0
        norm_a = np.zeros(self.num_actions)
        while(isValid==False and nTrials<10000):
            nTrials +=1
            norm_a = np.random.uniform(-1.0, 1.0, self.num_actions)
            a = self.env.denormalize_state(norm_a)
            sample_e = self.env.get_energy(a)
            #print(f'energy: {sample_e} --> {self.env.min_energy}/{self.env.max_energy}')
            if sample_e > self.env.min_energy and sample_e < self.env.max_energy:
                isValid = True
        #print(f'nTrails: {nTrials}')
        return norm_a

    # def cebaf_sampling(self, ndim, std_min=0.8):
    #     ntrails = 0
    #     isValid = False
    #     v = None
    #     while (ntrails<10000):
    #         # Create data circle
    #         norm_dim = np.random.normal(0, std_min, ndim + 2)
    #         norm = np.sum(norm_dim * norm_dim) ** (0.5)
    #         vector = [norm_dim[i] / norm for i in range(ndim)]
    #         a = self.env.denormalize_state(np.array(vector))
    #         sample_e = self.env.get_energy(a)
    #         #r = np.sqrt(sum([v * v for v in vector]))
    #         #sample_e = self.env.denormalize_energy(r)
    #         #a = self.env.denormalize_state(np.abs(vector)+0.25*np.ones(ndim))
    #         #sample_e = np.sqrt(sum([v * v for v in a]))
    #         # vector= np.random.uniform(0.5, 1, self.num_actions)
    #         # a = self.env.denormalize_state(np.abs(vector))
    #         # sample_e = np.sqrt(sum([v * v for v in a]))
    #
    #         #print(f'energy: {sample_e} --> {self.env.min_energy}/{self.env.max_energy}')
    #         #print(f'action: {a}')
    #         #print(f'r: {r} -> min/max: {self.env.normalize_energy(self.env.min_energy)}/{self.env.normalize_energy(self.env.max_energy)}')
    #         #sys.exit()
    #         ntrails += 1
    #         if sample_e > self.env.min_energy and  sample_e < self.env.max_energy:
    #             #isValid==True
    #             # #print(f'r: {r}/rcut: {rcut} -> good')
    #             # #td3_log.debug(f'ntrails: {ntrails}')
    #             v = np.array(vector)
    #         else:
    #             v = self.env.action_space.sample()
    #             #print(f'sample_e: {sample_e} -> {ntrails}')
    #     td3_log.debug(f'ntrails: {ntrails}')
    #     return v #self.env.action_space.sample()
    #     # else:
    #     #     self.cebaf_sampling(ndim)

    def genai_cebaf_sampling_v2(self, nsamples):
        # self.genai_cebaf_sampling_model.load_weights('../notebooks/PACES-MO-CEBAF-N-VEC-TF-20240912-181543')
        self.genai_cebaf_sampling_model.load_weights('../notebooks/PACES-MO-CEBAF-N-VEC-TF-v0-20240912-224155')
        #genai_cebaf_sampling_model = tf.keras.models.load_model('../notebooks/PACES-MO-CEBAF-N-VEC-TF-20240912-181242.keras')
        energy = np.random.uniform(self.env.min_energy, self.env.max_energy, nsamples)
        energy = np.expand_dims(energy, axis=1)
        noise = np.random.normal(0, 1, size=(nsamples, self.num_actions))
        # print(energy.shape)
        # print(noise.shape)
        norm_a = self.genai_cebaf_sampling_model.predict([energy, noise],verbose=None)
        #print(norm_a.shape)
        return norm_a

    def genai_cebaf_sampling(self):
        self.genai_cebaf_sampling_model.load_weights('../notebooks/PACES-MO-CEBAF-N-VEC-TF-20240912-181543')
        #genai_cebaf_sampling_model = tf.keras.models.load_model('../notebooks/PACES-MO-CEBAF-N-VEC-TF-20240912-181242.keras')
        energy = np.random.uniform(self.env.min_energy, self.env.max_energy, 1)
        energy = np.expand_dims(energy, axis=1)
        noise = np.random.normal(0, 0.5, size=(1, 10))
        # print(energy.shape)
        # print(noise.shape)
        norm_a = self.genai_cebaf_sampling_model.predict([energy, noise],verbose=None)
        #print(norm_a.shape)
        return norm_a[0]

    def action(self, state, alphas, train=True):
        """ Method used to provide the next action using the target model """
        # Warmup experience sample
        if self.buffer.size() < np.max([self.batch_size, self.warmup_size]):
            sampled_action = self.env.action_space.sample()
            # print(type(sampled_action))
            # print((sampled_action.shape))

            #sampled_action = self.genai_cebaf_samples[self.buffer.size()]
            # noise = (tf.random.normal(shape=(self.num_actions,), mean=0,
            #                           stddev=self.actor_model.action_scale * self.action_noise,
            #                           dtype=tf.float32)).numpy()
            # sampled_action = np.clip(sampled_action + noise, self.lower_bound, self.upper_bound)
            # print(type(sampled_action))
            # print((sampled_action.shape))
            # sys.exit()
            #td3_log.debug(f'default sampled_action: {(sampled_action)}')
            # td3_log.debug(f'default sampled_action: {(sampled_action.shape)}')
            # td3_log.debug(f'default sampled_action: {type(sampled_action)}')
            # print(f'Env name: {(self.env.__str__).__name__}')
            # sys.exit()
            # sampled_action = self.env.action_space.sample()
            # if 'mo_cebaf_env' in self.env.__str__:
            #sampled_action = self.cebaf_sampling()

            #self.num_actions)
            #td3_log.debug(f'sampled_action: {(sampled_action)}')
            # td3_log.debug(f'sampled_action: {(sampled_action.shape)}')
            #td3_log.debug(f'sampled_action: {type(sampled_action)}')
            #td3_log.debug(f'state: {(state)}')
            #sampled_action = state+np.random.normal(0, 1, self.num_actions)
            noise = np.zeros(self.num_actions)
            #sys.exit()
        # Warmup completed, sample from actor
        else:
            #sys.exit()
            state = tf.cast(tf.expand_dims(state, 0), tf.float32)
            alphas = tf.cast(tf.expand_dims(alphas, 0), tf.float32)
            sampled_action = self.actor_model(state, alphas).numpy()
            if train:
                # Update the noise
                if self.nactions % self.naction_for_noise_decay == 0:
                    self.action_noise = self.action_noise * self.action_decay
                    if self.action_noise < self.action_noise_min:
                        self.action_noise = self.init_action_noise
                    # td3_log.info(f'-> Updating action noise is {self.action_noise}')
                noise = (tf.random.normal(shape=(self.num_actions,), mean=0,
                         stddev=self.actor_model.action_scale * self.action_noise, dtype=tf.float32)).numpy()
                #noise = np.abs(np.sin(tf.random.uniform(shape=(self.num_actions,)).numpy() * 2))*self.actor_model.action_scale
                sampled_action = np.clip(sampled_action + noise, self.lower_bound, self.upper_bound)
            else:
                noise = np.zeros(self.num_actions)

            sampled_action = sampled_action.flatten()
            noise = noise.flatten()
            # assert sampled_action.shape == self.num_actions or sampled_action.shape == (self.num_actions,), \
            #     f"Sampled action shape is incorrect... {sampled_action.shape}"

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

    def memory(self, obs_tuple):
        # inefficient but can fix later
        init_part, last_element = obs_tuple[:-1], obs_tuple[-1]
        memory_with_default_priority = init_part + (self.buffer.max_priority,) + (last_element,)
        self.buffer.record(memory_with_default_priority)

    def load(self):
        """ Load the ML models """
        try:
            self.actor_model.load_weights(
                join(self.model_load_path, "actor_model.h5"))
            self.target_actor.load_weights(
                join(self.model_load_path, "target_actor.h5"))
            self.critic_model1.load_weights(
                join(self.model_load_path, "critic_model1.h5"))
            self.target_critic1.load_weights(
                join(self.model_load_path, "target_critic1.h5"))
            self.critic_model2.load_weights(
                join(self.model_load_path, "critic_model2.h5"))
            self.target_critic2.load_weights(
                join(self.model_load_path, "target_critic2.h5"))
            td3_log.info('Models loaded successfully')
        except:
            print("Error while loading models, initializing new models...")

    def save(self, post_fix="test"):
        """ Save the ML models """
        try:
            destination_file_path = os.path.join(self.logdir, 'models/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)

            destination_file_path = os.path.join(destination_file_path, post_fix + '/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)

            self.actor_model.save_weights(
                join(destination_file_path, "actor_model_" + post_fix + ".h5"))
            self.target_actor.save_weights(
                join(destination_file_path, "target_actor_" + post_fix + ".h5"))
            self.critic_model1.save_weights(
                join(destination_file_path, "critic_model1_" + post_fix + ".h5"))
            self.target_critic1.save_weights(
                join(destination_file_path, "target_critic1_" + post_fix + ".h5"))
            self.critic_model2.save_weights(
                join(destination_file_path, "critic_model2_" + post_fix + ".h5"))
            self.target_critic2.save_weights(
                join(destination_file_path, "target_critic2_" + post_fix + ".h5"))
            td3_log.info('Agent models saved successfully')
        except:
            td3_log.error("Error in saving the models...")

    def save_cfg(self):
        """ Save the actor cfg """
        try:
            destination_file_path = os.path.join(self.logdir, 'cfgs/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)
            destination_file_path = os.path.join(
                destination_file_path, os.path.basename(self.pfn_json_file))
            shutil.copy(self.pfn_json_file, destination_file_path)
            td3_log.info('Agent config saved successfully')
        except:
            td3_log.error("Error in saving the agent cfg...")