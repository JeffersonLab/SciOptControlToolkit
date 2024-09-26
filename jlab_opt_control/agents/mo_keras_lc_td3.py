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

import tensorflow as tf

import jlab_opt_control as jlab_opt_control
import jlab_opt_control.buffers
import jlab_opt_control.models
import jlab_opt_control.utils.cfg_utils as cfg_utils

processor = platform.processor()

td3_log = logging.getLogger("MO TD3-Agent")
td3_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


# Create data circle
import numpy as np
def circle_rdm_samples(ndim=2, nsamples=100, std_min=1, r_norm=1):
  all_vectors = []
  vectors = []
  ntrials = 0
  while len(vectors)<(nsamples):
    ntrials += 1
    norm_dim = np.random.normal(0,std_min, ndim+2)
    norm = np.sum(norm_dim*norm_dim)**(0.5)
    vector = [norm_dim[i]/norm*r_norm for i in range(ndim)]
    vectors.append(vector)
  return np.array(vectors)

class MO_KerasLCTD3(jlab_opt_control.Agent):

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

        # Constraints
        self.lyapunov_lr = float(cfg_utils.cfg_get(
            data, 'lyapunov_learning_rate', 1e-4))

        if processor == 'arm':
            td3_log.info('Using legacy Adam')
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(
                self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(
                self.actor_lr, epsilon=1e-08)
            self.lyapunov_optimizer = tf.keras.optimizers.legacy.Adam(
                self.actor_lr, epsilon=1e-08)
        else:
            self.critic_optimizer = tf.keras.optimizers.Adam(
                self.critic_lr, epsilon=1e-08)
            self.actor_optimizer = tf.keras.optimizers.Adam(
                self.actor_lr, epsilon=1e-08)
            self.lyapunov_optimizer = tf.keras.optimizers.Adam(
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
        relative_error = 2*(self.env.max_energy-self.env.min_energy)/(self.env.max_energy+self.env.min_energy)
        #print(f'relative_error: {relative_error}')
        #sys.exit()
        #self.init_action_noise = float(cfg_utils.cfg_get(data, 'init_actor_noise', relative_error))
        self.init_action_noise = relative_error/4
        self.action_noise = self.init_action_noise
        self.action_noise_min = float(cfg_utils.cfg_get(data, 'action_noise_min', 1e-5))
        self.action_noise_decay = float(cfg_utils.cfg_get(data, 'action_noise_decay', 0.95))
        self.naction_for_noise_decay = int(cfg_utils.cfg_get(data, 'naction_for_noise_decay', 1000))

        # model reset parameters
        # self.max_action_reset = 4
        # self.naction_reset = 0
        # self.naction_for_reset = 5000



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

        #
        # self.lyapunov_func = jlab_opt_control.models.make(self.critic_model_type, state_dim=self.num_states,
        #                                                action_dim=self.num_actions, reward_dim=self.num_rewards, logdir=self.logdir)

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones, weights, alphas):
        # Generate the proper noise
        noise = (tf.random.normal(tf.shape(actions), dtype=tf.float32) * 0.2)
        #noise = (tf.random.normal(tf.shape(actions), dtype=tf.float32) * 2.0*self.init_action_noise)
        #noise = circle_rdm_samples(self.num_actions, nsamples=states.shape[0], std_min=1, r_norm=2*self.action_noise)
        #noise = tf.convert_to_tensor(noise, dtype=tf.float32)

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
        self.critic_optimizer.apply_gradients(zip(
            gradients, self.critic_model1.trainable_variables + self.critic_model2.trainable_variables))

        td_errors_avg = (tf.abs(td_errors1) + tf.abs(td_errors2)) / 2

        return critic_loss1, critic_loss2, td_errors_avg

    # @tf.function
    # def barrier(self, x, mu):
    #     psi = tf.where(x < 0.0, 0.0, mu * x - 1.0 / mu * tf.math.log(1.0 / (mu * mu)) + 1.0 / mu)
    #     return psi

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

            # Get energy
            # min_gsets = tf.convert_to_tensor(self.env.linac.min_gsets, dtype=tf.float32)
            # max_gsets = tf.convert_to_tensor(self.env.linac.max_gset_to_use, dtype=tf.float32)
            # diff = max_gsets - min_gsets
            # pred_a = ((actions + 1.0) / 2.0) * diff + min_gsets
            # pred_energy = tf.reduce_sum(pred_a * self.env.linac.lengths, axis=1)
            # pred_energy = tf.expand_dims(pred_energy, axis=1)

            # # Lower energy bound
            # pred_de_min = pred_energy-self.env.min_energy
            # #penalty_min = tf.where(pred_de_min > 0, 0.0, 5*tf.math.abs(pred_de_min))
            # #pred_de_min = tf.convert_to_tensor(pred_de_min, dtype=tf.float32)
            # penalty_min = self.barrier( pred_de_min, 1e+3)
            # # #
            # # # Upper energy bound
            # pred_de_max = self.env.max_energy-pred_energy
            # penalty_max = tf.where(pred_de_max > 0, 0.0, 5*tf.math.abs(pred_de_max) )
            # pred_de_max = tf.convert_to_tensor(pred_de_max, dtype=tf.float32)
            # penalty_max = self.barrier( pred_de_max, 1e+3)
            #
            # penalty_energy = penalty_min + penalty_max

            # print(f'pred_energy: {pred_energy.numpy()[0]}; pred_de_min: {pred_de_min.numpy()[0]}; pred_de_max: {pred_de_max.numpy()[0]}; ')
            # print(f'\t; penalty_min: {penalty_min.numpy()[0]}; penalty_max: {penalty_max.numpy()[0]}; ')

            # # Upper bound trip
            # tr = tf.exp(-10.268+self.env.linac.trip_slopes*(pred_a - self.env.linac.trip_offsets))
            # tr = tf.where(self.env.linac.trip_slopes == 0, 0.0, tr)
            # pred_trip = 3600*tf.reduce_sum(tr, axis=1)
            # pred_dtrip = pred_trip - ref[1]
            # pred_dtrip = tf.convert_to_tensor( pred_dtrip, dtype=tf.float32)
            # penalty_trip = tf.where(pred_dtrip > 0, 5*tf.math.abs(pred_dtrip), 0)
            # #penalty_trip = self.barrier(pred_dtrip, 2)
            # #penalty_heat = tf.where(self.ntrain_calls > 1000, 0, penalty_heat)
            #
            # # Upper bound heat
            # pred_heat = tf.reduce_sum(((pred_a ** 2) * self.env.linac.lengths * 1e12) / (self.env.linac.shunts * self.env.linac.Q0s), axis=1)
            # pred_dheat = pred_heat - ref[0]
            # pred_dheat = tf.convert_to_tensor(pred_dheat, dtype=tf.float32)
            # penalty_heat = tf.where(pred_dheat > 0, 5*tf.math.abs(pred_dheat), 0)
            #penalty_heat = self.barrier(pred_dheat, 2)
            #penalty_heat = tf.where(self.ntrain_calls > 1000, 0, penalty_heat)

            # # Add all penalties
            # penalty_loss = tf.math.reduce_mean(penalty_trip + penalty_heat)
            #penalty_loss = tf.where(self.ntrain_calls > 1000, 0, penalty_loss)
            #penalty_loss = tf.math.reduce_mean(penalty_energy )
            loss = q_loss #+ penalty_loss


        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradient, self.actor_model.trainable_variables))

        return loss, q_loss, 0,0#tf.math.reduce_mean(penalty_min), tf.math.reduce_mean(penalty_max)

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
                states, actions, rewards, next_states, dones, alphas, lyapunov_vars = self.buffer.sample(self.batch_size)
            else:
                print("ERROR: Please check configuration of agent for buffer type.")

            # Convert to tensors
            state_batch = tf.convert_to_tensor(states, dtype=tf.float32)
            action_batch = tf.convert_to_tensor(actions, dtype=tf.float32)
            reward_batch = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(next_states, dtype=tf.float32)
            done_batch = tf.convert_to_tensor(dones, dtype=tf.float32)
            alpha_batch = tf.convert_to_tensor(alphas, dtype=tf.float32)
            lyapunov_batch = tf.convert_to_tensor(lyapunov_vars, dtype=tf.float32)

            # Train Lyapunov func
            # lyapunov_loss, lyapunov_heat, lyapunov_trip = self.train_lyapunov_func(state_batch, action_batch, lyapunov_batch)
            # tf.summary.scalar('Lyapunov Loss', data=lyapunov_loss, step=int(self.ntrain_calls))
            # tf.summary.scalar('Lyapunov Heat', data=lyapunov_heat, step=int(self.ntrain_calls))
            # tf.summary.scalar('Lyapunov Trip', data=lyapunov_trip, step=int(self.ntrain_calls))

            # Train critic
            if "PER" in self.buffer_type:
                critic_loss1, critic_loss2, td_errors = self.train_critic(state_batch, action_batch, reward_batch,
                                                                          next_state_batch, done_batch, weights_batch, alpha_batch)
            elif "ER" in self.buffer_type:
                critic_loss1, critic_loss2, td_errors = self.train_critic(state_batch, action_batch, reward_batch,
                                                                          next_state_batch, done_batch, None, alpha_batch)

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
                actor_loss, actor_qloss, penalty_min, penalty_max = self.train_actor(state_batch, alpha_batch)
                tf.summary.scalar('Actor Loss', data=actor_loss, step=int(self.ntrain_calls))
                tf.summary.scalar('Actor Q-Loss', data=actor_qloss, step=int(self.ntrain_calls))
                tf.summary.scalar('Actor Penalty Energy Min', data=penalty_min, step=int(self.ntrain_calls))
                tf.summary.scalar('Actor Penalty Energy Max', data=penalty_max, step=int(self.ntrain_calls))
                # tf.summary.scalar('Actor Loss', data=actor_loss, step=int(self.ntrain_calls))
                # tf.summary.scalar('Actor Penalty Energy Min', data=penalty_min, step=int(self.ntrain_calls))
                # tf.summary.scalar('Actor Penalty Energy Max', data=penalty_max, step=int(self.ntrain_calls))
                # tf.summary.scalar('Actor Penalty Trip', data=penalty_trip, step=int(self.ntrain_calls))

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

    def action(self, state, alphas, train=True):
        """ Method used to provide the next action using the target model """
        # Warmup experience sample
        if self.buffer.size() < np.max([self.batch_size, self.warmup_size]):
            sampled_action = self.env.action_space.sample()
            noise = np.zeros(self.num_actions)
        # Warmup completed, sample from actor
        else:
            #sys.exit()
            state = tf.cast(tf.expand_dims(state, 0), tf.float32)
            alphas = tf.cast(tf.expand_dims(alphas, 0), tf.float32)
            sampled_action = self.actor_model(state, alphas).numpy()
            if train:
                # Update the noise
                if self.nactions % self.naction_for_noise_decay == 0:
                    self.action_noise = self.action_noise * self.action_noise_decay
                    if self.action_noise < self.action_noise_min:
                        self.action_noise = self.init_action_noise
                    td3_log.info(f'-> Updating action noise is {self.action_noise}')
                #noise = (tf.random.normal(shape=(self.num_actions,), mean=0,
                #        stddev=self.actor_model.action_scale * self.action_noise, dtype=tf.float32)).numpy()
                noise = circle_rdm_samples(self.num_actions, nsamples=1, std_min=1, r_norm=self.action_noise)
                #td3_log.info(f'Sum noise: {np.sqrt(sum([v * v for v in noise]))}')
                #noise = np.abs(np.sin(tf.random.uniform(shape=(self.num_actions,)).numpy() * 2))*self.actor_model.action_scale
                sampled_action = np.clip(sampled_action + noise, self.lower_bound, self.upper_bound)
            else:
                noise = np.zeros(self.num_actions)

            sampled_action = sampled_action.flatten()
            noise = noise.flatten()
            assert sampled_action.shape == self.num_actions or sampled_action.shape == (self.num_actions,), \
                f"Sampled action shape is incorrect... {sampled_action.shape}"

        # Log the training action(s) taken
        max_shown_action = np.min([self.num_actions,10])
        if train:
            self.nactions = self.nactions + 1
            if self.num_actions == 0:
                tf.summary.scalar('Action', data=sampled_action,
                                  step=int(self.nactions))
            else:
                for i in range(max_shown_action):
                    tf.summary.scalar('Action #{}'.format(
                        i), data=sampled_action[i], step=int(self.nactions))
                    tf.summary.scalar('Action noise #{}'.format(
                        i), data=noise[i], step=int(self.nactions))

        # Insure action output by actor is in legal environment range
        sampled_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)

        return sampled_action, noise

    def memory(self, obs_tuple):
        #print(f'obs_tuple: {obs_tuple}')
        # inefficient but can fix later
        init_part, last_element = obs_tuple[:-1], obs_tuple[-1]
        memory_with_default_priority = init_part + (self.buffer.max_priority,) + (last_element,)
        #print(f'memory_with_default_priority: {memory_with_default_priority}')
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
