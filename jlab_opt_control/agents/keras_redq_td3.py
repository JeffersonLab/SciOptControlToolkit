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

import logging
import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
import tensorflow as tf
import numpy as np
import os
from os.path import join
import time
import json
import random
import platform
import sys
import shutil

from jlab_opt_control.agents.keras_td3 import KerasTD3

redq_log = logging.getLogger("REDQ-TD3-Agent")
redq_log.setLevel(logging.DEBUG)


class KerasREDQTD3(KerasTD3):
    """
    REDQ-TD3 implementation inheriting from TD3.
    Uses an ensemble of critics and higher update-to-data ratio.
    """

    def __init__(self, env, logdir, cfg='keras_redq_td3.cfg', **kwargs):
        """
        Initialize REDQ-TD3 agent by reusing TD3 initialization and adding REDQ-specific components.
        """
        # First, load the configuration to get REDQ-specific parameters
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        pfn_json_file = os.path.join(full_path, cfg)
        
        with open(pfn_json_file) as json_file:
            data = json.load(json_file)
            
        # Set REDQ-specific attributes before parent initialization
        self.critic_models = []
        self.target_critics = []
        self.num_critics = int(cfg_utils.cfg_get(data, 'num_critics', 10))
        self.utd_ratio = int(cfg_utils.cfg_get(data, 'utd_ratio', 20))
        self.in_target_min = int(cfg_utils.cfg_get(data, 'in_target_min', 2))
        self.train_steps = 0
        
        # Call parent initialization with modified parameters
        super().__init__(env, logdir, cfg, **kwargs)
        
        redq_log.info('Running KerasREDQTD3 __init__')
        redq_log.info(f'Number of critics: {self.num_critics}')
        redq_log.info(f'UTD ratio: {self.utd_ratio}')
        redq_log.info(f'Num min (M): {self.in_target_min}')

    def initialize_new_models(self):
        """
        Override parent's model initialization to create an ensemble of critics.
        """
        redq_log.info('Running KerasREDQTD3 initialize_new_models()')

        # Initialize actor and target actor (same as TD3)
        actor_kwargs = dict(state_dim=self.num_states, action_dim=self.num_actions,
                            min_action=self.lower_bound, max_action=self.upper_bound,
                            logdir=self.logdir)
        if self.actor_cfg is not None:
            actor_kwargs['cfg'] = self.actor_cfg
        self.actor_model = jlab_opt_control.models.make(self.actor_model_type, **actor_kwargs)
        self.target_actor = jlab_opt_control.models.make(self.actor_model_type, **actor_kwargs)

        # Run through model once to initialize variables
        self.actor_model(tf.zeros([1, self.num_states]))
        self.target_actor(tf.zeros([1, self.num_states]))

        self.actor_model.save_cfg()

        # Build critic kwargs once (branch-once), reused for every critic in the ensemble
        critic_kwargs = dict(state_dim=self.num_states, action_dim=self.num_actions,
                             logdir=self.logdir)
        if self.critic_cfg is not None:
            critic_kwargs['cfg'] = self.critic_cfg

        # Initialize ensemble of critics (REDQ-specific)
        for i in range(self.num_critics):
            seed = time.time_ns()
            str_seed = str(seed)
            seed = int(str_seed[9:-3])
            redq_log.debug(f'seed for critic {i}: {seed}')
            tf.random.set_seed(seed)

            critic_model = jlab_opt_control.models.make(self.critic_model_type, **critic_kwargs)
            target_critic = jlab_opt_control.models.make(self.critic_model_type, **critic_kwargs)
            
            # Run through model once to initialize variables
            critic_model(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_actions]))
            target_critic(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_actions]))
            
            # Copy weights from critic model to target critic
            target_critic.set_weights(critic_model.get_weights())
            
            self.critic_models.append(critic_model)
            self.target_critics.append(target_critic)
            
            # Only save config for the first critic (they all have the same architecture)
            if i == 0:
                critic_model.save_cfg()
            
            # Sleep for a different seed
            time.sleep(1 / 10)
        
        # Copy weights from actor model to target actor
        self.target_actor.set_weights(self.actor_model.get_weights())

    def train_critics(self, states, actions, rewards, next_states, dones, weights, critic_indices):
        """
        Train ensemble of critics with in-target minimization of subset of critics.
        """
        # Generate the proper noise
        noise = (tf.random.normal(tf.shape(actions), dtype=tf.float32) * 0.2)
        noise_clipped = tf.clip_by_value(
            noise, -self.noise_clip, self.noise_clip) * self.target_actor.action_scale
        next_actions = tf.clip_by_value(self.target_actor(
            next_states, training=False) + noise_clipped, self.lower_bound, self.upper_bound)

        # Get Q-values from selected target critics
        target_q_values = []
        for idx in critic_indices:
            target_q = self.target_critics[idx](next_states, next_actions, training=False)
            target_q_values.append(target_q)
        
        # Find the minimum Q-value among the selected critics (in-target minimization)
        target_q_stack = tf.stack(target_q_values, axis=1)
        target_q_min = tf.reduce_min(target_q_stack, axis=1)

        # Bellman equation for the q value
        q_targets = rewards + self.gamma * target_q_min * (1.0 - dones)

        critic_losses = []
        td_errors_list = []
        
        # Update each critic separately using the legacy optimizer
        for i, critic_model in enumerate(self.critic_models):
            with tf.GradientTape() as tape:
                q_values = critic_model(states, actions, training=True)
                td_errors = q_values - q_targets
                critic_loss = self.mse_loss(q_values, q_targets, sample_weight=weights)
                
            gradients = tape.gradient(critic_loss, critic_model.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(gradients, critic_model.trainable_variables))
            
            critic_losses.append(critic_loss)
            td_errors_list.append(tf.abs(td_errors))
        
        # Average TD errors for priority updates
        td_errors_avg = tf.reduce_mean(tf.stack(td_errors_list), axis=0)
        
        return critic_losses, td_errors_avg

    def train_actor(self, states):
        """
        Override parent's actor training to use random critic from ensemble.
        """
        # Use random critic from ensemble for policy update
        random_critic_idx = np.random.randint(0, self.num_critics)
        
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, training=True)
            q_value = self.critic_models[random_critic_idx](states, actions, training=False)
            loss = -tf.math.reduce_mean(q_value)

        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradient, self.actor_model.trainable_variables))
        return loss

    def train(self):
        """
        Override parent's train method to implement higher UTD ratio.
        """
        self.ntrain_calls += 1

        if self.buffer.size() > np.max([self.batch_size, self.warmup_size]):
            # Perform multiple updates per environment step (higher UTD ratio)
            for _ in range(self.utd_ratio):
                self.train_steps += 1
                
                # Get samples
                states, actions, rewards, next_states, dones, weights = self.buffer.sample(
                    self.batch_size)

                # Convert to tensors
                state_batch = tf.convert_to_tensor(states, dtype=tf.float32)
                action_batch = tf.convert_to_tensor(actions, dtype=tf.float32)
                reward_batch = tf.convert_to_tensor(rewards, dtype=tf.float32)
                next_state_batch = tf.convert_to_tensor(next_states, dtype=tf.float32)
                done_batch = tf.convert_to_tensor(dones, dtype=tf.float32)
                weights_batch = tf.convert_to_tensor(weights, dtype=tf.float32)
                
                # Randomly select subset of critics for in-target minimization
                critic_indices = random.sample(range(self.num_critics), self.in_target_min)
                
                # Train critics
                critic_losses, td_errors = self.train_critics(
                    state_batch, action_batch, reward_batch,
                    next_state_batch, done_batch, weights_batch,
                    critic_indices)

                # Log all critic losses to TensorBoard
                for i, critic_loss in enumerate(critic_losses):
                    tf.summary.scalar(f'Critic losses/Critic {i+1}', data=critic_loss, step=int(self.train_steps))

                # Update Priorities if using PER
                if "PER" in self.buffer_type:
                    new_priorities = td_errors.numpy().squeeze()
                    self.buffer.update_priorities(new_priorities)

                # Update actor and target networks
                if self.train_steps % self.actor_update_freq == 0:
                    actor_loss = self.train_actor(state_batch)
                    tf.summary.scalar('Actor Loss', data=actor_loss,
                                    step=int(self.train_steps))
                    self.soft_update(self.target_actor, self.actor_model)

                if self.train_steps % self.critic_update_freq == 0:
                    for i in range(self.num_critics):
                        self.soft_update(self.target_critics[i], self.critic_models[i])

    def load(self):
        """
        Override parent's load method to handle ensemble of critics.
        """
        try:
            model_load_count = 0
            expected_models = 1 + 1 + self.num_critics * 2  # actor + target_actor + num_critics*(critic+target)
            
            # Load actor models
            for file in os.listdir(self.model_load_path):
                if 'actor_model' in file and file.endswith('.weights.h5'):
                    self.actor_model.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif 'target_actor' in file and file.endswith('.weights.h5'):
                    self.target_actor.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
            
            # Load critic models - pattern for filenames must match save method
            for i in range(self.num_critics):
                for file in os.listdir(self.model_load_path):
                    if f'critic_model_{i}' in file and file.endswith('.weights.h5'):
                        self.critic_models[i].load_weights(join(self.model_load_path, file))
                        model_load_count += 1
                    elif f'target_critic_{i}' in file and file.endswith('.weights.h5'):
                        self.target_critics[i].load_weights(join(self.model_load_path, file))
                        model_load_count += 1
            
            if model_load_count == expected_models:
                redq_log.info('Models loaded successfully')
            else:
                redq_log.error(f'Models not loaded properly, loaded {model_load_count}/{expected_models}')
        except Exception as e:
            redq_log.error(f"Error while loading models: {e}, initializing new models...")

    def save(self, post_fix="default"):
        """
        Override parent's save method to handle ensemble of critics.
        """
        try:
            destination_file_path = os.path.join(self.logdir, 'models/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)

            destination_file_path = os.path.join(destination_file_path, post_fix + '/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)

            # Save actor models
            self.actor_model.save_weights(
                join(destination_file_path, f"actor_model_{post_fix}.weights.h5"))
            self.target_actor.save_weights(
                join(destination_file_path, f"target_actor_{post_fix}.weights.h5"))
            
            # Save all critic models in ensemble
            for i, (critic_model, target_critic) in enumerate(zip(self.critic_models, self.target_critics)):
                critic_model.save_weights(
                    join(destination_file_path, f"critic_model_{i}_{post_fix}.weights.h5"))
                target_critic.save_weights(
                    join(destination_file_path, f"target_critic_{i}_{post_fix}.weights.h5"))
                
            redq_log.info('Agent models saved successfully')
        except Exception as e:
            redq_log.error(f"Error in saving the models: {e}")