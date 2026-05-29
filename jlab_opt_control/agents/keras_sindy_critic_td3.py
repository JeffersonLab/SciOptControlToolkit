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
from jlab_opt_control.agents.keras_td3 import KerasTD3
from jlab_opt_control.utils.sindy_lib.polynomial_library import PolynomialLibrary
from jlab_opt_control.utils.sindy_lib.fourier_library import FourierLibrary
import jlab_opt_control.utils.cfg_utils as cfg_utils
import jlab_opt_control.buffers
import jlab_opt_control.models
import tensorflow as tf
import tensorboard.plugins.hparams.api as hp

import numpy as np
import os
from os.path import join
import time
import json
import platform

processor = platform.processor()

td3_log = logging.getLogger("TD3-Agent")
td3_log.setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")


class KerasSINDyCriticTD3(KerasTD3):

    def __init__(
        self, env, logdir, buffer_type=None, buffer_size=None, cfg="keras_sindy_critic_td3.cfg"
    ):
        """Define all key variables required for all agent"""
        super().__init__(env, logdir, buffer_type, buffer_size, cfg)

        # Extract relevant information for SINDy critic
        with open(self.pfn_json_file) as json_file:
            data = json.load(json_file)
            
        self.critic_sindy = None
        self.sindy_model_type = cfg_utils.cfg_get(data, "sindy_model", "sindy_network-v0")
        self.sindy_lr = float(cfg_utils.cfg_get(data, "sindy_learning_rate", 1e-4))
        self.sindy_beta = float(cfg_utils.cfg_get(data, "sindy_beta", 1.0)) #Weight to SINDy model in actor gradient

        self.sindy_library = cfg_utils.cfg_get(data, "sindy_library", "PolynomialLibrary")
        self.sindy_library_kwargs = cfg_utils.cfg_get(data, "sindy_library_kwargs", {
            "degree": 5,
            "include_bias": True,
            "include_interaction": True
        })
        
        self.sindy_optimizer = tf.keras.optimizers.Adam(
            self.sindy_lr, epsilon=1e-08
        )

        hparams = {
            "sindy_library": self.sindy_library,
            **self.sindy_library_kwargs,
            "sindy_beta": self.sindy_beta,
        }
        hp.hparams(hparams)

        self.initialize_sindy_model()
    
    def initialize_sindy_model(self):
        """Initialize new SINDy model from scratch"""

        td3_log.info("Initializing SINDy critic models")

        # Make intial state for init
        rng = np.random.default_rng(1)
        init_states_action = tf.convert_to_tensor(
            rng.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.num_states+self.num_actions]),
            dtype=tf.float32,
        )
        td3_log.debug(f"init_states_action:{init_states_action.shape}")

        # SINDy Poly library
        self.library = eval(self.sindy_library)(**self.sindy_library_kwargs)
        self.library.fit(init_states_action)
        lib_batch = self.library(init_states_action)

        # SINDy Critic
        seed1 = time.time_ns()
        str_seed1 = str(seed1)
        seed1 = int(str_seed1[9:-3])
        td3_log.debug(f"seed1:{seed1}")
        tf.random.set_seed(seed1)

        self.critic_sindy = jlab_opt_control.models.make(
            self.sindy_model_type,
            num_features_in=self.library.output_dim_,
            num_features_out=1,
            logdir=self.logdir + "/sindy_test/",
        )
        self.critic_sindy(lib_batch)

    def train_sindy_critic(self, states, actions, weights):
        target_q1 = self.critic_model1(states, actions, training=False)
        target_q2 = self.critic_model2(states, actions, training=False)
        q_targets = tf.math.minimum(target_q1, target_q2)
        
        with tf.GradientTape() as tape:
            # Use tf.concat instead of instantiating a new Concatenate layer
            states_actions = tf.concat([states, actions], axis=1)
            lib_batch = self.library(states_actions)
            s_values = self.critic_sindy(lib_batch)
            sindy_loss = self.mse_loss(s_values, q_targets, sample_weight=weights)

        gradients = tape.gradient(sindy_loss, self.critic_sindy.trainable_variables)
        self.sindy_optimizer.apply_gradients(zip(
            gradients, self.critic_sindy.trainable_variables))
        
        return sindy_loss

    def train_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, training=True)
            states_actions = tf.concat([states, actions], axis=1)  # fix here too
            lib_batch = self.library(states_actions)

            q_critic = self.critic_model1(states, actions, training=False)
            q_sindy = self.critic_sindy(lib_batch, training=False)
            q_value = self.sindy_beta * q_sindy + (1 - self.sindy_beta) * q_critic

            loss = -tf.math.reduce_mean(q_value)
            
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradient, self.actor_model.trainable_variables))
        return loss

    def train(self):
        """ Method used to train """
        self.ntrain_calls += 1

        if self.buffer.size() > np.max([self.batch_size, self.warmup_size]):

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

            # Train critic
            critic_loss1, critic_loss2, td_errors = self.train_critic(state_batch, action_batch, reward_batch,
                                                                          next_state_batch, done_batch, weights_batch)

            tf.summary.scalar('Critic Loss 1', data=critic_loss1,
                              step=int(self.ntrain_calls))
            tf.summary.scalar('Critic Loss 2', data=critic_loss2,
                              step=int(self.ntrain_calls))
            
            # Train SINDy Critic
            sindy_loss = self.train_sindy_critic(state_batch, action_batch, weights_batch)
            tf.summary.scalar('SINDy Loss', data=sindy_loss,
                              step=int(self.ntrain_calls))

            # Update Priorities
            if "PER" in self.buffer_type:
                new_priorities = td_errors.numpy().squeeze()
                self.buffer.update_priorities(new_priorities)

            # Train actor
            if self.ntrain_calls % self.actor_update_freq == 0:
                actor_loss = self.train_actor(state_batch)
                tf.summary.scalar('Actor Loss', data=actor_loss,
                                  step=int(self.ntrain_calls))
                self.soft_update(self.target_actor, self.actor_model)

            if self.ntrain_calls % self.critic_update_freq == 0:
                self.soft_update(self.target_critic1, self.critic_model1)
                self.soft_update(self.target_critic2, self.critic_model2)

    def load(self):
        """Load the ML models"""
        try:
            model_load_count = 0
            for file in os.listdir(self.model_load_path):
                if "actor_model" in file and file.endswith(".weights.h5"):
                    self.actor_model.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif "target_actor" in file and file.endswith(".weights.h5"):
                    self.target_actor.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif "critic_model1" in file and file.endswith(".weights.h5"):
                    self.critic_model1.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif "target_critic1" in file and file.endswith(".weights.h5"):
                    self.target_critic1.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif "critic_model2" in file and file.endswith(".weights.h5"):
                    self.critic_model2.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif "target_critic2" in file and file.endswith(".weights.h5"):
                    self.target_critic2.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
                elif "critic_sindy" in file and file.endswith(".weights.h5"):
                    self.critic_sindy.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
            if model_load_count == 7:
                td3_log.info("Models loaded successfully")
            else:
                td3_log.error(
                    "Models not loaded properly, please check model save directory"
                )
        except:
            td3_log.error("Error while loading models, initializing new models...")

    def save(self, post_fix="test"):
        """Save the ML models"""
        try:
            destination_file_path = os.path.join(self.logdir, "models/")
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)

            destination_file_path = os.path.join(destination_file_path, post_fix + "/")
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)

            self.actor_model.save_weights(
                join(destination_file_path, "actor_model_" + post_fix + ".weights.h5")
            )
            self.target_actor.save_weights(
                join(destination_file_path, "target_actor_" + post_fix + ".weights.h5")
            )
            self.critic_model1.save_weights(
                join(destination_file_path, "critic_model1_" + post_fix + ".weights.h5")
            )
            self.target_critic1.save_weights(
                join(destination_file_path, "target_critic1_" + post_fix + ".weights.h5")
            )
            self.critic_model2.save_weights(
                join(destination_file_path, "critic_model2_" + post_fix + ".weights.h5")
            )
            self.target_critic2.save_weights(
                join(destination_file_path, "target_critic2_" + post_fix + ".weights.h5")
            )
            self.critic_sindy.save_weights(
                join(destination_file_path, "critic_sindy_" + post_fix + ".weights.h5")
            )
            td3_log.info("Agent models saved successfully")
        except:
            td3_log.error("Error in saving the models...")
