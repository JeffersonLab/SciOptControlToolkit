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
import platform
import shutil
import sys
from os.path import join

import numpy as np
import tensorflow as tf

import jlab_opt_control as jlab_opt_control
import jlab_opt_control.buffers
import jlab_opt_control.models
import jlab_opt_control.utils.cfg_utils as cfg_utils

processor = platform.processor()

td3_log = logging.getLogger("MO TD3-Agent")
td3_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class MO_KerasMBControl():

    def __init__(self, env, logdir, buffer_type=None, buffer_size=None, cfg='mo_keras_mb.cfg', max_nsteps=5):
        """ Define all key variables required for all agent """

        # Get env info
        self.actor_model = None
        td3_log.info('Running KerasMOTD3 __init__')

        self.ntrain_calls = 0
        # Environment setup
        self.env = env
        self.max_nsteps = max_nsteps
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
        relative_path = "./../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)
        td3_log.debug(f'pfn_json_file:{self.pfn_json_file}')
        with open(self.pfn_json_file) as json_file:
            data = json.load(json_file)
        self.warmup_size = int(cfg_utils.cfg_get(data, 'warmup_size', 10000))
        self.batch_size = int(cfg_utils.cfg_get(data, 'batch_size', 100))
        self.model_load_path = cfg_utils.cfg_get(data, 'load_model', None)

        self.actor_model_type = cfg_utils.cfg_get(
            data, 'actor_model', "MO-Actor-FCNN-v0")

        self.logdir = logdir        


        # Setup Optimizers
        self.actor_lr = float(cfg_utils.cfg_get(
            data, 'actor_learning_rate', 1e-5))

        if processor == 'arm':
            td3_log.info('Using legacy Adam')
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(
                self.actor_lr, epsilon=1e-08)
        else:
            self.actor_optimizer = tf.keras.optimizers.Adam(
                self.actor_lr, epsilon=1e-08)

        self.initialize_new_models()

        # Load models for retraining
        if self.model_load_path is not None:
            self.load()

        # update counting
        self.noise_clip = 0.5

        try:
            os.mkdir(self.logdir)
        except OSError as error:
            td3_log.warning(error)
        file_writer = tf.summary.create_file_writer(self.logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = 0

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        td3_log.info('Running KerasTD3 initialize_new_models()')
    

        self.actor_model = jlab_opt_control.models.make(self.actor_model_type, 
                                                        state_dim=self.num_states, 
                                                        action_dim=self.num_actions, 
                                                        reward_dim=self.num_rewards, 
                                                        min_action=self.lower_bound, 
                                                        max_action=self.upper_bound, 
                                                        logdir=self.logdir)
        
        self.target_actor = jlab_opt_control.models.make(self.actor_model_type, 
                                                         state_dim=self.num_states, 
                                                         action_dim=self.num_actions, 
                                                         reward_dim=self.num_rewards, 
                                                         min_action=self.lower_bound, 
                                                         max_action=self.upper_bound, 
                                                         logdir=self.logdir)
        self.actor_model(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_rewards]))
        self.target_actor(tf.zeros([1, self.num_states]), tf.zeros([1, self.num_rewards]))

        self.target_actor.set_weights(self.actor_model.get_weights())
        

    #@tf.function
    def train_actor(self):
        
        scans = np.random.rand(100)
        alphas = tf.convert_to_tensor(np.stack([scans, (1-scans)*1.5], axis=1), dtype=tf.float32)
        states = self.env.reset()[0].numpy()
        states = tf.convert_to_tensor(np.array([states]*100), dtype=tf.float32)
        sigmas = tf.convert_to_tensor(np.ones(shape=(100,2))*0.1, dtype=tf.float32)
        noise = tf.random.normal(shape=alphas.shape, mean=0, stddev=10)

        with tf.GradientTape() as tape:
            # Unroll the future actions (next 5)
            
            actions = self.actor_model(states, alphas, training=True)
            next_state, reward, done, _, _ = self.env.step(actions)
            q_loss = reward * alphas
            q_loss = -tf.math.reduce_mean(q_loss)
            states = next_state

            loss = q_loss
            gamma = 0.99

            for i in range(self.max_nsteps-1):
                actions = self.actor_model(states, alphas, training=True)
                next_state, reward, done, _, _ = self.env.step(actions)
                q_loss = reward * alphas
                q_loss = -tf.math.reduce_mean(q_loss)
                loss = loss + gamma * q_loss
                gamma = gamma * gamma
                states = next_state

            # loss = q_loss

        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradient, self.actor_model.trainable_variables))
        
        cosine_loss = 0.
        return loss, q_loss, cosine_loss

    def train(self):
        """ Method used to train """ 
        self.ntrain_calls += 1   
        actor_loss, q_loss, mono_loss = self.train_actor()
        tf.summary.scalar('Actor Loss', data=actor_loss, step=int(self.ntrain_calls))
        tf.summary.scalar('Q-Loss', data=actor_loss, step=int(self.ntrain_calls))
        tf.summary.scalar('Mono Loss', data=mono_loss, step=int(self.ntrain_calls))
        
        self.target_actor.set_weights(self.actor_model.get_weights())
        


    def action(self, states, alphas, train=True):
        """ Method used to provide the next action using the target model """
        # states = self.env.reset()[0].numpy()
        # states = tf.convert_to_tensor(np.array([states]*100))
        sampled_action = self.actor_model(states, alphas, training=train)
        noise = np.random.rand(sampled_action.shape[0])
        

        # Insure action output by actor is in legal environment range
        return sampled_action, noise, alphas

    def load(self):
        """ Load the ML models """
        try:
            self.actor_model.load_weights(
                join(self.model_load_path, "actor_model.h5"))
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
            td3_log.info('Agent models saved successfully')
        except:
            td3_log.error("Error in saving the models...")

    def save_cfg(self):
        """ Save the actor cfg """
        try:
            destination_file_path = os.path.join(self.logdir, 'cfgs/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)
            shutil.copy(self.pfn_json_file, destination_file_path)
            td3_log.info('Agent config saved successfully')
        except:
            td3_log.error("Error in saving the agent cfg...")