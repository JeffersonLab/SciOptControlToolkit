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

import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
import tensorflow as tf
import numpy as np
import os
from os.path import join
import time
import json
import platform
import sys
processor = platform.processor()

import logging

dqn_log = logging.getLogger("DQN-Agent")
dqn_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

class KerasDQN(jlab_opt_control.Agent):

    def __init__(self, env, logdir, cfg='keras_td3.json', **kwargs):
        """ Define all key variables required for all agent """

        # Get env info
        super().__init__(**kwargs)

        self.env = env
        self.memory = deque(maxlen=200000)
        self.target_train_counter = 0
        self.total_actions_taken = 1
        self.individual_action_taken = np.ones(self.env.action_space.n)

        ## Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)

        self.warmup_size = int(cfg_utils.cfg_get(data, 'warmup_size', 1000))
        self.gamma = float(cfg_utils.cfg_get(data, 'gamma', 0.95))
        self.epsilon = float(cfg_utils.cfg_get(data, 'epsilon',1))
        self.epsilon_min = float(cfg_utils.cfg_get(data, 'epsilon_min',0.05))
        self.epsilon_decay = float(cfg_utils.cfg_get(data, 'epsilon_decay',0.995))
        self.learning_rate = float(data['learning_rate']) if float(data['learning_rate']) else 0.001
        self.batch_size = int(data['batch_size']) if int(data['batch_size']) else 32
        self.target_train_interval = int(data['target_train_interval']) if int(data['target_train_interval']) else 50
        self.tau = float(data['tau']) if float(data['tau']) else 1.0
        self.save_model = data['save_model'] if str(data['save_model']) else './model'

        self.model = self.get_policy_model()
        self.target_model = self.get_policy_model()

        dqn_log.info('Running KerasDQN __init__')
        file_writer = tf.summary.create_file_writer(self.logdir + '/metrics')
        file_writer.set_as_default()

        def get_policy_model(self):
            ## Input: state ##
            state_input = Input(self.env.observation_space.shape)
            h1 = Dense(56, activation='relu')(state_input)
            h2 = Dense(56, activation='relu')(h1)
            h3 = Dense(56, activation='relu')(h2)
            output = Dense(self.env.action_space.n, activation='linear')(h3)
            model = Model(input=state_input, output=output)
            adam = Adam(lr=self.learning_rate, clipnorm=1.0, clipvalue=0.5)
            model.compile(loss=tf.keras.losses.Huber(), optimizer=adam)
            model.summary()
            return model

        def action(self, state, train=True):
            if (np.random.rand() <= self.epsilon) and train==True:
                logger.info('Random action')
                action = random.randrange(self.env.action_space.n)
                ## Update randomness
                if len(self.memory) > (self.batch_size):
                    self.epsilon_adj()
            else:
                logger.info('NN action')
                np_state = np.array(state).reshape(1, len(state))
                act_values = self.target_model.predict(np_state)
                action = np.argmax(act_values[0])

            return action
