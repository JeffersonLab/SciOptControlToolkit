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
from jlab_opt_control.agents.keras_sindy_critic_td3 import KerasSINDyCriticTD3
import jlab_opt_control.buffers
import jlab_opt_control.models
import tensorflow as tf

import numpy as np
import platform

processor = platform.processor()

td3_log = logging.getLogger("TD3-Agent")
td3_log.setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")


class KerasSINDyUncertaintyTD3(KerasSINDyCriticTD3):
    def action(self, state, train=True, inference=False):
        """ Method used to provide the next action using the target model """
        # Warmup experience sample
        if (self.buffer.size() < np.max([self.batch_size, self.warmup_size])) and inference == False:
            sampled_action = self.env.action_space.sample()
        # Warmup completed, sample from actor or run inference
        else:
            state = tf.expand_dims(state, 0)
            if train:
                N = 256
                sampled_actions = (tf.random.uniform(
                    shape=(N, self.num_actions), 
                    minval=self.lower_bound, 
                    maxval=self.upper_bound, 
                    dtype=tf.float32)).numpy()
                sampled_states = np.repeat(state, N, axis=0)

                # NN Critic evaluation
                q_values = self.critic_model1(sampled_states, sampled_actions, training=False).numpy()

                # SINDy Critic evaluation
                sampled_states_actions = tf.keras.layers.Concatenate(axis=1)([sampled_states, sampled_actions])
                lib_batch = self.library(sampled_states_actions)
                s_values = self.critic_sindy(lib_batch, training=False).numpy()

                # Uncertainty is defined by discrepancy between NN and SINDy
                unc = np.abs(q_values - s_values)
                action_idx = np.argmax(unc)
                sampled_action = sampled_actions[action_idx, None]                                          
            else: # Choose the actor model output
                sampled_action = (self.actor_model(state)).numpy()

            sampled_action = sampled_action.flatten()
            assert sampled_action.shape == self.num_actions or sampled_action.shape == (self.num_actions,), \
                f"Sampled action shape is incorrect... {sampled_action.shape}"

        # Log the training action(s) taken and iterate aciton counter
        if train:
            self.nactions += 1
            for i in range(self.num_actions):
                tf.summary.scalar('Action #{}'.format(
                    i), data=sampled_action[i], step=int(self.nactions))
        else:
            self.inf_nactions += 1
            for i in range(self.num_actions):
                tf.summary.scalar('Inference Action #{}'.format(
                    i), data=sampled_action[i], step=int(self.inf_nactions))

        return sampled_action, np.zeros(self.num_actions)