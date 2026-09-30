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

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

circle_env_log = logging.getLogger("Circle2D-Env")
circle_env_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class Circle2D(gym.Env):
    """Target-circle task. backend='numpy' (default) is the original,
    non-differentiable single-episode env -- unchanged from before backend
    support existed. backend='tensorflow'/'torch' additionally support
    reset(batch_size=...)/step(tensor_actions) with gradients preserved,
    for DEPO-family agents to backprop through; tensorflow/torch are only
    imported when one of those backends is actually selected.
    """

    def __init__(self, rdm_reset_mode='fixed', statefull=True, max_episode_steps=1, backend='numpy'):
        if backend not in ('numpy', 'tensorflow', 'torch'):
            raise ValueError(f"backend must be one of ('numpy', 'tensorflow', 'torch'), got {backend!r}")
        self.backend = backend
        self._batched = False
        self.ndim = 2
        self.rdm_reset_mode = rdm_reset_mode
        self.statefull = statefull
        circle_env_log.info('############# Init Circle2D #############')
        circle_env_log.info(f'Backend: {self.backend}')
        circle_env_log.info(f'Statefull: {self.statefull}')
        circle_env_log.info(f'RDM Reset Mode: {self.rdm_reset_mode}')

        # Setup the action and observation
        self.action_space = spaces.Box(
            low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        circle_env_log.info(f'Action upper bound: {self.action_space.high}')
        circle_env_log.info(f'Action lower bound: {self.action_space.low}')
        self.observation_space = spaces.Box(
            low=-np.ones(self.ndim), high=np.ones(self.ndim), dtype=np.float64)
        circle_env_log.info(
            f'State upper bound: {self.observation_space.high}')
        circle_env_log.info(f'State lower bound: {self.observation_space.low}')

        # Define the target
        self.target_value = 0.95
        circle_env_log.info(f'Target value: {self.target_value}')

        # Keep count of steps
        self.nsteps = 0
        self._max_episode_steps = max_episode_steps
        circle_env_log.info(f'Max episode steps: {self._max_episode_steps}')

        # Reset the env
        self.states, _ = self.reset()

    def step(self, action):
        self.nsteps += 1

        if self.backend == 'numpy':
            self.states = self.states + action
            if self.statefull == False:
                self.states = action
            sqrt_states = np.square(self.states)
            radius = np.sqrt(np.sum(sqrt_states))
            reward = 1000.0*np.exp(-5.0*np.abs(radius - self.target_value)+1e-6)

            if self.nsteps >= self._max_episode_steps:
                return self.states, reward, True, True, {}

            return self.states, reward, False, False, {}

        done = self.nsteps >= self._max_episode_steps

        if self.backend == 'tensorflow':
            import tensorflow as tf
            action_t = action if tf.is_tensor(action) else tf.convert_to_tensor(action, dtype=tf.float32)
            if not self._batched:
                action_t = tf.expand_dims(action_t, 0)
            self.states = self.states + action_t if self.statefull else action_t
            radius = tf.sqrt(tf.reduce_sum(tf.square(self.states), axis=-1))
            reward = 1000.0 * tf.exp(-5.0 * tf.abs(radius - self.target_value) + 1e-6)
            batch_n = self.states.shape[0]
            terminated, truncated = tf.fill([batch_n], done), tf.fill([batch_n], done)
            if not self._batched:
                return self.states[0].numpy(), float(reward[0]), bool(done), bool(done), {}
            return self.states, reward, terminated, truncated, {}

        # torch
        import torch
        action_t = action if torch.is_tensor(action) else torch.as_tensor(action, dtype=torch.float32)
        if not self._batched:
            action_t = action_t.unsqueeze(0)
        self.states = self.states + action_t if self.statefull else action_t
        radius = torch.sqrt(torch.sum(torch.square(self.states), dim=-1))
        reward = 1000.0 * torch.exp(-5.0 * torch.abs(radius - self.target_value) + 1e-6)
        batch_n = self.states.shape[0]
        terminated = torch.full((batch_n,), done, dtype=torch.bool)
        truncated = torch.full((batch_n,), done, dtype=torch.bool)
        if not self._batched:
            return self.states[0].detach().numpy(), float(reward[0]), bool(done), bool(done), {}
        return self.states, reward, terminated, truncated, {}

    def reset(self, *, batch_size=None, seed=None, options=None):
        if self.backend == 'numpy':
            if batch_size is not None:
                raise NotImplementedError('batch_size is only supported by the numpy backend')
            self.nsteps = 0
            if self.rdm_reset_mode == 'uniform':
                self.states = self.observation_space.sample()
            if self.rdm_reset_mode == 'fixed':
                self.states = np.zeros(self.ndim)
            return self.states, ''

        self.nsteps = 0
        self._batched = batch_size is not None
        n = batch_size if self._batched else 1
        uniform = self.rdm_reset_mode == 'uniform'

        if self.backend == 'tensorflow':
            import tensorflow as tf
            shape = [n, self.ndim]
            self.states = tf.random.uniform(shape, -1, 1) if uniform else tf.zeros(shape)
            return (self.states if self._batched else self.states[0].numpy()), {}

        # torch
        import torch
        shape = (n, self.ndim)
        self.states = torch.empty(shape).uniform_(-1, 1) if uniform else torch.zeros(shape)
        return (self.states if self._batched else self.states[0].numpy()), {}
