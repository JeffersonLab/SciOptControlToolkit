"""DEPO — Differentiable Environment Policy Optimization.

A gradient-based, critic-free agent trained by backpropagating through a
differentiable environment.

Standard TD3/DDPG/SAC in this package learn a critic to approximate the value
of a state-action pair, then move the actor to increase that estimate. DEPO
skips the critic entirely: it requires the environment's own step() to be
differentiable (built from TF ops, not numpy — see
jlab_opt_control/envs/circle_env.py's Circle2D(backend='tensorflow') for the
interface contract and a reference implementation), unrolls the current actor `unroll_steps` steps forward
inside a single tf.GradientTape, sums the (discounted, done-masked) rewards
from that unroll as a direct stand-in for a value function, and backpropagates
straight through the chain of environment steps into the actor's weights.

IMPORTANT — this agent's train() runs its own internal training rollout
(see train_actor() below) against a private deep copy of the env it was
constructed with, not the live env object the driver is stepping through
the "real" episode. That rollout has nothing to do with whatever point the
driver's real episode is at — it resets fresh every call. This means the
driver's own logged episode reward is decorative for this agent's actual
learning signal — the real signal is entirely internal to train_actor().

Consequences of having no critic and no bootstrapped value: there's also no
replay buffer (nothing sampled from the past — every update is a fresh
on-policy rollout of the current actor through the current dynamics) and no
target networks (soft_update() is a no-op). memory() is a no-op too, purely
so this agent can still sit under drivers/run_continuous.py's per-step
`agent.memory(...); agent.train()` calling convention unchanged.
"""

import copy
import json
import logging
import os
import shutil
import sys
from os.path import join

import numpy as np
import tensorflow as tf

import jlab_opt_control as jlab_opt_control
import jlab_opt_control.buffers
import jlab_opt_control.models
import jlab_opt_control.utils.cfg_utils as cfg_utils

depo_log = logging.getLogger("DEPO-Agent")
depo_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class KerasDEPO(jlab_opt_control.Agent):

    def __init__(self, env, logdir, cfg='keras_depo.cfg', **kwargs):
        """ Define all key variables required for all agent """
        depo_log.info('Running KerasDEPO __init__')
        self.actor_model = None
        self.ntrain_calls = 0
        self.env = env
        # train_actor() unrolls its own rollout through this private copy
        # rather than the live env the driver is stepping through the
        # "real" episode, so training no longer corrupts the driver's
        # in-progress episode state. Unwrapped because wrappers like
        # TimeLimit (applied by the driver when --nsteps is set) don't
        # support DEPO's batched reset(batch_size=...)/step() contract.
        self.train_env = copy.deepcopy(getattr(env, 'unwrapped', env))

        try:
            assert "Box" in str(type(env.action_space)), 'Invalid action space'
            self.num_states = env.observation_space.shape[0]
            self.num_actions = env.action_space.shape[0]
            self.upper_bound = env.action_space.high
            self.lower_bound = env.action_space.low
            depo_log.info(f'Action upper bound: {self.upper_bound}')
            depo_log.info(f'Action lower bound: {self.lower_bound}')
        except:
            depo_log.error('Action space not valid for this agent.')
            sys.exit(0)

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)
        depo_log.debug(f'pfn_json_file:{self.pfn_json_file}')
        with open(self.pfn_json_file) as json_file:
            data = json.load(json_file)

        self.model_load_path = kwargs.get('load_model', cfg_utils.cfg_get(data, 'load_model', None))
        self.lr_decay_interval = int(cfg_utils.cfg_get(data, 'lr_decay_interval', 1000))
        self.min_lr = float(cfg_utils.cfg_get(data, 'min_lr', 1e-8))
        self.decay_rate = float(cfg_utils.cfg_get(data, 'decay_rate', 0.7))

        # The two settings that generalize this beyond a single hardcoded
        # unroll step: how many steps to unroll per train() call, and the
        # per-step discount applied when summing their rewards.
        self.unroll_steps = int(kwargs.get('unroll_steps', cfg_utils.cfg_get(data, 'unroll_steps', 1)))
        self.gamma = float(kwargs.get('discount', cfg_utils.cfg_get(data, 'discount', 0.99)))
        # Number of independent rollouts averaged per training step. Only
        # actually reduces gradient variance when reset()/the actor/the env
        # have some stochasticity (e.g. rdm_reset_mode='uniform'); under a
        # fully deterministic setup every one of these is an identical copy
        # of the same rollout, so it's a pure compute multiplier with no
        # statistical benefit — set it to 1 in that case.
        self.rollout_batch_size = int(cfg_utils.cfg_get(data, 'rollout_batch_size', 128))

        self.actor_model_type = cfg_utils.cfg_get(data, 'actor_model', "actor_fcnn-v0")
        self.actor_cfg = kwargs.get('actor_cfg')

        self.logdir = logdir

        self.actor_lr = float(cfg_utils.cfg_get(data, 'actor_learning_rate', 1e-4))
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr, epsilon=1e-08)

        # No replay buffer to train from (see module docstring), but the
        # driver's agent.buffer.save(...) calls are unconditional.
        self.buffer = jlab_opt_control.buffers.make(
            'NoOpBuffer-v0', state_dim=self.num_states, action_dim=self.num_actions, logdir=self.logdir)

        self.initialize_new_models()

        if self.model_load_path is not None:
            self.load()

        try:
            os.mkdir(self.logdir)
        except OSError as error:
            depo_log.warning(error)
        file_writer = tf.summary.create_file_writer(self.logdir + '/metrics')
        file_writer.set_as_default()
        self.nactions = 0
        self.inf_nactions = 0

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        depo_log.info('Running KerasDEPO initialize_new_models()')

        actor_kwargs = dict(state_dim=self.num_states, action_dim=self.num_actions,
                             min_action=self.lower_bound, max_action=self.upper_bound,
                             logdir=self.logdir)
        if self.actor_cfg is not None:
            actor_kwargs['cfg'] = self.actor_cfg
        self.actor_model = jlab_opt_control.models.make(self.actor_model_type, **actor_kwargs)

        # Run through the model once to initialize variables
        self.actor_model(tf.zeros([1, self.num_states]))
        self.actor_model.save_cfg()

    def train_actor(self):
        """Unroll the current actor unroll_steps forward through the
        differentiable env and backprop the discounted, done-masked sum of
        rewards straight into the actor's weights."""
        states, _ = self.train_env.reset(batch_size=self.rollout_batch_size)

        discounted_return = tf.zeros([self.rollout_batch_size, 1], dtype=tf.float32)
        mask = tf.ones([self.rollout_batch_size, 1], dtype=tf.float32)

        with tf.GradientTape() as tape:
            for k in range(self.unroll_steps):
                actions = self.actor_model(states, training=True)
                states, reward, terminated, truncated, _ = self.train_env.step(actions)

                reward = tf.reshape(tf.cast(reward, tf.float32), [-1, 1])
                discounted_return += (self.gamma ** k) * reward * mask

                done = tf.reshape(
                    tf.logical_or(tf.cast(terminated, tf.bool), tf.cast(truncated, tf.bool)),
                    [-1, 1])
                mask = mask * (1.0 - tf.cast(done, tf.float32))

            scalar_return = tf.reduce_mean(discounted_return)
            loss = -scalar_return

        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

        return loss, scalar_return

    def train(self):
        """ Method used to train """
        self.ntrain_calls += 1
        if self.ntrain_calls % self.lr_decay_interval == 0:
            current_lr = self.actor_optimizer.learning_rate.numpy()
            if current_lr > self.min_lr:
                self.actor_optimizer.learning_rate.assign(current_lr * self.decay_rate)

        actor_loss, discounted_return = self.train_actor()

        tf.summary.scalar('Actor Loss', data=actor_loss, step=int(self.ntrain_calls))
        tf.summary.scalar('Discounted Return', data=discounted_return, step=int(self.ntrain_calls))

        return discounted_return

    def action(self, state, train=True):
        """ Method used to provide the next action using the actor model """
        state = tf.expand_dims(state, 0)
        sampled_action = (self.actor_model(state)).numpy().flatten()
        noise = np.zeros(self.num_actions)

        if train:
            self.nactions += 1
            for i in range(self.num_actions):
                tf.summary.scalar('Action #{}'.format(i), data=sampled_action[i], step=int(self.nactions))
        else:
            self.inf_nactions += 1
            for i in range(self.num_actions):
                tf.summary.scalar('Inference Action #{}'.format(i), data=sampled_action[i], step=int(self.inf_nactions))

        return sampled_action, noise

    def memory(self, obs_tuple):
        """No-op: this agent has no replay buffer. It exists only so this
        agent can sit under drivers/run_continuous.py's per-step
        `agent.memory(...); agent.train()` calling convention unchanged."""
        pass

    def soft_update(self):
        """ No target networks — nothing to update. """
        return

    def load(self):
        """ Load the ML models """
        try:
            model_load_count = 0
            for file in os.listdir(self.model_load_path):
                if 'actor_model' in file and file.endswith('.weights.h5'):
                    self.actor_model.load_weights(join(self.model_load_path, file))
                    model_load_count += 1
            if model_load_count == 1:
                depo_log.info('Models loaded successfully')
            else:
                depo_log.error('Models not loaded properly, please check model save directory')
        except:
            depo_log.error("Error while loading models, initializing new models...")

    def save(self, post_fix="test"):
        """ Save the ML models """
        try:
            destination_file_path = os.path.join(self.logdir, 'models/')
            os.makedirs(destination_file_path, exist_ok=True)
            destination_file_path = os.path.join(destination_file_path, post_fix + '/')
            os.makedirs(destination_file_path, exist_ok=True)

            self.actor_model.save_weights(
                join(destination_file_path, "actor_model_" + post_fix + ".weights.h5"))
            depo_log.info('Agent models saved successfully')
        except:
            depo_log.error("Error in saving the models...")

    def save_cfg(self):
        """ Save the agent cfg """
        try:
            destination_file_path = os.path.join(self.logdir, 'cfgs/')
            os.makedirs(destination_file_path, exist_ok=True)
            destination_file_path = os.path.join(
                destination_file_path, os.path.basename(self.pfn_json_file))
            shutil.copy(self.pfn_json_file, destination_file_path)
            depo_log.info('Agent config saved successfully')
        except:
            depo_log.error("Error in saving the agent cfg...")
