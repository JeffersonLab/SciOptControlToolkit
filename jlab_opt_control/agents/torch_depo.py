"""TorchDEPO -- torch port of KerasDEPO (keras_depo.py). Same algorithm,
same config keys, same no-critic/no-buffer/no-target-network design (see
keras_depo.py's module docstring for the full rationale -- it applies
unchanged here). The only real differences are mechanical: torch builds the
autograd graph automatically (no explicit GradientTape), parameter groups
hold the learning rate instead of a Keras optimizer attribute, and this
agent is meant to be paired with a torch-differentiable env
(reset(batch_size=...) / step(actions) returning torch tensors with
gradients preserved) rather than a TF-based differentiable env.
"""

import json
import logging
import os
import shutil
import sys
from os.path import join

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import jlab_opt_control as jlab_opt_control
import jlab_opt_control.models
import jlab_opt_control.utils.cfg_utils as cfg_utils

depo_log = logging.getLogger("TorchDEPO-Agent")
depo_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class TorchDEPO(jlab_opt_control.Agent):

    def __init__(self, env, logdir, cfg='torch_depo.cfg', **kwargs):
        """ Define all key variables required for all agent """
        depo_log.info('Running TorchDEPO __init__')
        self.actor_model = None
        self.ntrain_calls = 0
        self.env = env

        try:
            assert "Box" in str(type(env.action_space)), 'Invalid action space'
            self.num_states = env.observation_space.shape[0]
            self.num_actions = env.action_space.shape[0]
            self.upper_bound = env.action_space.high
            self.lower_bound = env.action_space.low
            depo_log.info(f'Action upper bound: {self.upper_bound}')
            depo_log.info(f'Action lower bound: {self.lower_bound}')
        except Exception:
            depo_log.error('Action space not valid for this agent.')
            sys.exit(0)

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

        self.unroll_steps = int(kwargs.get('unroll_steps', cfg_utils.cfg_get(data, 'unroll_steps', 1)))
        self.gamma = float(kwargs.get('discount', cfg_utils.cfg_get(data, 'discount', 0.99)))
        self.rollout_batch_size = int(cfg_utils.cfg_get(data, 'rollout_batch_size', 128))

        self.actor_model_type = cfg_utils.cfg_get(data, 'actor_model', "actor_fcnn_torch-v0")
        self.actor_cfg = kwargs.get('actor_cfg')

        self.logdir = logdir

        self.actor_lr = float(cfg_utils.cfg_get(data, 'actor_learning_rate', 1e-4))

        self.initialize_new_models()

        if self.model_load_path is not None:
            self.load()

        try:
            os.mkdir(self.logdir)
        except OSError as error:
            depo_log.warning(error)
        self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, 'metrics'))
        self.nactions = 0
        self.inf_nactions = 0

    def initialize_new_models(self):
        """ Initialize new models from scratch """
        depo_log.info('Running TorchDEPO initialize_new_models()')

        actor_kwargs = dict(state_dim=self.num_states, action_dim=self.num_actions,
                             min_action=self.lower_bound, max_action=self.upper_bound,
                             logdir=self.logdir)
        if self.actor_cfg is not None:
            actor_kwargs['cfg'] = self.actor_cfg
        self.actor_model = jlab_opt_control.models.make(self.actor_model_type, **actor_kwargs)
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.actor_lr, eps=1e-08)

        self.actor_model.save_cfg()

    def train_actor(self):
        """Unroll the current actor unroll_steps forward through the
        differentiable env and backprop the discounted, done-masked sum of
        rewards straight into the actor's weights. No explicit "tape" is
        needed -- torch builds the graph automatically as long as nothing
        along this path is .detach()'d or wrapped in no_grad() by the env's
        own step()."""
        states, _ = self.env.reset(batch_size=self.rollout_batch_size)

        discounted_return = torch.zeros(self.rollout_batch_size, 1)
        mask = torch.ones(self.rollout_batch_size, 1)

        for k in range(self.unroll_steps):
            actions = self.actor_model(states)
            states, reward, terminated, truncated, _ = self.env.step(actions)

            reward = reward.reshape(-1, 1).to(torch.float32)
            discounted_return = discounted_return + (self.gamma ** k) * reward * mask

            done = torch.logical_or(terminated.bool(), truncated.bool()).reshape(-1, 1).to(torch.float32)
            mask = mask * (1.0 - done)

        scalar_return = discounted_return.mean()
        loss = -scalar_return

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.detach(), scalar_return.detach()

    def train(self):
        """ Method used to train """
        self.ntrain_calls += 1
        if self.ntrain_calls % self.lr_decay_interval == 0:
            for group in self.actor_optimizer.param_groups:
                if group['lr'] > self.min_lr:
                    group['lr'] *= self.decay_rate

        actor_loss, discounted_return = self.train_actor()

        self.writer.add_scalar('Actor Loss', actor_loss.item(), self.ntrain_calls)
        self.writer.add_scalar('Discounted Return', discounted_return.item(), self.ntrain_calls)

        return discounted_return

    def action(self, state, train=True):
        """ Method used to provide the next action using the actor model """
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            sampled_action = self.actor_model(state_t).numpy().flatten()
        noise = np.zeros(self.num_actions)

        if train:
            self.nactions += 1
            for i in range(self.num_actions):
                self.writer.add_scalar(f'Action #{i}', sampled_action[i], self.nactions)
        else:
            self.inf_nactions += 1
            for i in range(self.num_actions):
                self.writer.add_scalar(f'Inference Action #{i}', sampled_action[i], self.inf_nactions)

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
        """ Load the ML models

        self.model_load_path is only ever set from an explicit --load_model
        (or load_model=...) request (see __init__/run_continuous.py) -- never
        an ambient "resume if present" check. A failure here means the
        caller asked for a specific checkpoint and didn't get it, so this
        raises (after logging) instead of silently continuing with a
        freshly-initialized actor, which would look like a successful load
        while actually training/evaluating from random weights."""
        try:
            self.actor_model.load_state_dict(torch.load(join(self.model_load_path, "actor_model.pt")))
            depo_log.info('Models loaded successfully')
        except Exception as error:
            depo_log.error(f"Error while loading models from {self.model_load_path}: {error}")
            raise

    def save(self, post_fix="test"):
        """ Save the ML models

        Written as models/<post_fix>/actor_model.pt -- the post_fix already
        makes the directory unique, so the filename itself stays fixed,
        matching what load() looks for (join(model_load_path,
        "actor_model.pt")) when model_load_path is set to one of these
        directories."""
        try:
            destination_file_path = os.path.join(self.logdir, 'models/')
            os.makedirs(destination_file_path, exist_ok=True)
            destination_file_path = os.path.join(destination_file_path, post_fix + '/')
            os.makedirs(destination_file_path, exist_ok=True)

            torch.save(
                self.actor_model.state_dict(),
                join(destination_file_path, "actor_model.pt"),
            )
            depo_log.info('Agent models saved successfully')
        except Exception:
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
        except Exception:
            depo_log.error("Error in saving the agent cfg...")
