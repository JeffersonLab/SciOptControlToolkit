"""Torch port of ActorFCNN (actor_fcnn.py) -- same architecture, same cfg
file, same action_scale/action_bias convention, driving TorchDEPO instead of
KerasDEPO. Doesn't subclass jlab_opt_control.core.model_core.Model (that
base class extends tf.keras.Model, TF-specific) -- just a plain nn.Module,
since TorchDEPO's whole training path (actor and env) is plain PyTorch.
"""

import json
import logging
import os
import shutil

import torch
import torch.nn as nn

import jlab_opt_control.utils.cfg_utils as cfg_utils

act_log = logging.getLogger("TorchActor")
act_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
}


class TorchActorFCNN(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, logdir, cfg='actor_fcnn.cfg'):
        super().__init__()

        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)

        with open(self.pfn_json_file, 'r') as f:
            cfg_data = json.load(f)
        hidden_layers = cfg_data.get('hidden_layers', 2)
        nodes_per_layer = cfg_data.get('nodes_per_layer', [256, 256])
        activation_functions = cfg_data.get('activation_functions', ["relu"] * hidden_layers)

        self.logdir = logdir

        if hidden_layers != len(nodes_per_layer):
            act_log.error("Number of nodes per layer does not match the number of hidden layers in the config.")
        elif hidden_layers != len(activation_functions):
            act_log.error("Number of activation functions does not match the number of hidden layers in the config.")

        layers = []
        prev_dim = state_dim
        for nodes, activation_name in zip(nodes_per_layer, activation_functions):
            layers.append(nn.Linear(prev_dim, nodes))
            layers.append(_ACTIVATIONS[activation_name.lower()]())
            prev_dim = nodes
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        min_action_t = torch.as_tensor(min_action, dtype=torch.float32)
        max_action_t = torch.as_tensor(max_action, dtype=torch.float32)
        self.register_buffer("action_scale", (max_action_t - min_action_t) / 2)
        self.register_buffer("action_bias", (max_action_t + min_action_t) / 2)

    def forward(self, state):
        return self.net(state) * self.action_scale + self.action_bias

    def save_cfg(self):
        try:
            destination_file_path = os.path.join(self.logdir, 'cfgs/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)
            destination_file_path = os.path.join(
                destination_file_path, os.path.basename(self.pfn_json_file))
            if not os.path.exists(destination_file_path):
                shutil.copy(self.pfn_json_file, destination_file_path)
                act_log.info('Actor model config saved successfully')
        except Exception:
            act_log.error("Error in saving the actor model cfg...")
