import jlab_opt_control as jlab_opt_control
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import shutil
import os
import logging
import json

act_log = logging.getLogger("Actor")
act_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class ActorGaussian(Model):
    def __init__(self, state_dim, action_dim, min_action, max_action, logdir, cfg='actor_gaussian.cfg'):
        super().__init__()
 
        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)
 
        # Read configuration for architecture
        with open(self.pfn_json_file, 'r') as f:
            cfg_data = json.load(f)
        hidden_layers = cfg_data.get('hidden_layers', 2)  # Default to 2 if not specified
        nodes_per_layer = cfg_data.get('nodes_per_layer', [256, 256])  # Default
        activation_functions = cfg_data.get('activation_functions', ["relu"] * hidden_layers + ["tanh"])  # Defaults
 
        self.logdir = logdir

        # Error Checking
        if hidden_layers != len(nodes_per_layer) or hidden_layers != len(activation_functions)-1:
            if hidden_layers != len(nodes_per_layer):
                act_log.error("Number of nodes per layer does not match the number of hidden layers in the config.")
            else:  # hidden_layers != len(activation_functions)+1
                act_log.error("Number of activation functions (+1 for output layer) does not match the number of hidden layers in the config.")
        if activation_functions[-1] not in ["tanh"]:
            act_log.error("Final layer activation function needs to be tanh. Scaling for the action space will not work as intended.")

        # Dynamic Actor Architecture
        self.hidden_layers = []
        for i in range(hidden_layers):
            # Layer construction with dynamic activation functions
            self.hidden_layers.append(layers.Dense(nodes_per_layer[i], activation=activation_functions[i], input_shape=(state_dim,) if i == 0 else ()))
        # Output layer with its specified activation function
        self.mean_layer = layers.Dense(action_dim)
        self.std_layer = layers.Dense(action_dim, activation="tanh")
 
        self.action_scale = tf.constant((max_action - min_action) / 2, dtype=tf.float32)
        self.action_bias = tf.constant((max_action + min_action) / 2, dtype=tf.float32)
        self.max_action = max_action
 
    def call(self, state, training=False):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        mean = self.mean_layer(x)
        std = self.std_layer(x) + 1. + 1e-6  # Add 1 to convert the range to positive [0.001, 2]; Add a tiny number to avoid zero
        # clip log_std to avoid explosion
        # log_std = tf.clip_by_value(log_std, -20.0, 1.0)
        # log_std = (log_std * 10) - 9
        # mean = mean * 3

        # std = tf.exp(log_std)

        normal_dist = tfp.distributions.Normal(mean, std)
        unnorm_action = normal_dist.sample()

        # Apply the tanh squashing to keep the gaussian bounded in (-1,1)
        action = tf.tanh(unnorm_action)

        # Calculate the log probability
        unnorm_log_pi = normal_dist.log_prob(unnorm_action)
        # Change log probability to account for tanh squashing as mentioned in
        # Appendix C of the paper
        norm_log_pi = unnorm_log_pi - tf.reduce_sum(tf.math.log(1 - action**2 + 1e-6), axis=1, keepdims=True)
        norm_action = action * self.action_scale + self.action_bias
        
        return norm_action, norm_log_pi


    def save_cfg(self):
        """ Save the model cfg """
        try:
            destination_file_path = os.path.join(self.logdir, 'cfgs/')
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)
            destination_file_path = os.path.join(
                destination_file_path, os.path.basename(self.pfn_json_file))
            if not os.path.exists(destination_file_path):
                shutil.copy(self.pfn_json_file, destination_file_path)
                act_log.info('Actor model config saved successfully')
        except:
            act_log.error("Error in saving the actor model cfg...")
