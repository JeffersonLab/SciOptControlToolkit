import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
from tensorflow.keras import layers
import shutil
import os
import logging
import json

act_log = logging.getLogger("Actor")
act_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class ActorFCNN(Model):
    def __init__(self, state_dim, action_dim, min_action, max_action, logdir, cfg='actor_fcnn.cfg'):
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
        activation_function = cfg_data.get('activation_function', "relu")  # Default

        if hidden_layers != len(nodes_per_layer):
            crit_log.error("Mismatch between number of hidden layers and number of nodes per hidden layer provided in config...")
 
        self.logdir = logdir
 
        # Dynamic Actor Architecture
        self.hidden_layers = []
        for i in range(hidden_layers):
            if i == 0:
                # Only the first layer needs the input_shape
                self.hidden_layers.append(layers.Dense(nodes_per_layer[i], activation=activation_function, input_shape=(state_dim,)))
            else:
                self.hidden_layers.append(layers.Dense(nodes_per_layer[i], activation=activation_function))
        self.output_layer = layers.Dense(action_dim, activation='tanh')
 
        self.action_scale = tf.constant((max_action - min_action) / 2, dtype=tf.float32)
        self.action_bias = tf.constant((max_action + min_action) / 2, dtype=tf.float32)
        self.max_action = max_action
 
    def call(self, state, training=False):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x * self.action_scale + self.action_bias

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
