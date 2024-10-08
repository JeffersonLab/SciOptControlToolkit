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
        activation_functions = cfg_data.get('activation_functions', ["relu"] * hidden_layers + ["tanh"])  # Defaults
        kernel_initializer = cfg_data.get('kernel_initializer', 'glorot_uniform')
 
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
            self.hidden_layers.append(layers.Dense(nodes_per_layer[i], use_bias=False, kernel_initializer=kernel_initializer, input_shape=(state_dim,) if i == 0 else ()))
            self.hidden_layers.append(layers.Activation(activation_functions[i]))
        # Output layer with its specified activation function
        self.output_layer = layers.Dense(action_dim, activation=activation_functions[-1])

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