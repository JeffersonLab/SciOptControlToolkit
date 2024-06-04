import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
from tensorflow.keras import layers
import shutil
import os
import logging
import json

crit_log = logging.getLogger("Critic-v2")
crit_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class CriticFCNN_v2(Model):
    def __init__(self, state_dim, action_dim, logdir, cfg='critic_fcnn_v2.cfg'):
        super().__init__()
 
        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)
 
        # Read configuration for architecture
        with open(self.pfn_json_file, 'r') as f:
            cfg_data = json.load(f)
        self.hidden_layers = cfg_data.get('hidden_layers', 2)  # Default to 2 if not specified
        nodes_per_layer = cfg_data.get('nodes_per_layer', [256, 256])  # Default

        self.logdir = logdir

       # Error Checking
        if self.hidden_layers != len(nodes_per_layer):
            crit_log.error("Number of nodes per layer does not match the number of hidden layers in the config.")

        # Dynamic Q network Architecture
        init = tf.keras.initializers.GlorotUniform()
        self.init_bn = tf.keras.layers.BatchNormalization()
        #init = tf.keras.initializers.RandomUniform(minval=-5, maxval=5)  # GlorotUniform(seed)
        self.denses1, self.bn1, self.act1 = [], [], []
        for i in range(self.hidden_layers):
            self.denses1.append(tf.keras.layers.Dense(nodes_per_layer[i],
                                                      kernel_initializer=init,
                                                      input_shape=(state_dim + action_dim,) if i == 0 else ()))
            self.bn1.append(tf.keras.layers.BatchNormalization())
            self.act1.append(tf.keras.activations.tanh)
        # Output layer
        self.output_layer = layers.Dense(1, activation="linear")
 
    def call(self, state, action, training=False):
        x = tf.concat([state, action], axis=1)  # Concatenate state and action as input
        x = self.init_bn(x)
        for i in range(self.hidden_layers):
            x = self.denses1[i](x)
            x = self.bn1[i](x)
            x = self.act1[i](x)
        x = self.output_layer(x)
        return x

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
                crit_log.info('Critic model config saved successfully')
        except:
            crit_log.error("Error in saving the critic model cfg...")
