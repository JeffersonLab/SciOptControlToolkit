'''
TensorFlow version
'''
import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
from tensorflow.keras import layers

import os
import json
import logging
import shutil

uqsindy_log = logging.getLogger("UQ-SINDy")
uqsindy_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

class UQSINDyNetwork(Model):
    def __init__(self, num_features_in, num_features_out, batch_size, logdir, cfg='uqsindy_network.cfg'):
        super().__init__()
        
        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)

        # Read configuration for architecture
        with open(self.pfn_json_file, 'r') as f:
            cfg_data = json.load(f)
        num_features_in = num_features_in#cfg_data.get("num_features_in", 10) #Default
        num_features_out = num_features_out# cfg_data.get("num_features_out", 1) #Default
        self.batch_size = batch_size# cfg_data.get("batch_size", 1024)

        hidden_layers = cfg_data.get('hidden_layers', 2)  # Default to 2 if not specified
        nodes_per_layer = cfg_data.get('nodes_per_layer', [256, 256])  # Default
        activation_functions = cfg_data.get('activation_functions', ["tanh"] * hidden_layers + ["linear"])  # Defaults
        
        self.logdir = logdir
        
        # Error Checking
        if hidden_layers != len(nodes_per_layer) or hidden_layers != len(activation_functions)-1:
            if hidden_layers != len(nodes_per_layer):
                uqsindy_log.error("Number of nodes per layer does not match the number of hidden layers in the config.")
            else:  # hidden_layers != len(activation_functions)+1
                uqsindy_log.error("Number of activation functions (+1 for output layer) does not match the number of hidden layers in the config.") 


        # Parameters
        self.mu = tf.Variable(
            initial_value=tf.zeros([num_features_in, num_features_out], dtype=tf.float32),
            trainable=True,
        )
        self.log_var = tf.Variable(
            initial_value=tf.zeros([num_features_in, num_features_out], dtype=tf.float32),
            trainable=True,
        )

        # Network architecture
        total_features = num_features_in * num_features_out
        self.hidden_layers = []
        for i in range(hidden_layers):
            # Layer construction with dynamic activation functions
            self.hidden_layers.append(layers.Dense(nodes_per_layer[i], activation=activation_functions[i]))
        # Output layer with its specified activation function
        self.output_layer = layers.Dense(total_features, activation=activation_functions[-1])

    def sample_posterior(self, batch_size=1024):
        # Draw from latent distribution
        randn = tf.random.normal(shape=(batch_size, *self.mu.shape), dtype=tf.float32)
        betas = self.mu[None] + randn * tf.exp(0.5 * self.log_var[None])

        # Flatten last layer for use in neural network
        b, Ni, No = betas.shape
        betas = tf.reshape(betas, [b, Ni*No])

        for layer in self.hidden_layers:
            betas = layer(betas)
        betas = self.output_layer(betas)

        # Reshape last layer
        betas = tf.reshape(betas, [b, Ni, No])
        return betas

    def call(self, inputs, training=False):
        """ forward pass of model """
        x = inputs
        betas = self.sample_posterior(self.batch_size)
        BX = tf.einsum('nd,bdo->bno', x, betas)
        return BX

    def negative_log_likelihood(self, x, y0):
        """ Log likelihood of data given parameter distribution """
        BX  = self(x) # Distribution of predictions from distribution of parameters
        log_p_x = -0.5 * tf.reduce_sum(tf.square(x), axis=-1)[None] # Shape [1,N,]
        log_p_y = -0.5 * tf.reduce_sum(tf.square(y0 - BX), axis=-1) #Shape [B,N,]
        log_p_Xy = tf.reduce_sum(log_p_x + log_p_y, axis=1)
        return -log_p_Xy

    def kld(self):
        """ KL-Divergence of latent space from unit normal prior """
        kld = 0.5 * tf.reduce_sum(self.mu**2 + tf.exp(self.log_var) - self.log_var - 1)
        return kld

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
                uqsindy_log.info('SINDy model config saved successfully')
        except:
            uqsindy_log.error("Error in saving the actor model cfg...")