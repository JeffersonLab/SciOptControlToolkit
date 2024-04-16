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

sindy_log = logging.getLogger("SINDy")
sindy_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

class SINDyNetwork(Model):
    def __init__(self, num_features_in, num_features_out, logdir, cfg="sindy_network.cfg"):
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
        num_features_out = num_features_out #cfg_data.get("num_features_out", 1) #Default

        self.logdir = logdir

        self.coefs = tf.Variable(
            initial_value=tf.zeros([num_features_in, num_features_out]),
            trainable=True,
        )
    
    def call(self, inputs, training=False):
        """ forward pass of model """
        x = inputs
        x = tf.matmul(x, self.coefs)
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
                sindy_log.info('SINDy model config saved successfully')
        except:
            sindy_log.error("Error in saving the actor model cfg...")