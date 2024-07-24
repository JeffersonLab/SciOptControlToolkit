import logging
import os
import shutil

import tensorflow as tf
from tensorflow.keras import layers

from jlab_opt_control.core.model_core import Model

act_log = logging.getLogger("Actor")
act_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class MOActorFCNN(Model):
    def __init__(self, state_dim, action_dim, reward_dim, min_action, max_action, logdir, cfg='mo_actor_fcnn.cfg'):
        super().__init__()

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)

        self.logdir = logdir

        # Actor Architecture
        # input_shape = (state_dim + reward_dim,) # No need to input state since it's always the same for CEBAF one step env
        input_shape = (reward_dim,)
        self.input_layer = layers.Dense(128, activation="relu", input_shape=input_shape)
        hidden_layers = 4
        self.hidden_layers = []
        for i in range(hidden_layers):
            self.hidden_layers.append(layers.Dense(128, activation="leaky_relu"))
        self.output_layer = layers.Dense(action_dim, activation='tanh')

        self.action_scale = tf.constant(
            (max_action - min_action) / 2, dtype=tf.float32)
        self.action_bias = tf.constant(
            (max_action + min_action) / 2, dtype=tf.float32)

        self.max_action = max_action

    def call(self, state, alphas, training=False):
        # Ideally need to concat state with alpha but for CEBAF, init state is always same
        concatenated_input = alphas
        a = self.input_layer(concatenated_input)
        for layer in self.hidden_layers:
            a = layer(a)
        a = self.output_layer(a)
        return a * self.action_scale + self.action_bias

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