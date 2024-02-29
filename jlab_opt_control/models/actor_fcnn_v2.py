import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
from tensorflow.keras import layers
import shutil
import os
import logging
import time

act_log = logging.getLogger("Actor")
act_log.setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class ActorFCNN_v2(Model):
    def __init__(self, state_dim, action_dim, min_action, max_action, logdir, cfg='actor_fcnn_v2.cfg'):
        super().__init__()

        act_log.debug(f'state_dim: {state_dim}')
        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)

        self.logdir = logdir

        # Actor Architecture
        self.nlayers = 4
        self.nodes = 256
        seed = time.time_ns()
        init = tf.keras.initializers.GlorotUniform(seed)
        self.dense1, self.act1, self.bn1 = [], [], []
        self.dense1.append(tf.keras.layers.Dense(self.nodes, kernel_initializer=init, input_shape = (state_dim,)))
        self.act1.append(tf.keras.activations.tanh)
        self.bn1.append(tf.keras.layers.BatchNormalization())
        for i in range(1, self.nlayers):
            self.dense1.append(tf.keras.layers.Dense(self.nodes, kernel_initializer=init))
            self.act1.append(tf.keras.activations.tanh)
            self.bn1.append(tf.keras.layers.BatchNormalization())
        self.out = tf.keras.layers.Dense(action_dim, activation='tanh')

        self.action_scale = tf.constant(
            (max_action - min_action) / 2, dtype=tf.float32)
        self.action_bias = tf.constant(
            (max_action + min_action) / 2, dtype=tf.float32)

        self.max_action = max_action

    def call(self, state, training=False):
        act_log.debug(f'call state_dim: {state.shape}')
        a = state
        for i in range(self.nlayers):
            a = self.dense1[i](a)
            a = self.bn1[i](a)
            a = self.act1[i](a)
        a = self.out(a)
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
