import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
from tensorflow.keras import layers
import shutil
import os
import logging

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

        self.logdir = logdir

        # Actor Architecture
        self.l1 = layers.Dense(256, activation="relu",
                               input_shape=(state_dim,))
        self.l2 = layers.Dense(256, activation="relu")
        self.l3 = layers.Dense(action_dim, activation='tanh')

        self.action_scale = tf.constant(
            (max_action - min_action) / 2, dtype=tf.float32)
        self.action_bias = tf.constant(
            (max_action + min_action) / 2, dtype=tf.float32)

        self.max_action = max_action

    def call(self, state, training=False):
        a = self.l1(state)
        a = self.l2(a)
        a = self.l3(a)
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
