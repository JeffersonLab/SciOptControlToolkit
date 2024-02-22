import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
from tensorflow.keras import layers
import shutil
import os
import logging

crit_log = logging.getLogger("Critic")
crit_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class CriticFCNN(Model):
    def __init__(self, state_dim, action_dim, logdir, cfg='critic_fcnn.cfg'):
        super().__init__()

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)

        self.logdir = logdir

        # Q network Architecture
        self.l1 = layers.Dense(256, activation="relu",
                               input_shape=(state_dim + action_dim,))
        self.l2 = layers.Dense(256, activation="relu")
        self.l3 = layers.Dense(1)

    def call(self, state, action, training=False):
        x = tf.concat([state, action], axis=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
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
