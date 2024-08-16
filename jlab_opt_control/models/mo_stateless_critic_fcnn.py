import logging
import os
import shutil

from tensorflow.keras import layers

from jlab_opt_control.core.model_core import Model

crit_log = logging.getLogger("Critic")
crit_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class MO_StatelessCriticFCNN(Model):
    def __init__(self, state_dim, action_dim, reward_dim, logdir, cfg='mo_critic_fcnn.cfg'):
        super().__init__()

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)

        self.logdir = logdir

        # Q network Architecture
        #self.input_layer = layers.Dense(256, activation="relu", input_shape=(state_dim + action_dim,))
        self.input_layer = layers.Dense(128, activation="relu", input_shape=(action_dim,))
        hidden_layers = 3
        self.hidden_layers = []
        for i in range(hidden_layers):
            self.hidden_layers.append(layers.Dense(128, activation="leaky_relu"))
        self.output_layer = layers.Dense(reward_dim)

    def call(self, state, action, training=False):
        # Dynamic
        x = self.input_layer(action)
        for layer in self.hidden_layers:
            x = layer(x)
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