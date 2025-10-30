import tensorflow as tf
from tensorflow.keras import layers
import logging

from jlab_opt_control.models.critic_fcnn import CriticFCNN

crit_log = logging.getLogger("MO_Critic")
crit_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

class MOCriticFCNN(CriticFCNN):
    def __init__(self, state_dim, action_dim, logdir, reward_dim, cfg='critic_fcnn.cfg'):
        super().__init__(state_dim, action_dim, logdir, cfg)

        # Overwrite output layer
        self.output_layer = layers.Dense(reward_dim, activation="linear")

        crit_log.info("Initialized Multi-Objective Critic FCNN Model")

