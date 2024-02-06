import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
from tensorflow.keras import layers

class critic_FCNN(Model):
    def __init__(self, state_dim, action_dim, cfg='critic_FCNN.cfg'):
        super().__init__()

        # Q network Architecture
        self.l1 = layers.Dense(256, activation="relu", input_shape=(state_dim + action_dim,))
        self.l2 = layers.Dense(256, activation="relu")
        self.l3 = layers.Dense(1)

    def call(self, state, action, training=False):
        x = tf.concat([state, action], axis=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x