import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model

# class actor_FCNN(jlab_opt_control.Model):
#     def __init__(self, state_dim, action_dim, min_action, max_action)


class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        super().__init__()

        # Actor Architecture
        self.l1 = layers.Dense(256, activation="relu", input_shape=(state_dim,))
        self.l2 = layers.Dense(256, activation="relu")
        self.l3 = layers.Dense(action_dim, activation='tanh')

        self.action_scale = tf.constant((max_action - min_action) / 2, dtype=tf.float32)
        self.action_bias = tf.constant((max_action + min_action) / 2, dtype=tf.float32)

        self.max_action = max_action

    def call(self, state, training=False):
        a = self.l1(state)
        a = self.l2(a)
        a = self.l3(a)
        return a * self.action_scale + self.action_bias