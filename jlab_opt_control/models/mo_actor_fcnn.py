import tensorflow as tf
from tensorflow.keras import layers
import logging

from jlab_opt_control.models.actor_fcnn import ActorFCNN

act_log = logging.getLogger("MO_Actor")
act_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

class MOActorFCNN(ActorFCNN):
    def __init__(self, state_dim, action_dim, min_action, max_action, logdir, reward_dim=False, cfg='actor_fcnn.cfg'):
        super().__init__(state_dim, action_dim, min_action, max_action, logdir, cfg)
        
        # Overwrite the first hidden layer
        self.hidden_layers[0] = layers.Dense(self.nodes_per_layer[0], activation=self.activation_functions[0], input_shape=(state_dim+reward_dim,))

    def call(self, inputs, training=False):
        # Concatenate state and alphas for multi-objective input
        x = tf.concat([inputs[0], inputs[1]], axis=1)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x * self.action_scale + self.action_bias