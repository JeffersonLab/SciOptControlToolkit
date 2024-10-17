import json
import logging
import os
import shutil

import tensorflow as tf
from tensorflow.keras import layers

from jlab_opt_control.core.model_core import Model

act_log = logging.getLogger("Actor")
act_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


class MO_ActorFCNN_CIC(Model):
    def __init__(self, state_dim, action_dim, reward_dim, min_action, max_action, logdir, cfg='mo_actor_fcnn.cfg'):
        super().__init__()

        # Projections Layer
        import score.envs as gym
        self.action_dim = action_dim
        if self.action_dim==8:
            env_id = 'SCORE-MO-CEBAF-8D-VEC-TF'
        elif self.action_dim == 16:
            env_id = 'SCORE-MO-CEBAF-16D-VEC-TF'
        elif self.action_dim == 32:
            env_id = 'SCORE-MO-CEBAF-32D-VEC-TF'
        elif self.action_dim==197:
            env_id = 'SCORE-MO-CEBAF-N-VEC-TF-v0'
        else:
            act_log.error('Invalid CEBAF env')
            sys.exit()

        act_log.info(f'Using env: {env_id}')
        self.gc_env = gym.make(env_id)
        self.trip_high = self.gc_env.linac.max_allowed_trip
        self.heat_high = self.gc_env.linac.max_allowed_heat

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)

        # Read configuration for architecture
        with open(self.pfn_json_file, 'r') as f:
            cfg_data = json.load(f)
            
        hidden_layers = cfg_data.get('hidden_layers', 2)  # Default to 2 if not specified
        nodes_per_layer = cfg_data.get('nodes_per_layer', [256, 256])  # Default
        activation_functions = cfg_data.get('activation_functions', ["relu"] * hidden_layers)  # Defaults

        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.66)
        self.logdir = logdir
        
        # Error Checking
        if hidden_layers != len(nodes_per_layer):
            act_log.error("Number of nodes per layer does not match the number of hidden layers in the config.")
        elif hidden_layers != len(activation_functions):
            act_log.error("Number of activation functions does not match the number of hidden layers in the config.")

        # Actor Architecture
        input_shape = (state_dim + reward_dim,) # No need to input state since it's always the same for CEBAF one step env
        #input_shape = (reward_dim,)
        self.input_layer = layers.Dense(128, activation="tanh", input_shape=input_shape)
        
        # Dynamic Actor Architecture
        self.hidden_layers = []
        for i in range(hidden_layers):
            # Layer construction with dynamic activation functions
            self.hidden_layers.append(layers.Dense(nodes_per_layer[i], activation=activation_functions[i]))#,
                                                   #kernel_initializer=initializer))
        # Output layer with its specified activation function
        self.output_layer = layers.Dense(action_dim, activation="tanh")
        self.projection = tf.keras.layers.Dense(action_dim, use_bias=True)

 
        self.action_scale = tf.constant((max_action - min_action) / 2, dtype=tf.float32)
        self.action_bias = tf.constant((max_action + min_action) / 2, dtype=tf.float32)
        self.max_action = max_action

        self.max_iter = 25
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.005)

        self.err = 99999
        self.tolerance = 1e-2

    def call(self, state, alphas, training=False):
        # Ideally need to concat state with alpha but for CEBAF, init state is always same
        if training:
            self.iterations = 0
            while self.iterations < self.max_iter:
                with tf.GradientTape(persistent=False) as tape:
                    tape.watch(alphas)
                    concatenated_input = tf.concat([state, alphas], axis=1)
                    a = self.input_layer(concatenated_input)
                    for layer in self.hidden_layers:
                        a = layer(a)
                    a = self.output_layer(a)
                    self.iterations += 1
                    # Calculate safe x
                    safe_a = self.projection(a) + a
                    safe_a = tf.clip_by_value(safe_a, -1, 1)
                    pred_actions = self.gc_env.denormalize_action(safe_a)
                    pred_energies = self.gc_env.get_energy(pred_actions)[:, 0]
                    self.err_min = 10.0 * tf.keras.activations.relu(self.gc_env.min_energy - pred_energies)/self.gc_env.min_energy
                    self.err_max = 10.0 * tf.keras.activations.relu(pred_energies - self.gc_env.max_energy)/self.gc_env.max_energy
                    # pred_trip = self.gc_env.linac.getTripRates(gradients=pred_actions)
                    # pred_heat = self.gc_env.linac.getRFHeat(gradients=pred_actions)
                    # self.err_trip = (pred_trip - self.trip_high) / self.trip_high
                    # self.err_heat = (pred_heat - self.heat_high) / self.heat_high
                    self.err = tf.reduce_mean(self.err_min + self.err_max ) #+ self.err_trip + self.err_heat )
                    # distance = tf.keras.losses.CosineSimilarity()(safe_a, safe_a)
                    # self.err -= distance
                    if self.err < self.tolerance:
                        break
                # Update gradient
                gradients = tape.gradient(self.err, self.projection.trainable_variables)
                gradients = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients]

                self.opt.apply_gradients(zip(gradients, self.projection.trainable_variables))
        else:
            concatenated_input = tf.concat([state, alphas], axis=1)
            a = self.input_layer(concatenated_input)
            for layer in self.hidden_layers:
                a = layer(a)
            a = self.output_layer(a)
            safe_a = self.projection(a) + a
            safe_a = tf.clip_by_value(safe_a, -1, 1)

        return safe_a * self.action_scale + self.action_bias

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