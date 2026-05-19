"""
TensorFlow version
"""

import jlab_opt_control as jlab_opt_control
import jlab_opt_control.utils.cfg_utils as cfg_utils
from jlab_opt_control.core.model_core import Model
import tensorflow as tf
from tensorflow.keras import layers

import os
import json
import logging
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

uqsindy_log = logging.getLogger("UQ-SINDy")
uqsindy_log.setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")


class UQSINDyNetwork(Model):
    def __init__(
        self,
        num_features_in,
        num_features_out,
        min_action=None,
        max_action=None,
        batch_size=128,
        logdir="./results",
        cfg="uqsindy_network.cfg",
    ):
        super().__init__()

        # Load configuration
        absolute_path = os.path.dirname(__file__)
        relative_path = "../cfgs/"
        full_path = os.path.join(absolute_path, relative_path)
        self.pfn_json_file = os.path.join(full_path, cfg)

        # Read configuration for architecture
        with open(self.pfn_json_file, "r") as f:
            cfg_data = json.load(f)
        self.batch_size = batch_size

        self.using_tanh = False

        hidden_layers = cfg_data.get("hidden_layers", 3)
        nodes_per_layer = cfg_data.get("nodes_per_layer", [512, 512, 512])
        activation_functions = cfg_data.get(
            "activation_functions", ["tanh"] * hidden_layers + ["linear"]
        )

        self.logdir = logdir

        # Error Checking — fail loudly instead of just logging
        if hidden_layers != len(nodes_per_layer):
            raise ValueError(
                f"hidden_layers ({hidden_layers}) does not match "
                f"len(nodes_per_layer) ({len(nodes_per_layer)})"
            )
        if hidden_layers != len(activation_functions) - 1:
            raise ValueError(
                f"hidden_layers ({hidden_layers}) does not match "
                f"len(activation_functions) - 1 ({len(activation_functions) - 1}). "
                f"Expected {hidden_layers + 1} activations (one per hidden layer "
                f"plus output), got {len(activation_functions)}."
            )

        # Parameters — use add_weight so Keras tracks them as trainable variables
        self.mu = self.add_weight(
            name="mu",
            shape=(num_features_in, num_features_out),
            initializer="ones",
            trainable=True,
        )
        self.log_var = self.add_weight(
            name="log_var",
            shape=(num_features_in, num_features_out),
            initializer="ones",
            trainable=True,
        )

        ortho_init = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)
        zeros_init = tf.keras.initializers.Zeros()

        # Network architecture
        total_features = num_features_in * num_features_out
        self.hidden_layers = []
        for i in range(hidden_layers):
            self.hidden_layers.append(
                layers.Dense(
                    nodes_per_layer[i],
                    kernel_initializer=ortho_init,
                    bias_initializer=zeros_init,
                    activation=activation_functions[i],
                )
            )
        # Output layer with its specified activation function
        self.output_layer = layers.Dense(
            total_features,
            kernel_initializer=ortho_init,
            bias_initializer=zeros_init,
            activation=activation_functions[-1],
        )

    def plot_coefficients(self, feature_names, action_names):
        """Plot coefficient distribution using Box-Whisker plot"""
        weight_dist = self.sample_posterior()

        assert len(feature_names) == weight_dist.shape[1]
        assert len(action_names) == weight_dist.shape[2], \
            f"number weights: {weight_dist.shape[2]}"

        df = []
        for i, action in enumerate(action_names):
            df.append(pd.DataFrame(weight_dist[:, :, i], columns=feature_names))
            df[-1]["action"] = action
        df = pd.concat(df, ignore_index=True).reset_index()
        df = pd.melt(
            df,
            id_vars=["index", "action"],
            value_vars=feature_names,
            var_name="Term",
            value_name="Coefficient",
        )

        fig, ax = plt.subplots(dpi=150)
        sns.boxplot(
            data=df,
            x="Coefficient",
            y="Term",
            hue="action",
            whis=(0, 100),
            ax=ax,
        )
        ax.axvline(0, color="grey", zorder=-10)
        plt.tight_layout()

        return fig

    @tf.function
    def sample_posterior(self, sampling_size=1024):
        # Draw from latent distribution
        randn = tf.random.normal(
            shape=(sampling_size, *self.mu.shape), dtype=tf.float32
        )
        betas = self.mu[None] + randn * tf.exp(0.5 * self.log_var[None])

        # Flatten last layer for use in neural network
        b, Ni, No = betas.shape
        betas = tf.reshape(betas, [b, Ni * No])

        for layer in self.hidden_layers:
            betas = layer(betas)
        betas = self.output_layer(betas)

        # Reshape last layer
        betas = tf.reshape(betas, [b, Ni, No])
        return betas

    @tf.function
    def call(self, inputs, nsamples=0, training=False):
        """forward pass of model"""
        if nsamples == 0:
            nsamples = self.batch_size
        x = inputs
        betas = self.sample_posterior(nsamples)
        BX = tf.einsum("nd,bdo->bno", x, betas)

        return BX

    def negative_log_likelihood(self, x, y0):
        """Log likelihood of data given parameter distribution"""
        BX = self(x)
        log_p_x = -0.5 * tf.reduce_sum(tf.square(x), axis=-1)[None]
        log_p_y = -0.5 * tf.reduce_sum(tf.square(y0 - BX), axis=-1)
        log_p_Xy = tf.reduce_sum(log_p_x + log_p_y, axis=1)
        return -log_p_Xy

    def kld(self):
        """KL-Divergence of latent space from unit normal prior"""
        kld = 0.5 * tf.reduce_sum(
            self.mu**2 + tf.exp(self.log_var) - self.log_var - 1
        )
        return kld

    def save_cfg(self):
        """Save the model cfg"""
        try:
            destination_file_path = os.path.join(self.logdir, "cfgs/")
            if not os.path.exists(destination_file_path):
                os.makedirs(destination_file_path)
            destination_file_path = os.path.join(
                destination_file_path, os.path.basename(self.pfn_json_file)
            )
            if not os.path.exists(destination_file_path):
                shutil.copy(self.pfn_json_file, destination_file_path)
                uqsindy_log.info("SINDy model config saved successfully")
        except:
            uqsindy_log.error("Error in saving the actor model cfg...")