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

sindy_log = logging.getLogger("SINDy")
sindy_log.setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")


class SINDyNetwork(Model):
    def __init__(
            self, 
            num_features_in, # From environment and library
            num_features_out, # From environment
            logdir='./results'):
        super().__init__()

        self.logdir = logdir
        self.coefs = tf.Variable(
            initial_value=tf.zeros([num_features_in, num_features_out]),
            trainable=True,
        )

    def plot_coefficients(self, feature_names, action_names=None):
        """Plot coefficients as barplot"""
        assert len(feature_names) == self.coefs.shape[0]

        # Use Pandas dataframe to collect data
        df = pd.DataFrame(data=self.coefs.numpy().T, columns=feature_names)
        if action_names is not None:
            assert len(action_names) == self.coefs.shape[1]
            df["action"] = action_names
            df = pd.melt(
                df,
                id_vars=["action"],
                value_vars=feature_names,
                var_name="Term",
                value_name="Coefficient",
            )

        # Generate coefficients barplot using Seaborn
        fig, ax = plt.subplots(dpi=150)
        sns.barplot(data=df, x="Coefficient", y="Term", hue="action" if action_names is not None else None, ax=ax)
        ax.axvline(0, color="grey", zorder=-10)
        plt.tight_layout()

        return fig

    def call(self, inputs, training=False):
        """forward pass of model"""
        x = inputs
        x = tf.matmul(x, self.coefs)
        return x

    def save_cfg(self):
        """Save the model cfg"""
        try:
            sindy_log.info("SINDy model requires no configuration, not saving")
        except:
            sindy_log.error("Error in saving the actor model cfg...")
