# Authors:  Malachi Schram, Steven
# Script: SNGP regression TF
# Org: TJNAF, UC

import jlab_datascience_dev.keras_models.gaussian_process_layer as gpl
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import RandomUniform


class KerasCriticDGPA(tf.keras.Model):
    def __init__(
            self,
            hidden_size,
            num_inputs,
            num_outputs,
            fourier_dim,
            noise_scale=1.0,
            length_scale=1.0,
            train_length_scale=True,
            train_gp=True,
    ):
        super().__init__()

        self.model_prior = tf.eye(fourier_dim) * noise_scale
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layer_std = 1.0 / np.sqrt(float(hidden_size))

        # Layer 1
        self.l1 = tf.keras.layers.Dense(self.num_inputs,
                                        kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
                                        bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))
        self.r1 = tf.keras.layers.Activation(tf.nn.leaky_relu)

        # Layer 2
        self.l2 = tf.keras.layers.Dense(self.hidden_size,
                                        kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
                                        bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))
        self.r2 = tf.keras.layers.Activation(tf.nn.leaky_relu)

        self.gp = gpl.GaussianProcessLayer(
            n_fourier_features=fourier_dim,
            n_out=num_outputs,
            trainable=train_gp,
            noise_scale=noise_scale,
            length_scale=length_scale,
            train_length_scale=train_length_scale,
            do_custom_cov_update=True,
            name='gp'
        )

        self.mean = tf.Variable(0.0, trainable=False, name='err_mean')
        self.var = tf.Variable(0.0, trainable=False, name='err_var')
        self.old_var = tf.Variable(0.0, trainable=False, name='old_err_var')
        self.k = tf.Variable(0.0, trainable=False, name='err_k')

    def set_variables(self, mean, var, old_var, k):
        self.mean = mean
        self.var = var
        self.old_var = old_var
        self.k = k

    def call(self, inputs, training):
        x = inputs

        # Block 1
        x1 = self.l1(x)
        x1 = self.r1(x1)

        # Block 2
        x2 = self.l2(x1)
        x2 = self.r2(x2)

        output, stddevs, ffs = self.gp(x2)

        return output, stddevs, ffs

    def update_variance(self, y, y_pred):
        n_samples = tf.cast(tf.shape(y)[0], tf.float32)

        err = tf.reshape(y_pred, tf.shape(y)) - y

        old_mean = self.mean
        old_var = self.var

        new_mean = tf.math.reduce_mean(err)
        new_var = tf.math.reduce_variance(err)

        first_term = old_mean * (self.k / (self.k + n_samples))
        second_term = new_mean * (n_samples / (self.k + n_samples))
        self.mean.assign(first_term + second_term)

        first_term = old_var * (self.k / (self.k + n_samples))
        second_term = new_var * (n_samples / (self.k + n_samples))
        third_term = (self.k * n_samples) / (self.k + n_samples) ** 2
        fourth_term = (old_mean - new_mean) ** 2
        self.var.assign(first_term + second_term + (third_term * fourth_term))

        self.k.assign_add(n_samples)
