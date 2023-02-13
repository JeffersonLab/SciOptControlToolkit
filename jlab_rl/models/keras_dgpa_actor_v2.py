# Authors:  Malachi Schram, Steven
# Script: SNGP regression TF
# Org: TJNAF, UC

import jlab_datascience_dev.keras_models.gaussian_process_layer as gpl
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import RandomUniform


# class SetNoiseCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         var = self.model.get_variance()
#         self.model.get_layer('gp').set_noise_scale(var)
#
#
# class ResetCovarianceCallback(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         if epoch > 0:
#             self.model.get_layer('gp').reset_prior()


class Keras_Actor_DGPA(tf.keras.Model):
    def __init__(
            self,
            hidden_size,
            num_inputs,
            num_outputs,
            fourier_dim,
            upper_bound,
            lower_bound,
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
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.layer_std = 1.0 / np.sqrt(float(hidden_size))
        #self.layer_std = 1.0 / np.sqrt(float(self.num_inputs))

        # Layer 1
        self.l1 = tf.keras.layers.Dense(self.hidden_size,
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

        # Output
        # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # self.mean_pred = tf.keras.layers.Dense(self.num_outputs, activation="tanh",
        #                                        kernel_initializer=last_init, use_bias=True)
        #
        self.mean_activation = tf.keras.layers.Activation(tf.nn.tanh)
        # Rescale for tanh [-1,1]
        self.scaler = tf.keras.layers.Lambda(
            lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)

    # def train_step(self, data):
    #     x, y = data
    #
    #     with tf.GradientTape() as tape:
    #         y_pred, y_std, _ = self(x, training=True)
    #
    #         # loss_model = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #
    #         # Calculate the log loss
    #         term1 = tf.math.log(2 * np.pi * ((y_std + 1) ** 2))
    #         term2 = (y_pred - y) ** 2 / ((y_std + 1) ** 2)
    #         loss_gp = tf.math.reduce_mean(0.5 * (term1 + term2))
    #
    #         loss = loss_gp  # loss_model + loss_gp
    #
    #     self.update_variance(y, y_pred)
    #
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #
    #     self.compiled_metrics.update_state(y, y_pred)
    #     return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training):
        x = inputs

        # Block 1
        x1 = self.l1(x)
        x1 = self.r1(x1)

        # Block 2
        x2 = self.l2(x1)
        x2 = self.r2(x2)

        # TODO: need to make a gp per output
        # for a in range(nactions):
        output, stddevs, ffs = self.gp(x2)

        # Apply tanh to the output
        output = self.mean_activation(output)
        #output = self.mean_pred(output)

        # Apply boundaries
        output = self.scaler(output)

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

    # def get_variance(self):
    #     momentum = 0.9
    #     var = tf.Variable((1 - momentum) * self.var + momentum * self.old_var)
    #     self.old_var.assign(self.var)
    #     self.mean.assign(0.0)
    #     self.var.assign(0.0)
    #     self.k.assign(0.0)
    #     return var

    # def fit(self, *args, **kwargs):
    #     kwargs["callbacks"] = list(kwargs.get("callbacks", []))
    #     kwargs["callbacks"].append(SetNoiseCallback())
    #     kwargs["callbacks"].append(ResetCovarianceCallback())
    #
    #     return super().fit(*args, **kwargs)
