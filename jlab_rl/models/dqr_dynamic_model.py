# Authors:  Malachi Schram
# Script: MLP dynamic model for model-based RL
# Org: TJNAF

import tensorflow as tf
import numpy as np


class DQR_DynamicModel(tf.keras.Model):
    def __init__(self, hidden_size=32, ndlayers=5, nrff=0, dropout_percent=0.1,
                 quantiles=[0.001, 0.023, 0.159, 0.5, 0.841, 0.977, 0.999]):
        super(DQR_DynamicModel, self).__init__()

        # Setup input parameters
        self.hidden_size = hidden_size
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.ndlayers = ndlayers
        self.output_size = 3
        self.nrff = nrff
        self.dropout_percent = dropout_percent

        self.mse = tf.keras.losses.MeanSquaredError()
        # Setup layers
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dense_layer0 = tf.keras.layers.Dense(self.hidden_size, activation="relu", name='d1')
        self.dense_layers, self.mc_layers = [], []
        for layer in range(self.ndlayers):
            self.dense_layers.append(
                tf.keras.layers.Dense(self.hidden_size, activation="relu", name='dense{}'.format(layer)))
            self.mc_layers.append(tf.keras.layers.Dropout(self.dropout_percent))

        # Add RFF layer ?
        self.rff_scale = tf.Variable(0.01, constraint=lambda z: tf.clip_by_value(z, 0.001, 0.01))
        if self.nrff > 0:
            self.rff_map = tf.keras.layers.Dense(self.nrff,
                                                 trainable=False,
                                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                       stddev=1.0),
                                                 bias_initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
                                                 name='rff_map'
                                                 )

        # Setup quantile layers
        self.quantile_layers = []
        for i in range(len(self.quantiles)):
            q = self.quantiles[i]
            self.quantile_layers.append(tf.keras.layers.Dense(self.output_size, name="{}_q{}".format(i, int(q * 100))))

    # @tf.function
    def call(self, inputs):
        inputs = self.flatten(inputs)
        out = self.dense_layer0(inputs)
        # Loop over layers
        for layer in range(self.ndlayers):
            x = self.dense_layers[layer](out)
            x = self.mc_layers[layer](x)
            out += x

        if (self.nrff > 0):
            out = self.rff_scale * out
            y = self.rff_map(out)
            y1 = tf.math.cos(y)
            y2 = tf.math.sin(y)
            out = tf.keras.layers.concatenate([y1, y2])

        outputs = []
        for i in range(len(self.quantile_layers)):
            outputs.append(self.quantile_layers[i](out))
        return outputs

    # @tf.function
    def train_step(self, data):
        x, y = data
        losses = []
        with tf.GradientTape() as tape:
            y_pred = self(x)
            for i in range(len(self.quantiles)):
                q = self.quantiles[i]
                error = tf.subtract(y, y_pred[i])
                losses.append(tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1))
        # Compute gradients
        gradients = tape.gradient(losses, self.trainable_variables)
        del tape

        # Apply gradients to variables
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': self.mse(y, y_pred)}

    def test_step(self, data):
        x, y = data
        y_pred = self.call(x)
        return {'loss': self.mse(y, y_pred)}

    # Assuming Gaussian uncertainty for this prediction
    def pred_uq(self, input):
        x = input
        predictions = self(x)
        y_preds = predictions[3]
        y_pred_stds1_1 = tf.abs(predictions[2] - predictions[3])
        y_pred_stds1_2 = tf.abs(predictions[4] - predictions[3])
        y_pred_stds = (y_pred_stds1_1 + y_pred_stds1_2) / 2.0
        return y_preds, y_pred_stds