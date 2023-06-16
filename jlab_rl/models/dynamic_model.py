# Authors:  Malachi Schram
# Script: MLP dynamic model for model-based RL
# Org: TJNAF

import numpy as np
import tensorflow as tf


class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, rate, switch=True):
        if switch:
            return tf.nn.dropout(inputs, rate=rate)
        else:
            return inputs


class DynamicModel(tf.keras.Model):
    def __init__(self, ndlayers, hidden_size, drop_value, output_size, nrff):
        super(DynamicModel, self).__init__()

        self.ntrials = 15
        self.ndlayers = ndlayers
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.nrff = nrff
        self.lower_uq = 1


        # Noise related
        self.drop_percent = tf.Variable(drop_value, constraint=lambda z: tf.clip_by_value(z, 0.001, 0.005))
        # Layers
        self.dense_layer0 = tf.keras.layers.Dense(self.hidden_size, activation="relu", name='d1')
        self.dense_layers, self.mc_layers = [], []
        for layer in range(self.ndlayers):
            self.dense_layers.append(tf.keras.layers.Dense(self.hidden_size,
                                                           activation="relu", name='dense{}'.format(layer)))
            self.mc_layers.append(CustomDropout(self.drop_percent))

        if (self.nrff > 0):
            # RFF scale
            self.rff_scale = tf.Variable(drop_value, constraint=lambda z: tf.clip_by_value(z, 0.001, 0.05))
            # RFF Scale
            self.rff_map = tf.keras.layers.Dense(self.nrff,
                                                 trainable=False,
                                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                       stddev=1.0),
                                                 bias_initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
                                                 name='rff_map'
                                                 )
        # Predict the next state and the reward
        self.output_layer = tf.keras.layers.Dense(self.output_size, activation="linear", name='output')

    def call(self, inputs):
        out = self.dense_layer0(inputs)
        # Loop over layers
        for layer in range(self.ndlayers):
            x = self.dense_layers[layer](out)
            x = self.mc_layers[layer](x, self.drop_percent)
            out += x

        if (self.nrff > 0):
            out = self.rff_scale * out
            y = self.rff_map(out)
            y1 = tf.math.cos(y)
            y2 = tf.math.sin(y)
            y_out = tf.keras.layers.concatenate([y1, y2])
            return self.output_layer(y_out)

        out = self.output_layer(out)
        return out

    @tf.function
    def get_loss(self, y_preds, y_pred_stds, y):
        diff_sqrt_term = tf.math.squared_difference(y_preds, y)
        diff_sqrt_term = 0.5 * diff_sqrt_term / (tf.math.square(y_pred_stds) + self.lower_uq)
        std_term = 0.5 * tf.math.log(tf.math.square(y_pred_stds) + self.lower_uq)
        return tf.math.reduce_mean(diff_sqrt_term + std_term)

    # TODO: Needs to be optimized!!!
    def predict_uq(self, input):
        x = input
        predictions = [self(x) for _ in range(self.ntrials)]
        y_preds = tf.reduce_mean(predictions, axis=0)
        y_pred_stds = tf.math.reduce_std(predictions, axis=0)
        return y_preds, y_pred_stds

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_preds, y_pred_stds = self.predict_uq(x)
            loss = self.get_loss(y_preds, y_pred_stds, y)

        grad = tape.gradient(loss, self.trainable_variables)
        del tape

        # Apply gradients to variables
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {'loss': loss, 'mse': tf.reduce_mean((y - y_preds) ** 2)}

    def test_step(self, data):
        x, y = data
        y_preds, y_stds = self.pred_uq(x)
        loss = self.get_loss(y_preds, y_stds, y)
        return {'loss': loss, 'mse': tf.reduce_mean((y - y_preds) ** 2), }