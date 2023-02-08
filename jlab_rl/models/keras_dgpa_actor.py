# Authors:  Malachi Schram, Kishan Rajput, Karthik
# Script: SNGP regression TF
# Org: TJNAF, UC

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform

def error_test(y, y_pred, y_std, c):
    in_1sigma = [y[i] for i in range(y.shape[0]) if
                 y[i] > (y_pred[i] - c * y_std[i]) and y[i] < (y_pred[i] + c * y_std[i])]
    ratio = len(in_1sigma) / y.shape[0]
    ratio_error = abs(ratio - 0.68)
    return ratio_error

from scipy.spatial import distance

def euclidean_dist(x):
    return np.ndarray.astype(distance.pdist(x).flatten(), np.float32)

def euclidean_dist_v2(x):
    dist = 0
    for i in range(x.shape[1]):
        if i == 0:
            dist = np.ndarray.astype(distance.pdist(x[:, i, :]).flatten(), np.float32)
        else:
            dist += np.ndarray.astype(distance.pdist(x[:, i, :]).flatten(), np.float32)
    return dist / x.shape[1]

class Keras_Actor_DGPA(tf.keras.Model):
    def __init__(self, hidden_size, num_inputs, num_outputs, fourier_dim, upper_bound, lower_bound,
                 lipschitz_bounds=[0.75, 1.25]):
        super(Keras_Actor_DGPA, self).__init__()

        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fourier_dim = fourier_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.l_bounds = lipschitz_bounds

        self.cov = tf.eye(self.fourier_dim)
        self.scale = 1.0
        self.counts = 0

        self.calib = 1.0
        # Network
        self.spec_norm_bound = 10
        """
        num_inputs = Dimension of inputs
        num_outputs = Dimension of Random Fourier feature
        hidden_size = Dimension of hidden features
        """
        self.layer_std = 1.0 / np.sqrt(float(hidden_size))

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # Input
        inputs = tf.keras.layers.Input(shape=(self.num_inputs,))

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

        # Output
        self.mean_pred = tf.keras.layers.Dense(self.num_outputs, activation="tanh",
                                               kernel_initializer=last_init, use_bias=True)

        # Rescale for tanh [-1,1]
        self.scaler = tf.keras.layers.Lambda(
            lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)

        self.W = tf.random.normal([int(fourier_dim / 2), hidden_size], 0, 1)
        self.b = tf.random.uniform([int(fourier_dim / 2), 1], minval=0, maxval=1) * 2.0 * np.pi


    def get_calibration(self, data):
        '''
        '''
        x, y = data
        # tf.print('\t calib x/y/calib:', x.shape, '/', y.shape, '/', self.calib)
        y_pred, y_std = self(x, training=False)
        y_std = y_std / self.calib

        from functools import partial
        tf.print(y[0])
        tf.print(y_pred[0])
        tf.print(y_std[0])
        cost_function = partial(error_test, y, y_pred, y_std)

        from scipy.optimize import minimize
        x0 = 1.0 / self.num_inputs  # initial guess
        res = minimize(cost_function, x0, method='Nelder-Mead', tol=1e-6)
        self.calib = res['x']
        tf.print('\t calibration: ', self.calib)
        return self.calib

    def train_step(self, data):
        ''' defines the logic of a training step '''
        self.counts += 1
        x, y = data

        with tf.GradientTape() as tape:
            # Mean prediction
            y_pred, fourier_pred, hidden_x, input_x = self(x, training=True)
            loss_mu = self.compiled_loss(
                y,
                y_pred,
            )

            hidden_x = (hidden_x - tf.reduce_min(hidden_x)) / (tf.reduce_max(hidden_x) - tf.reduce_min(hidden_x))
            l1, l2 = self.l_bounds
            inp_dist = tf.numpy_function(euclidean_dist_v2, [input_x], tf.float32)
            hidden_dist = tf.numpy_function(euclidean_dist, [hidden_x], tf.float32)

            c1 = l1 * inp_dist - hidden_dist
            c2 = hidden_dist - l2 * inp_dist
            #     constraint: l1*inp_dist <= hidden_dist <= l2*inp_dist
            #     c1: l1*inp_dist - hidden_dist <= 0
            #     c2: hidden_dist - l2*inp_dist <= 0
            loss1 = tf.reduce_mean(tf.nn.relu(c1))
            loss2 = tf.reduce_mean(tf.nn.relu(c2))
            distance_loss = (loss1 + loss2) / 2.0
            ########################################################################

            loss = loss_mu + distance_loss

        # Compute gradients
        grad = tape.gradient(loss, self.trainable_variables)
        del tape

        # Apply gradients to variables
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        # Predictive Covariance update
        phi = tf.stop_gradient(fourier_pred)
        P = tf.linalg.matmul(phi, tf.transpose(phi))
        S = tf.eye(self.fourier_dim) - \
            tf.linalg.matmul(tf.linalg.inv(P + (self.scale ** 2) * tf.eye(self.fourier_dim)), P)
        # Bug is here

        if self.counts > 1:
            self.cov = 0.99 * (self.cov) + 0.01 * tf.linalg.matmul(P, S)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False, return_hidden=False):
        ''' define the forward-pass of your model '''
        # means, sigmas = [],[]
        x = inputs

        # Block 1
        x1 = self.l1(x)
        x1 = self.r1(x1)

        # Block 2
        x2 = self.l2(x1)
        x2 = self.r2(x2)

        y = x2

        # compute projection onto Random Fourier feature space
        y1 = 1. / np.sqrt(self.fourier_dim / 2) * np.sqrt(2) * tf.math.cos(self.W @ tf.transpose(y) + self.b)
        y2 = 1. / np.sqrt(self.fourier_dim / 2) * np.sqrt(2) * tf.math.sin(self.W @ tf.transpose(y) + self.b)

        phis = tf.concat([y1, y2], 0)

        mean_preds = self.mean_pred(tf.transpose(phis))
        mean_preds = self.scaler(mean_preds)

        vars = []
        for i in range(phis.shape[1]):
            phi = tf.expand_dims(phis[:, i], axis=1)
            var = tf.linalg.matmul(tf.transpose(phi), phi) \
                  - tf.linalg.matmul(tf.linalg.matmul(tf.transpose(phi), self.cov), phi)
            # Note that this is now the standard deviation
            vars.append(tf.sqrt(var[0, 0]) * self.calib)
        # tf.print('var:',var)
        if training == True:
            return mean_preds, phis, y, inputs, tf.expand_dims(tf.convert_to_tensor(vars), axis=1)
        else:
            if return_hidden:
                return mean_preds, tf.expand_dims(tf.convert_to_tensor(vars), axis=1), y
            else:
                return mean_preds, tf.expand_dims(tf.convert_to_tensor(vars), axis=1)

        # if training == True:
        #     return mean_preds, phis, y, inputs
        # else:
        #     vars = []
        #     for i in range(phis.shape[1]):
        #         phi = tf.expand_dims(phis[:, i], axis=1)
        #         # tf.print('call phi:', phi.shape)
        #         var = tf.linalg.matmul(tf.transpose(phi), phi) \
        #               - tf.linalg.matmul(tf.linalg.matmul(tf.transpose(phi), self.cov), phi)
        #         # Note that this is now the standard deviation
        #         vars.append(tf.sqrt(var[0, 0]) * self.calib)
        #         # vars.append((var[0, 0]))
        #     if return_hidden:
        #         return mean_preds, tf.expand_dims(tf.convert_to_tensor(vars), axis=1), y
        #     return mean_preds, tf.expand_dims(tf.convert_to_tensor(vars), axis=1)
