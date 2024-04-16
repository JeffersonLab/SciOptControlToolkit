import unittest
import numpy as np
import os

import jlab_opt_control.models
from jlab_opt_control.utils.sindy_utils import PolynomialLibrary
import tensorflow as tf

class MyTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        # Create data following parameters from UQ-SINDy paper
        self.X = rng.normal(loc=0., scale=1., size=[400, 10])
        print('self.X:',self.X.shape)
        eps = rng.normal(loc=0., scale=0.25, size=[400, 1])
        self.beta = np.array([[0.3, 0.2, -0.3, 0, 0, 0, 0, 0, 0, 0]]).T
        print('self.beta:',self.beta.shape)
        self.y0 = self.X @ self.beta + eps
        print('self.y0:',self.y0.shape)

    def test_numpy_lstsq(self):
        beta = np.linalg.lstsq(self.X, self.y0, rcond=None)[0]
        self.assertTrue(np.allclose(beta, self.beta, atol=5e-2))

    # def test_sindy_network(self):
    #     # Setup model
    #     model = jlab_opt_control.models.make('sindy_network-v0', logdir='results/test')
    #
    #     # Setup optimizers
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, epsilon=1e-8)
    #
    #     # Setup library
    #     library = PolynomialLibrary(degree=1, include_bias=False)
    #     library.fit(tf.zeros([1, self.beta.shape[0]]))
    #
    #     # Run through model once to initialize variables
    #     model(tf.zeros([1, library.output_dim_]))
    #
    #     # Create loss function
    #     mse_loss = tf.keras.losses.MeanSquaredError()
    #
    #     # Train model
    #     n_steps = 100
    #     for step in range(n_steps):
    #         X_batch = tf.convert_to_tensor(self.X, dtype=tf.float32)
    #         y0_batch = tf.convert_to_tensor(self.y0, dtype=tf.float32)
    #         lib_batch = library(X_batch)
    #         with tf.GradientTape() as tape:
    #             y1 = model(lib_batch)
    #             loss = mse_loss(y0_batch, y1)
    #
    #         gradients = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     beta = model.coefs.numpy()
    #     self.assertTrue(np.allclose(beta, self.beta, atol=5e-2))

    def test_uqsindy_network(self):
        # Convert everything to tensors
        print(f'self.X: {self.X.shape}')
        X_batch = tf.convert_to_tensor(self.X, dtype=tf.float32)
        y0_batch = tf.convert_to_tensor(self.y0, dtype=tf.float32)
        print(f'X_batch: {X_batch.shape}')
        print(f'y0_batch: {y0_batch.shape}')

        # Setup library
        num_poly = 4
        library = PolynomialLibrary(degree=num_poly, include_bias=False)
        library.fit(X_batch)
        lib_batch = library(X_batch)
        print(f'lib_batch: {library.output_dim_}')
        print(f'lib_batch: {lib_batch.shape}')

        # Setup model
        model = jlab_opt_control.models.make('uqsindy_network-v0',
                                             num_features_in=library.output_dim_,
                                             num_features_out=y0_batch.shape[1],
                                             batch_size = self.X.shape[0],
                                             logdir='results/test')

        # Setup optimizers
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, epsilon=1e-8)

        # Run through model once to initialize variables
        model(lib_batch)

        # Train model
        n_steps = 100
        for step in range(n_steps):
            with tf.GradientTape() as tape:
                neg_log_p_Xy = model.negative_log_likelihood(lib_batch, y0_batch)
                loss = tf.reduce_mean(neg_log_p_Xy) + model.kld()
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        beta = tf.reduce_mean(model.sample_posterior(batch_size=10), axis=0).numpy()
        print(f'self.beta: {self.beta.shape}')
        print(f'beta: {beta.shape}')
        #self.assertTrue(np.allclose(beta, self.beta, atol=5e-2))

if __name__=='__main__':
    unittest.main()