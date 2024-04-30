import unittest
import numpy as np
import os

from jlab_opt_control.utils.sindy_lib.polynomial_library import PolynomialLibrary
from jlab_opt_control.utils.sindy_lib.fourier_library import FourierLibrary

import tensorflow as tf


class MyTestCase(unittest.TestCase):
    def test_polynomial_library(self):
        rng = np.random.default_rng(1)
        x = rng.normal(loc=0.0, scale=1.0, size=[10, 2])

        lib = PolynomialLibrary(degree=2, include_bias=True)
        y = lib.fit_transform(x).numpy()
        self.assertEqual(lib.output_dim_, 6)

        print("Polynomial feature names: ", lib.get_feature_names())
        self.assertTrue(np.allclose(y[:, 0], x[:, 0]))
        self.assertTrue(np.allclose(y[:, 1], x[:, 1]))
        self.assertTrue(np.allclose(y[:, 2], x[:, 0] * x[:, 0]))
        self.assertTrue(np.allclose(y[:, 3], x[:, 0] * x[:, 1]))
        self.assertTrue(np.allclose(y[:, 4], x[:, 1] * x[:, 1]))
        self.assertTrue(np.allclose(y[:, 5], np.ones_like(x[:, 0])))

    def test_fourier_library(self):
        rng = np.random.default_rng(1)
        x = rng.normal(loc=0.0, scale=1.0, size=[10, 2])

        # Test sin/cos library
        lib = FourierLibrary(include_sin=True, include_cos=True, n_frequencies=2)
        y = lib.fit_transform(x).numpy()
        print("Fourier feature names: ", lib.get_feature_names())
        self.assertTrue(np.allclose(y[:, 0], np.sin(1 * x[:, 0])))
        self.assertTrue(np.allclose(y[:, 1], np.sin(2 * x[:, 0])))
        self.assertTrue(np.allclose(y[:, 2], np.sin(1 * x[:, 1])))
        self.assertTrue(np.allclose(y[:, 3], np.sin(2 * x[:, 1])))

        self.assertTrue(np.allclose(y[:, 4], np.cos(1 * x[:, 0])))
        self.assertTrue(np.allclose(y[:, 5], np.cos(2 * x[:, 0])))
        self.assertTrue(np.allclose(y[:, 6], np.cos(1 * x[:, 1])))
        self.assertTrue(np.allclose(y[:, 7], np.cos(2 * x[:, 1])))

        # Test sin library
        lib = FourierLibrary(include_sin=True, include_cos=False, n_frequencies=2)
        y = lib.fit_transform(x).numpy()
        print("Sin feature names: ", lib.get_feature_names())
        self.assertTrue(np.allclose(y[:, 0], np.sin(1 * x[:, 0])))
        self.assertTrue(np.allclose(y[:, 1], np.sin(2 * x[:, 0])))
        self.assertTrue(np.allclose(y[:, 2], np.sin(1 * x[:, 1])))
        self.assertTrue(np.allclose(y[:, 3], np.sin(2 * x[:, 1])))

        # Test cos library
        lib = FourierLibrary(include_sin=False, include_cos=True, n_frequencies=2)
        y = lib.fit_transform(x).numpy()
        print("Cos feature names: ", lib.get_feature_names())
        self.assertTrue(np.allclose(y[:, 0], np.cos(1 * x[:, 0])))
        self.assertTrue(np.allclose(y[:, 1], np.cos(2 * x[:, 0])))
        self.assertTrue(np.allclose(y[:, 2], np.cos(1 * x[:, 1])))
        self.assertTrue(np.allclose(y[:, 3], np.cos(2 * x[:, 1])))


if __name__ == "__main__":
    unittest.main()
