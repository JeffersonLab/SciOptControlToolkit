import unittest
import numpy as np
import os

from jlab_opt_control.utils.sindy_utils import FourierLibrary
import tensorflow as tf

class MyTestCase(unittest.TestCase):
    def test_fourier_library(self):
        rng = np.random.default_rng(1)

        x = rng.normal(loc=0., scale=1., size=[10, 2])
        lib_sincos = FourierLibrary(include_sin=True, include_cos=True, n_frequencies=2)
        lib_sin    = FourierLibrary(include_sin=True, include_cos=False, n_frequencies=2)
        lib_cos    = FourierLibrary(include_sin=False, include_cos=True, n_frequencies=2)

        y_sincos = lib_sincos.fit_transform(x).numpy()
        y_sin    = lib_sin.fit_transform(x).numpy()
        y_cos    = lib_cos.fit_transform(x).numpy()

        # Test sin/cos library 
        lib = FourierLibrary(include_sin=True, include_cos=True, n_frequencies=2)
        y = lib.fit_transform(x).numpy()
        print('Feature names: ', lib.get_feature_names())
        self.assertTrue(np.allclose(y[:,0], np.sin(1*x[:,0])))
        self.assertTrue(np.allclose(y[:,1], np.sin(2*x[:,0])))
        self.assertTrue(np.allclose(y[:,2], np.sin(1*x[:,1])))
        self.assertTrue(np.allclose(y[:,3], np.sin(2*x[:,1])))

        self.assertTrue(np.allclose(y[:,4], np.cos(1*x[:,0])))
        self.assertTrue(np.allclose(y[:,5], np.cos(2*x[:,0])))
        self.assertTrue(np.allclose(y[:,6], np.cos(1*x[:,1])))
        self.assertTrue(np.allclose(y[:,7], np.cos(2*x[:,1])))

        # Test sin library 
        lib = FourierLibrary(include_sin=True, include_cos=False, n_frequencies=2)
        y = lib.fit_transform(x).numpy()
        print('Feature names: ', lib.get_feature_names())
        self.assertTrue(np.allclose(y[:,0], np.sin(1*x[:,0])))
        self.assertTrue(np.allclose(y[:,1], np.sin(2*x[:,0])))
        self.assertTrue(np.allclose(y[:,2], np.sin(1*x[:,1])))
        self.assertTrue(np.allclose(y[:,3], np.sin(2*x[:,1])))
    
        # Test cos library 
        lib = FourierLibrary(include_sin=False, include_cos=True, n_frequencies=2)
        y = lib.fit_transform(x).numpy()
        print('Feature names: ', lib.get_feature_names())
        self.assertTrue(np.allclose(y[:,0], np.cos(1*x[:,0])))
        self.assertTrue(np.allclose(y[:,1], np.cos(2*x[:,0])))
        self.assertTrue(np.allclose(y[:,2], np.cos(1*x[:,1])))
        self.assertTrue(np.allclose(y[:,3], np.cos(2*x[:,1])))


if __name__=='__main__':
    unittest.main()