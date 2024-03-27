from abc import ABC, abstractmethod
import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ Define all key variables required for all model """

    @abstractmethod
    def call(self, inputs, training=False):
        """ forward pass of model """
        pass

    @abstractmethod
    def save_cfg(self):
        """ Save the model cfg """
        pass
