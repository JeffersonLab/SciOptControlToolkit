from abc import abstractmethod
import tensorflow as tf

class SINDyLibrary(tf.keras.layers.Layer):
    """ SINDy library base interface following scikit-learn Transformer style """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ Define all key variables required for library """

    @abstractmethod
    def fit(self, x):
        """ Determine parameters such as input and output dimension """
        pass

    @abstractmethod
    def get_feature_names(self, input_features=None):
        """ Give names of features produced by library """
        pass

    @abstractmethod
    def call(self, X):
        """ Differentiable forward function of the Keras layer """
        pass

    def transform(self, x):
        """ Calls library transformation function, included for sklearn compatibility """
        return self.call(x)

    def fit_transform(self, x):
        """ Included for sklearn compatibility """
        self.fit(x)
        return self.transform(x)