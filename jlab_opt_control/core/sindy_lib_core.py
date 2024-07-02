from abc import abstractmethod
import tensorflow as tf
import itertools
import string


class SINDyLibrary(tf.keras.layers.Layer):
    """SINDy library base interface following scikit-learn Transformer style"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ Define all key variables required for library """

    @abstractmethod
    def fit(self, x):
        """Determine parameters such as input and output dimension"""
        pass

    @abstractmethod
    def get_feature_names(self, input_features=None):
        """Give names of features produced by library"""
        pass

    @abstractmethod
    def call(self, X):
        """Differentiable forward function of the Keras layer"""
        pass

    def transform(self, x):
        """Calls library transformation function, included for sklearn compatibility"""
        return self.call(x)

    def fit_transform(self, x):
        """Included for sklearn compatibility"""
        self.fit(x)
        return self.transform(x)

    def __add__(self, x):
        return ConcatLibrary([self, x])

    def __mul__(self, x):
        return ProductLibrary([self, x])

    def __rmul__(self, x):
        return ProductLibrary([self, x])


class ConcatLibrary(SINDyLibrary):
    """Concatenate multiple libraries"""

    def __init__(self, libraries):
        super().__init__()
        self.libraries = libraries

    def fit(self, x):
        self.input_dim_ = x.shape[-1]
        self.output_dim_ = 0

        # Fit all libraries on the input
        # Number of output features is the sum of each library's output dim
        for library in self.libraries:
            library.fit(x)
            self.output_dim_ += library.output_dim_
        return self

    def get_feature_names(self, input_features=None):
        feature_names = []
        for library in self.libraries:
            feature_names += library.get_feature_names(input_features)
        return feature_names

    def call(self, x):
        n_samples, n_features = x.shape
        assert n_features == self.input_dim_

        return tf.concat([library.call(x) for library in self.libraries], axis=-1)


class ProductLibrary(SINDyLibrary):
    def __init__(self, libraries):
        super().__init__()
        self.libraries = libraries

    def fit(self, x):
        self.input_dim_ = x.shape[-1]
        self.output_dim_ = 1
        self.sumstr_ = []

        # Fit all libraries on the input
        # Number of output features is product of each library's output dim
        for i, library in enumerate(self.libraries):
            library.fit(x)
            self.output_dim_ *= library.output_dim_
            self.sumstr_.append("B" + string.ascii_lowercase[i])

        self.sumstr_ = (
            ",".join(self.sumstr_)
            + "->B"
            + string.ascii_lowercase[: len(self.libraries)]
        )

        return self

    def get_feature_names(self, input_features=None):
        feature_names_lib = [
            library.get_feature_names(input_features) for library in self.libraries
        ]
        feature_names = [
            " ".join(prod) for prod in itertools.product(*feature_names_lib)
        ]
        return feature_names

    def call(self, x):
        n_samples, n_features = x.shape
        assert n_features == self.input_dim_

        # Compute each library's output
        feature_lib = [library(x) for library in self.libraries]

        # Compute the products of each library output
        library = tf.einsum(self.sumstr_, *feature_lib)
        library = tf.reshape(library, [n_samples, self.output_dim_])
        return library
