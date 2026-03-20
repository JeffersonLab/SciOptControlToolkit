from jlab_opt_control.core.sindy_lib_core import SINDyLibrary
import tensorflow as tf
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
import numpy as np


class PolynomialLibrary(SINDyLibrary):
    """
    Library of polynomial functions up to a specified degree
    """

    def __init__(
        self,
        degree=3,
        include_interaction=True,
        interaction_only=False,
        include_bias=True,
    ):
        super().__init__()

        # Input validation
        if degree < 0 or not isinstance(degree, int):
            raise ValueError("degree must be a positive integer")

        self.degree = degree
        self.include_interaction = include_interaction
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def _combinations(self):
        # Determine combinations as in sklearn.preprocessing.PolynomialFeatures
        if self.include_interaction:
            comb = combinations if self.interaction_only else combinations_w_r
            _combinations = chain.from_iterable(
                comb(range(self.input_dim_), i) for i in range(1, self.degree + 1)
            )
        else:
            _combinations = chain(
                (
                    exp * (feat_idx,)
                    for exp in range(1, self.degree + 1)
                    for feat_idx in range(self.input_dim_)
                )
            )

        return _combinations

    def fit(self, x):
        self.input_dim_ = x.shape[-1]

        self.output_dim_ = sum(1 for _ in self._combinations())
        if self.include_bias:
            self.output_dim_ += 1

        return self

    def get_feature_names(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.input_dim_)]
        assert len(input_features) == self.input_dim_

        feature_names = []
        powers = np.vstack(
            [np.bincount(c, minlength=self.input_dim_) for c in self._combinations()]
        )
        for row in powers:
            inds = np.where(row)[0]
            name = " ".join(
                f"{input_features[ind]}^{exp}" if exp != 1 else input_features[ind]
                for ind, exp in zip(inds, row[inds])
            )
            feature_names.append(name)

        if self.include_bias:
            feature_names.append("1")

        return feature_names

    def call(self, x):
        n_samples, n_features = x.shape
        assert n_features == self.input_dim_

        library = []
        for i, comb in enumerate(self._combinations()):
            term = tf.gather(x, comb, axis=-1)
            term = tf.math.reduce_prod(term, -1, keepdims=True)
            library.append(term)

        if self.include_bias:
            library.append(tf.ones([n_samples, 1], dtype=x.dtype))

        library = tf.concat(library, axis=1)  # [n_samples, n_features]
        return library
