import tensorflow as tf


class PolynomialLibrary(tf.keras.layers.Layer):
    """
    Library of polynomial functions up to a specified degree
    """

    def __init__(self, degree=3, include_bias=False):
        super().__init__()
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, x, y0=None):
        self.fit_predict(x, y0)
        return self

    def fit_predict(self, x, y0=None):
        y = self.call(x)
        self.input_dim_ = x.shape[-1]
        self.output_dim_ = y.shape[-1]
        return y

    @tf.function
    def call(self, x):
        B, L = x.shape
        library = []
        if self.include_bias:
            library.append(tf.ones((B,), dtype=tf.float32))

        if self.degree > 0:
            for i in range(L):
                library.append(x[:, i])

        if self.degree > 1:
            for i in range(L):
                for j in range(i, L):
                    library.append(x[:, i] * x[:, j])

        if self.degree > 2:
            for i in range(L):
                for j in range(i, L):
                    for k in range(j, L):
                        library.append(x[:, i] * x[:, j] * x[:, k])

        if self.degree > 3:
            for i in range(L):
                for j in range(i, L):
                    for k in range(j, L):
                        for l in range(k, L):
                            library.append(x[:, i] * x[:, j] * x[:, k] * x[:, l])

        return tf.stack(library, axis=1)  # [B, L]

class FourierLibrary(tf.keras.layers.Layer):
    """
    Library of fourier functions up to a specified number of frequencies
    Generally following the scikit-learn Transformer style
    """

    def __init__(self, include_sin=True, include_cos=True, n_frequencies=1):
        super().__init__()

        # Input validation
        if not (include_sin or include_cos):
            raise ValueError("include_sin and include_cos cannot both be false")
        if n_frequencies < 1 or not isinstance(n_frequencies, int):
            raise ValueError("n_frequencies must be a positive integer")

        self.include_sin = include_sin
        self.include_cos = include_cos
        self.n_frequencies = n_frequencies

    def fit(self, x):
        self.input_dim_ = x.shape[-1]
        if self.include_sin and self.include_cos:
            self.output_dim_ = 2 * self.n_frequencies * x.shape[-1]
        else:
            self.output_dim_ = self.n_frequencies * x.shape[-1]
        return self

    def transform(self, x):
        return self.call(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def get_feature_names(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.input_dim_)]
        assert len(input_features) == self.input_dim_

        freqs = []
        for feat in input_features:
            for i in range(1, self.n_frequencies + 1):
                freqs.append(f"{i} {feat}")

        feature_names = []
        if self.include_sin:
            feature_names = feature_names + [f"sin {freq}" for freq in freqs]
        if self.include_cos:
            feature_names = feature_names + [f"cos {freq}" for freq in freqs]

        return feature_names

    def call(self, x):
        n_samples, n_features = x.shape
        assert n_features == self.input_dim_

        freqs = tf.range(1, self.n_frequencies + 1, dtype=x.dtype)
        xf = tf.einsum("bl,f->blf", x, freqs)
        xf = tf.reshape(xf, [n_samples, self.n_frequencies * n_features])

        library = []
        if self.include_sin:
            library.append(tf.sin(xf))
        if self.include_cos:
            library.append(tf.cos(xf))

        library = tf.concat(library, axis=1)  # [n_samples, n_features]
        return library

