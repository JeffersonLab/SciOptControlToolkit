import tensorflow as tf

class PolynomialLibrary(tf.keras.layers.Layer):
    '''
    Library of polynomial functions up to a specified degree
    '''
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
