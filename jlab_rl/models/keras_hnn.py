import tensorflow as tf

class MLP(tf.keras.Model):

  def __init__(self, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()

    self.dense1 = tf.keras.layers.Dense(hidden_dim, activation=nonlinearity)
    self.dense2 = tf.keras.layers.Dense(hidden_dim, activation=nonlinearity)
    self.dense3 = tf.keras.layers.Dense(output_dim, use_bias=False, activation='linear')

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    return self.dense3(x)

  def time_derivative(self, x, t=None, separate_fields=False):
    return self(x)

