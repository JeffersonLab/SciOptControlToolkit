import tensorflow as tf

class ConstraintCircleGenerator(tf.keras.Model):

  def __init__(self, ndims=2, nlayers=5, lower_bound=None, upper_bound=None):
    super().__init__()
    self.ndims = ndims
    self.nlayers = nlayers
    nmul = 4
    self.denses = []
    self.mc_drops = []
    for i in range(self.nlayers):
      self.denses.append(tf.keras.layers.Dense(nmul*128, activation=tf.nn.leaky_relu))
      self.mc_drops.append(tf.keras.layers.Dropout(0.2))
    self.out = tf.keras.layers.Dense(self.ndims, activation='tanh')
    self.upper_bound = upper_bound
    self.lower_bound = lower_bound

  def call(self, inputs):
    rdm_gaussians = inputs
    # Create some sample
    normalization = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(rdm_gaussians), axis=1))
    repeat_normalization = tf.repeat(tf.expand_dims(normalization, axis=1), self.ndims+2, axis=1)
    rdm_circle_all = tf.math.divide(rdm_gaussians[:], repeat_normalization)
    rdm_circle = rdm_circle_all[:,:-2]

    # Push through the dense layers
    x = self.denses[0](rdm_circle_all)
    x = self.mc_drops[0](x)
    for i in range(1, self.nlayers):
      x = self.denses[i](x)
      x = self.mc_drops[i](x)
    x = self.out(x)
    x = x + rdm_circle
    # Rescale for tanh [-1,1]
    if self.lower_bound.all() != None:
      x = tf.keras.layers.Lambda(
        lambda xi: ((xi + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(x)
    return x