import tensorflow as tf

class ConstraintCircleGenerator(tf.keras.Model):

  def __init__(self, ndims=2, nlayers=5, lower_bound=None, upper_bound=None):
    super().__init__()
    self.ndims = ndims
    self.nlayers = nlayers
    init = tf.keras.initializers.GlorotUniform()
    nmul = 4
    self.denses1 = []
    self.bn1 = []
    self.act1 = []
    for i in range(self.nlayers):
      self.denses1.append(tf.keras.layers.Dense(nmul*32, kernel_initializer=init))
      self.bn1.append(tf.keras.layers.BatchNormalization())
      self.act1.append(tf.keras.layers.LeakyReLU(0.2))

    self.denses2 = []
    self.bn2 = []
    self.act2 = []
    for i in range(self.nlayers):
      self.denses2.append(tf.keras.layers.Dense(nmul*32, kernel_initializer=init))
      self.bn2.append(tf.keras.layers.BatchNormalization())
      self.act2.append(tf.keras.layers.LeakyReLU(0.2))

    self.out1 = tf.keras.layers.Dense(self.ndims, kernel_initializer=init,
                                      activation=tf.nn.leaky_relu)
    self.out2 = tf.keras.layers.Dense(self.ndims, kernel_initializer=init,
                                      activation=tf.nn.leaky_relu)
    self.out = tf.keras.layers.Dense(self.ndims, kernel_initializer=init,
                                     activation='tanh')
    self.upper_bound = upper_bound
    self.lower_bound = lower_bound

  def call(self, inputs):
    states, rdm_gaussians = inputs

    # Find random samples in circle phase space
    normalization = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(rdm_gaussians), axis=1))
    repeat_normalization = tf.repeat(tf.expand_dims(normalization, axis=1), self.ndims+2, axis=1)
    rdm_circle_all = tf.math.divide(rdm_gaussians[:], repeat_normalization)
    rdm_circle = rdm_circle_all[:,:-2]

    # Random mask
    x1 = rdm_circle_all
    for i in range(0, self.nlayers):
      x1 = self.denses1[i](x1)
      x1 = self.bn1[i](x1)
      x1 = self.act1[i](x1)
    x1 = self.out1(x1)

    # Add the rdm circle variables and state plots the generative prediction
    x2 = tf.keras.layers.concatenate([states, rdm_circle])
    for i in range(0, self.nlayers):
      x2 = self.denses2[i](x2)
      x2 = self.bn2[i](x2)
      x2 = self.act2[i](x2)
    x2 = self.out2(x2)

    x = tf.keras.layers.concatenate([x1, x2])
    x = self.out(x)

    # Rescale using tanh [-1,1] to ensure it's within the parameter space
    # if self.lower_bound.all() != None:
    # x = tf.keras.layers.Lambda(
    #   lambda xi: ((xi + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(x+states)

    return x