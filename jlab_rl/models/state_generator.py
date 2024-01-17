import tensorflow as tf
import time

class Generator(tf.keras.Model):

  def __init__(self, ndims=2, nlayers=5, lower_bound=None, upper_bound=None):
    super().__init__()
    time.sleep(1/10)
    seed = time.time_ns()
    print('Generator seed:', seed)
    tf.random.set_seed(seed)
    self.ndims = ndims
    self.nlayers = nlayers
    init = tf.keras.initializers.GlorotUniform(seed)
    nodes = 256
    #self.rdm_layer = tf.keras.layers.GaussianNoise(stddev=1, seed=seed)
    self.denses1 = []
    self.bn1 = []
    self.act1 = []
    for i in range(self.nlayers):
      self.denses1.append(tf.keras.layers.Dense(nodes, kernel_initializer=init))
      self.bn1.append(tf.keras.layers.BatchNormalization())
      self.act1.append(tf.keras.layers.LeakyReLU(0.2))

    self.denses2 = []
    self.bn2 = []
    self.act2 = []
    for i in range(self.nlayers):
      self.denses2.append(tf.keras.layers.Dense(nodes, kernel_initializer=init))
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
    states, rdm_variables = inputs

    # Random mask
    x1 = rdm_variables
    for i in range(0, self.nlayers):
      x1 = self.denses1[i](x1)
      x1 = self.bn1[i](x1)
      x1 = self.act1[i](x1)
    x1 = self.out1(x1)

    # States
    x2 = states
    #x2 = self.rdm_layer(x2)
    for i in range(0, self.nlayers):
      x2 = self.denses2[i](x2)
      x2 = self.bn2[i](x2)
      x2 = self.act2[i](x2)
    x2 = self.out2(x2)

    x = tf.keras.layers.concatenate([x1, x2])
    x = self.out(x)

    # Rescale using tanh [-1,1] to ensure it's within the parameter space
    # if self.lower_bound.all() != None:
    x = tf.keras.layers.Lambda(
      lambda xi: ((xi + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(x)
    return x
  

# class Generator_v2(tf.keras.Model):
#
#     def __init__(self, ndims=1, nlayers=4, lower_bound=None, upper_bound=None):
#       super().__init__()
#       self.ndims = ndims
#       self.nlayers = nlayers
#
#       self.denses1 = []
#       self.act1 = []
#       for i in range(self.nlayers):
#         self.denses1.append(tf.keras.layers.Dense(128))
#         self.act1.append(tf.keras.layers.LeakyReLU(0.2))
#
#       self.out = tf.keras.layers.Dense(self.ndims,
#                                       activation='sigmoid')
#       self.upper_bound = upper_bound
#       self.lower_bound = lower_bound
#
#     def call(self, inputs):
#       x = tf.concat([inputs[0], inputs[1]], 1)
#       for i in range(0, self.nlayers):
#         x = self.denses1[i](x)
#         x = self.act1[i](x)
#       x = self.out(x)
#
#       # Rescale using tanh [-1,1] to ensure it's within the parameter space
#       # if self.lower_bound.all() != None:
#       # x = tf.keras.layers.Lambda(
#       #   lambda xi: ((xi + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(x)
#       return x

class Generator_v3(tf.keras.Model):

  def __init__(self, ndims=2, nlayers=5, lower_bound=None, upper_bound=None):
    super().__init__()
    seed = time.time_ns()
    tf.random.set_seed(seed)
    self.ndims = ndims
    self.nlayers = nlayers

    init = tf.keras.initializers.GlorotUniform(seed)
    self.nodes = 256
    self.denses1, self.denses2 = [], []
    self.bn1 = []
    self.act1, self.act2 = [], []
    for i in range(self.nlayers):
      self.denses1.append(tf.keras.layers.Dense(self.nodes, kernel_initializer=init))

      # Option #0
      #self.bn1.append(tf.keras.layers.BatchNormalization())
      #self.act1.append(tf.keras.activations.relu)
      # Option 1
      #self.act1.append(tf.keras.layers.LeakyReLU(0.2))
      # Option #2
      self.act1.append(tf.keras.activations.tanh)
      self.bn1.append(tf.keras.layers.BatchNormalization())
      # Option #3
      #self.act1.append(tf.keras.activations.selu)


    last_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
    self.out = tf.keras.layers.Dense(self.ndims, kernel_initializer=last_init, activation='tanh')
    self.upper_bound = upper_bound
    self.lower_bound = lower_bound

  def call(self, inputs):
    states, rdm_variables = inputs
    #print(f'states 1: {states.shape}')
    #states = tf.keras.layers.BatchNormalization()(states)
    #print(f'states 2: {states.shape}')
    #print(f'rdm_variables 2: {rdm_variables.shape}')
    x1 = tf.keras.layers.concatenate([rdm_variables, states])
    # x1 = self.denses1[0](x1)
    for i in range(0, self.nlayers):
      x1 = self.denses1[i](x1)
      x1 = self.act1[i](x1)
      x1 = self.bn1[i](x1)

    x = self.out(x1)

    # Rescale using tanh [-1,1] to ensure it's within the parameter space
    # if self.lower_bound.all() != None:
    x = tf.keras.layers.Lambda(
      lambda xi: ((xi + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(x)

    return x