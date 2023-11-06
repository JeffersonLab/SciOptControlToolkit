class GenerativeActor(tf.keras.Model):

  def __init__(self, nactions=2, layer_nodes=[256, 256, 256, 256], lower_bound=None, upper_bound=None):
    super().__init__()
    seed = time.time_ns()
    tf.random.set_seed(seed)
    self.nactions = nactions
    self.nlayers = len(layer_nodes)
    init = tf.keras.initializers.GlorotUniform(seed)
    self.nodes = 256
    self.denses1 = []
    self.bn1 = []
    self.act1 = []
    for i in range(self.nlayers):
      self.denses1.append(tf.keras.layers.Dense(layer_nodes[i], kernel_initializer=init))
      self.bn1.append(tf.keras.layers.BatchNormalization())
      self.act1.append(tf.keras.activations.relu)

    self.out = tf.keras.layers.Dense(self.nactions, kernel_initializer=init, activation='tanh')
    self.upper_bound = upper_bound
    self.lower_bound = lower_bound

  def call(self, inputs):
    states, rdm_variables = inputs
    x1 = tf.keras.layers.concatenate([rdm_variables, states])
    for i in range(0, self.nlayers):
      x1 = self.denses1[i](x1)
      x1 = self.bn1[i](x1)
      x1 = self.act1[i](x1)

    x = self.out(x1)

    # Rescale using tanh [-1,1] to ensure it's within the parameter space
    # if self.lower_bound.all() != None:
    x = tf.keras.layers.Lambda(
      lambda xi: ((xi + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(x)

    return x