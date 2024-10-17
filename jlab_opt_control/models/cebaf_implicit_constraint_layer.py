import tensorflow as tf

class CEBAFImplicitConstraintLayer(tf.keras.layers.Layer):
    def __init__(self, env, max_iter=10000, trip_high=6, heat_high=2500, **kwargs):
        super(CEBAFImplicitConstraintLayer, self).__init__(**kwargs)
        self.env = env
        self.trip_high = self.env.linac.max_allowed_trip
        self.heat_high = self.env.linac.max_allowed_heat

        nactions = env.action_space.shape[0]
        self.projection1 = tf.keras.layers.Dense(128, use_bias=True)
        self.projection2 = tf.keras.layers.Dense(256, use_bias=True)
        self.projection3 = tf.keras.layers.Dense(nactions, use_bias=True)
        self.projection = tf.keras.layers.Dense(nactions, use_bias=True)
        self.tolerance = 0.05
        self.max_iter = max_iter
        self.iterations = 0
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.005)


    def call(self, x, train=False):
        safe_x = self.projection(x) + x
        safe_x = tf.clip_by_value(safe_x, -1, 1)
        if train:
            self.iterations = 0
            self.err = 99999
            while self.iterations < self.max_iter:
                with tf.GradientTape(persistent=False) as tape:
                    self.iterations += 1
                    # Calculate safe x
                    x = self.projection1(x)
                    x = self.projection2(x)
                    x = self.projection3(x)
                    safe_x = self.projection(x) + x
                    safe_x = tf.clip_by_value(safe_x, -1, 1)
                    pred_actions = self.env.denormalize_action(safe_x)
                    pred_energies = self.env.get_energy(pred_actions)[:, 0]
                    self.err_min = 75*tf.keras.activations.relu(self.env.min_energy - pred_energies)/self.env.min_energy
                    self.err_max = 75*tf.keras.activations.relu(pred_energies - self.env.max_energy)/self.env.max_energy
                    # pred_trip = self.env.linac.getTripRates(gradients=pred_actions)
                    # pred_heat = self.env.linac.getRFHeat(gradients=pred_actions)
                    # self.err_trip = (pred_trip - self.trip_high)/self.trip_high
                    # self.err_heat = (pred_heat - self.heat_high)/self.heat_high
                    self.err = tf.reduce_mean(self.err_min + self.err_max)# + self.err_trip + self.err_heat)
                    #distance = tf.keras.losses.CosineSimilarity()(safe_x,safe_x)
                    #self.err_trip = tf.keras.activations.relu(pred_trip - self.trip_high)/self.trip_high
                    #self.err_heat = tf.keras.activations.relu(pred_heat - self.heat_high)/self.heat_high
                    #self.err -= distance

                if self.err < self.tolerance:
                    break
                # Update gradient
                gradients = tape.gradient(self.err, self.projection.trainable_variables)
                gradients = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients]
                self.opt.apply_gradients(zip(gradients, self.projection.trainable_variables))
        return safe_x