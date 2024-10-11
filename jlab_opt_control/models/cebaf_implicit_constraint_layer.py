import tensorflow as tf

class CEBAFImplicitConstraintLayer(tf.keras.layers.Layer):
    def __init__(self, env, max_iter=10000, trip_high=6, heat_high=2500, **kwargs):
        super(CEBAFImplicitConstraintLayer, self).__init__(**kwargs)
        self.env = env
        nrewards = env.reward_space.shape[0]
        nactions = env.action_space.shape[0]
        print(f'nactions: {nactions}')

        self.input_layer = tf.keras.layers.Dense(nactions, use_bias=True)
        self.projection = tf.keras.layers.Dense(nactions, use_bias=True)
        self.tolerance = 0.05
        self.max_iter = max_iter
        self.iterations = 0
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
        #self.err = 0
        self.trip_high = trip_high
        self.heat_high = heat_high

    def call(self, actions, alphas, train=False):

        x = tf.concat([actions, alphas], axis=1)
        x = self.input_layer(x)
        safe_x = self.projection(x) + x
        safe_x = tf.clip_by_value(safe_x, -1, 1)
        if train:
            self.iterations = 0
            self.err = 99999
            while self.iterations < self.max_iter:
                with tf.GradientTape(persistent=False) as tape:
                    self.iterations += 1
                    # Calculate safe x
                    safe_x = self.projection(x) + x
                    safe_x = tf.clip_by_value(safe_x, -1, 1)
                    pred_actions = self.env.denormalize_action(safe_x)
                    # # Get energy
                    # min_gsets = tf.convert_to_tensor(self.env.linac.min_gsets, dtype=tf.float32)
                    # max_gsets = tf.convert_to_tensor(self.env.linac.max_gset_to_use, dtype=tf.float32)
                    # diff = max_gsets - min_gsets
                    # pred_env_actions = ((pred_actions + 1.0) / 2.0) * diff + min_gsets
                    # pred_energies = tf.reduce_sum(pred_env_actions * self.env.linac.lengths, axis=1)
                    # pred_energies = tf.expand_dims(pred_energies, axis=1)
                    # self.err_min = tf.keras.activations.relu(self.env.min_energy - pred_energies)
                    # self.err_max = tf.keras.activations.relu(pred_energies - self.env.max_energy)
                    # self.err = tf.reduce_mean(self.err_min + self.err_max)

                    pred_energies = self.env.get_energy(pred_actions)[:, 0]
                    pred_trip = self.env.linac.getTripRates(gradients=pred_actions)
                    pred_heat = self.env.linac.getRFHeat(gradients=pred_actions)
                    self.err_min = tf.keras.activations.relu(self.env.min_energy - pred_energies)
                    self.err_max = tf.keras.activations.relu(pred_energies - self.env.max_energy)
                    self.err_trip = tf.keras.activations.relu(pred_trip - self.trip_high)/self.trip_high
                    self.err_heat = tf.keras.activations.relu(pred_heat - self.heat_high)/self.heat_high
                    self.err = tf.reduce_mean(self.err_min + self.err_max + self.err_trip + self.err_heat)
                    #print(f'energy: {pred_energies}; self.err: {self.err}')

                if self.err < self.tolerance:
                    break
                # Update gradient
                gradients = tape.gradient(self.err, self.projection.trainable_variables)
                self.opt.apply_gradients(zip(gradients, self.projection.trainable_variables))
            # print(f'IC Error {self.err}')
#       td3_log.info(f'Iter: {self.projection_layer.iterations} at tolerance {self.projection_layer.err}')
        return safe_x