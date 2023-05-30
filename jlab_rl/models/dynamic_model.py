# Authors:  Malachi Schram
# Script: MLP dynamic model for model-based RL
# Org: TJNAF

# import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D
# from tensorflow.keras.initializers import RandomUniform
# # from official.nlp.modeling.layers.spectral_normalization import SpectralNormalization
# from tensorflow.keras.optimizers import Adam


class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, rate, switch=True):
        if switch:
            return tf.nn.dropout(inputs, rate=rate)
        else:
            return inputs


class DynamicModel(tf.keras.Model):
    def __init__(self, ndlayers, hidden_size, drop_value, num_states, num_actions):
        super(DynamicModel, self).__init__()

        self.ntrials = 5
        self.ndlayers = ndlayers
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.low_mc = 0.0001
        #self.drop_value = tf.Variable(drop_value)
        #self.drop_percent = tf.sigmoid(self.drop_value)+self.low_mc
        self.drop_percent = tf.Variable(drop_value, constraint=lambda z: tf.clip_by_value(z, 0.01, 0.5))
        #self.state_drop_percent = tf.Variable(drop_percent, constraint=lambda z: tf.clip_by_value(z, 0.001, 0.5))
        #self.reward_drop_percent = tf.Variable(drop_percent, constraint=lambda z: tf.clip_by_value(z, 0.001, 0.5))

        # Random later
        #self.rdm_layer1 = tf.keras.layers.Dense(self.hidden_size, activation="tanh", name='rdm', trainable=False)
        #self.rdm_layer2 = tf.keras.layers.Dense(self.hidden_size, activation="tanh", name='rdm', trainable=False)
        self.dense_layer0 = tf.keras.layers.Dense(self.hidden_size, activation="tanh", name='in_dense')
        # Layers
        self.dense_layers, self.mc_layers = [], []
        for layer in range(self.ndlayers):
            self.dense_layers.append(
                tf.keras.layers.Dense(self.hidden_size, activation="tanh", name='dense{}'.format(layer)))
            self.mc_layers.append(CustomDropout(self.drop_percent))
        #self.dense_layer0 = tf.keras.layers.Dense(self.hidden_size, activation="relu", name='dense1')
        #self.dense_layers.append(tf.keras.layers.Dense(self.hidden_size,
        #                                              activation="tanh", name='dense{}'.format(self.ndlayers)))
        # Predict the next state and the reward
        self.state_layer = tf.keras.layers.Dense(self.num_states, activation="linear", name='next_states_pred')
        self.reward_layer = tf.keras.layers.Dense(1, activation="linear", name='reward_pred')

    def call(self, inputs):
        state_inputs, action_inputs = inputs
        state_actions = tf.keras.layers.Concatenate(axis=1)([state_inputs, action_inputs])
        out = self.dense_layer0(state_actions)
        # Loop over layers
        for layer in range(self.ndlayers):
            x = self.dense_layers[layer](out)
            x = self.mc_layers[layer](x, self.drop_percent)
            out += x
        #state_actions = self.rdm_layer2(state_actions)
        # Loop over layers
        #self.drop_percent = tf.sigmoid(self.drop_value)+self.low_mc
        # for layer in range(self.ndlayers):
        #     state_actions = self.dense_layers[layer](state_actions)
        #     state_actions = self.mc_layers[layer](state_actions, self.drop_percent)
        #
        #state_actions = self.dense_layers[-1](state_actions)
        #state_actions1 = self.state_mc_layer(state_actions, self.state_mc_layer)
        next_states = self.state_layer(out)
        #state_actions2 = self.reward_mc_layer(state_actions, self.reward_mc_layer)
        rewards = self.reward_layer(out)
        return next_states, rewards

    # def get_loss(self, y_preds, y):
    #     y_pred = tf.math.reduce_mean(y_preds, axis=0)
    #     y_pred_std = tf.math.reduce_std(y_preds, axis=0) + 1e-5
    #     diff_sqrt_term = tf.math.squared_difference(y_pred, y)
    #     diff_sqrt_term = 0.5 * diff_sqrt_term / (tf.math.square(y_pred_std))
    #     std_term = 0.5 * tf.math.log(tf.math.square(y_pred_std))
    #     return tf.math.reduce_mean(diff_sqrt_term + std_term)

    @tf.function
    def get_loss(self, y_preds, y_pred_stds, y):
        diff_sqrt_term = tf.math.squared_difference(y_preds, y)
        diff_sqrt_term = 0.5 * diff_sqrt_term / (tf.math.square(y_pred_stds))
        std_term = 0.5 * tf.math.log(tf.math.square(y_pred_stds))
        return tf.math.reduce_mean(diff_sqrt_term + std_term)

    def predict_uq(self, inputs):

        # Repeat entries for ave & std
        states_rep = tf.repeat(inputs[0], self.ntrials, axis=0)
        actions_rep = tf.repeat(inputs[1], self.ntrials, axis=0)

        # Predict
        next_states_rep, rewards_rep = self([states_rep, actions_rep])
        #tf.print('rewards_rep', rewards_rep)

        # Restructure
        next_states_rep_rsh = tf.reshape(next_states_rep, shape=(inputs[0].shape[0],
                                                                 self.ntrials, next_states_rep.shape[-1]))
        rewards_rep_rsh = tf.reshape(rewards_rep, shape=(inputs[0].shape[0], self.ntrials, rewards_rep.shape[-1]))

        # Get ave and std
        next_states_ave = tf.reduce_mean(next_states_rep_rsh, axis=1)
        next_states_std = tf.math.reduce_std(next_states_rep_rsh, axis=1)
        rewards_ave = tf.reduce_mean(rewards_rep_rsh, axis=1)
        #tf.print('rewards_ave', rewards_ave)
        rewards_std = tf.math.reduce_std(rewards_rep_rsh, axis=1)

        return next_states_ave, next_states_std, rewards_ave, rewards_std

    def train_step(self, data):
        x, y = data
        states, actions = x
        next_states, rewards = y

        with tf.GradientTape() as tape:
            next_states_ave, next_states_std, rewards_ave, rewards_std = self.predict_uq([states, actions])
            next_state_loss = self.get_loss(next_states_ave, next_states_std, next_states)
            reward_loss = self.get_loss(rewards_ave, rewards_std, rewards)
            # predictions = [self([states, actions]) for _ in range(self.ntrials)]
            # next_state_preds, reward_preds = list(zip(*predictions))
            # print(predictions.shape)
            # next_state_preds, reward_preds = tf.stack(predictions, axis=1)
            # next_state_preds, reward_preds = [], []
            # for _ in range(15):
            #     ns,r = self([states, actions])
            #     next_state_preds.append(ns)
            #     reward_preds.append(r)
            # # preds = [self([states, actions]) for _ in range(25)]
            # # print(preds)
            # #next_state_preds, reward_preds = tf.convert_to_tensor([self([states, actions]) for _ in range(25)])
            # next_state_preds = tf.convert_to_tensor(next_state_preds)
            # reward_preds = tf.convert_to_tensor(reward_preds)

            # # Calculate error for states
            # next_state_loss = self.get_loss(next_state_preds, next_states)
            #
            # # Calculate error for rewards
            # reward_loss = self.get_loss(reward_preds, rewards)
            #reward_loss = tf.multiply(reward_loss,2.0)
            # tf.print('sloss:', next_state_loss)
            # tf.print('rloss:', reward_loss)
            # Combined error
            loss = tf.add(next_state_loss, reward_loss)

        tf.print(loss)
        grad = tape.gradient(loss, self.trainable_variables)
        del tape

        # Apply gradients to variables
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        #self.compiled_metrics.update_state(rewards, rewards_ave)
        #self.add_metric(loss, aggregation='loss', name='loss')
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}