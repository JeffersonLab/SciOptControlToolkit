# Authors:  Malachi Schram
# Script: MLP dynamic model for model-based RL
# Org: TJNAF

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D
from tensorflow.keras.initializers import RandomUniform
# from official.nlp.modeling.layers.spectral_normalization import SpectralNormalization
from tensorflow.keras.optimizers import Adam

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
    def __init__(self, ndlayers, hidden_size, drop_percent, num_states, num_actions):
        super(DynamicModel, self).__init__()

        self.ntrials = 5
        self.ndlayers = ndlayers
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        #min_max_constraint = tf.keras.constraints.MinMaxNorm(min_value=0.01, max_value=1.0, rate=1.0, axis=0)
        self.drop_percent = tf.Variable(drop_percent, constraint= lambda z: tf.clip_by_value(z, 0.01, 0.5))

        # Layers
        self.dense_layers, self.mc_layers = [], []
        for l in range(self.ndlayers):
            self.dense_layers.append(tf.keras.layers.Dense(self.hidden_size, activation="tanh", name='dense{}'.format(l)))
            self.mc_layers.append(CustomDropout(self.drop_percent))
        # Predict the next state and the reward
        self.state_layer = tf.keras.layers.Dense(self.num_states, activation="linear", name='next_states_pred')
        self.reward_layer = tf.keras.layers.Dense(1, activation="linear", name='reward_pred')

    def call(self, inputs):
        state_inputs, action_inputs = inputs
        #print(state_inputs.shape)
        #print(action_inputs.shape)
        state_actions = tf.keras.layers.Concatenate(axis=1)([state_inputs, action_inputs])
        #print(state_actions.shape)
        # Loop over layers
        for l in range(self.ndlayers):
            state_actions = self.dense_layers[l](state_actions)
            state_actions = self.mc_layers[l](state_actions, self.drop_percent)
        next_states = self.state_layer(state_actions)
        rewards = self.reward_layer(state_actions)
        return next_states, rewards

    def get_loss(self, y_preds, y):
        y_pred = tf.math.reduce_mean(y_preds, axis=0)
        y_pred_std = tf.math.reduce_std(y_preds, axis=0) + 1e-5
        diff_sqrt_term = tf.math.squared_difference(y_pred, y)
        diff_sqrt_term = 0.5 * diff_sqrt_term / (tf.math.square(y_pred_std))
        std_term = 0.5 * tf.math.log(tf.math.square(y_pred_std))
        return tf.math.reduce_mean(diff_sqrt_term + std_term)

    def predict_uq(self, inputs):
        predictions = [self.call(inputs) for _ in range(self.ntrials)]
        next_state_preds, reward_preds = list(zip(*predictions))
        next_state_pred = tf.math.reduce_mean(next_state_preds, axis=0)
        next_state_pred_std = tf.math.reduce_std(next_state_preds, axis=0) + 1e-5
        reward_pred = tf.math.reduce_mean(reward_preds, axis=0)
        reward_pred_std = tf.math.reduce_std(reward_preds, axis=0) + 1e-5
        return next_state_pred, next_state_pred_std, reward_pred, reward_pred_std

    def train_step(self, data):
        x, y = data
        states, actions = x
        #print(states)
        next_states, rewards = y

        with tf.GradientTape() as tape:
            predictions = [self([states, actions]) for _ in range(self.ntrials)]
            next_state_preds, reward_preds = list(zip(*predictions))
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

            # Calculate error for states
            next_state_loss = self.get_loss(next_state_preds, next_states)

            # Calculate error for rewards
            reward_loss = self.get_loss(reward_preds, rewards)

            # Combined error
            loss = next_state_loss + reward_loss

        grad = tape.gradient(loss, self.trainable_variables)
        del tape

        # Apply gradients to variables
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        #self.compiled_metrics.update_state(y, y_pred)
        #self.add_metric(loss, aggregation='loss', name='loss')
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}