import os
import csv
import json
import random
import logging
import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import Adam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)


class DQN:
    def __init__(self, env, cfg='../cfg/dqn_setup.json', arch_type='MLP', nmodels=0):
        self.arch_type = arch_type
        self.env = env
        self.memory = deque(maxlen=2000)
        self.avg_reward = 0
        self.target_train_counter = 0

        self.total_actions_taken = 1
        self.individual_action_taken = np.ones(self.env.action_space.n)
        logger.info('Agent action space:{}'.format(self.env.action_space.n))
        logger.info('Agent state space:{}'.format(self.env.observation_space.shape))

        # Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)

        self.gamma = float(data['gamma']) if float(data['gamma']) else 0.95  # discount rate
        self.epsilon = float(data['epsilon']) if float(data['epsilon']) else 1.0  # exploration rate
        self.epsilon_min = float(data['epsilon_min']) if float(data['epsilon_min']) else 0.05
        self.epsilon_decay = float(data['epsilon_decay']) if float(data['epsilon_decay']) else 0.995
        self.learning_rate = float(data['learning_rate']) if float(data['learning_rate']) else 0.001
        self.batch_size = int(data['batch_size']) if int(data['batch_size']) else 32
        self.tau = float(data['tau']) if float(data['tau']) else 1.0
        self.warmup_step = float(data['warmup_step']) if float(data['warmup_step']) else 100
        self.save_model = ''

        if self.arch_type == 'LSTM':
            logger.info('Defined Arch Type:{}'.format(self.arch_type))
            self.model = self._build_lstm_model()
            self.target_model = self._build_lstm_model()
        elif self.arch_type == 'MLP_Ensemble':
            logger.info('Defined Arch Type:{}'.format(self.arch_type))
            self.model = self._build_ensemble(nmodels)
            self.target_model = self._build_ensemble(nmodels)
        else:
            logger.info('Using Default Arch Type:{}'.format(self.arch_type))
            self.model = self._build_model()
            self.target_model = self._build_model()

        # Save information
        train_file_name = 'dqn_{}_lr{}.log'.format(self.arch_type, self.learning_rate)
        self.train_file = open(train_file_name, 'w')
        self.train_writer = csv.writer(self.train_file, delimiter=" ")

    def _build_model(self):
        # Input: state
        state_input = Input(self.env.observation_space.shape)
        h1 = Dense(128, activation='relu')(state_input)
        h2 = Dense(128, activation='relu')(h1)
        h3 = Dense(128, activation='relu')(h2)
        # Output: value mapped to action
        output = Dense(self.env.action_space.n, activation='linear')(h3)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=self.learning_rate, clipnorm=1.0, clipvalue=0.5)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=adam)
        model.summary()
        return model

    def _build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(1, self.env.observation_space.shape[0])))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(self.env.action_space.n, ))
        adam = Adam(lr=self.learning_rate, clipnorm=1.0, clipvalue=0.5)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=adam)
        model.summary()
        return model

    def _build_ensemble(self, nmodel=5):
        # Define input
        state_input = Input(self.env.observation_space.shape)
        outputs = []
        for _ in range(nmodel):
            # Input: state
            h1 = Dense(128, activation='relu')(state_input)
            h2 = Dense(128, activation='relu')(h1)
            h3 = Dense(128, activation='relu')(h2)
            # Output: value mapped to action
            output = Dense(self.env.action_space.n, activation='linear')(h3)
            outputs.append(output)

        out_avg = tf.keras.layers.Average()(outputs)
        model = tf.keras.models.Model(inputs=[state_input], outputs=out_avg)
        adam = Adam(lr=self.learning_rate, clipnorm=1.0, clipvalue=0.5)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=adam)
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        action = 0
        policy_type = 0
        if (np.random.rand() <= self.epsilon) or (len(self.memory) <= self.warmup_step):
            logger.info('Random action')
            action = random.randrange(self.env.action_space.n)
            # Update randomness
            if len(self.memory) > self.batch_size:
                self.epsilon_adj()
        else:
            logger.info('NN action')
            np_state = np.array(state).reshape(1, len(state))
            if self.arch_type == 'LSTM':
                np_state = np.array(state).reshape(1, 1, len(state))
            logger.info('NN action shape{}'.format(np_state.shape))
            act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])
            policy_type = 1

        return action, policy_type

    def play(self, state):
        np_state = np.array(state).reshape(1, len(state))
        if self.arch_type == 'LSTM':
            np_state = np.array(state).reshape(1, 1, len(state))
        act_values = self.target_model.predict(np_state)
        return np.argmax(act_values[0])

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        logger.info('### TRAINING MODEL ###')
        losses = []
        minibatch = random.sample(self.memory, self.batch_size)
        batch_states = []
        batch_target = []
        for state, action, reward, next_state, done in minibatch:
            np_state = np.array(state).reshape(1, len(state))
            np_next_state = np.array(next_state).reshape(1, len(next_state))
            if self.arch_type == 'LSTM':
                np_state = np.array(state).reshape(1, 1, len(state))
                np_next_state = np.array(next_state).reshape(1, 1, len(next_state))
            expected_q = 0
            if not done:
                expected_q = self.gamma * np.amax(self.target_model.predict(np_next_state)[0])
            target = reward + expected_q
            target_f = self.target_model.predict(np_state)
            target_f[0][action] = target

            if batch_states == []:
                batch_states = np_state
                batch_target = target_f
            else:
                batch_states = np.append(batch_states, np_state, axis=0)
                batch_target = np.append(batch_target, target_f, axis=0)

        history = self.model.fit(batch_states, batch_target, epochs=1, verbose=0)
        losses.append(history.history['loss'][0])
        self.train_writer.writerow([np.mean(losses)])
        self.train_file.flush()

        logger.info('### TRAINING TARGET MODEL ###')
        self.target_train()

        return np.mean(losses)

    def target_train(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.target_model.load_weights(name)

    def save(self, name):
        abspath = os.path.abspath(self.save_model + name)
        path = os.path.dirname(abspath)
        if not os.path.exists(path):
            os.makedirs(path)
        # Save JSON config to disk
        model_json_name = self.save_model + name + '.json'
        json_config = self.target_model.to_json()
        with open(model_json_name, 'w') as json_file:
            json_file.write(json_config)
        # Save weights to disk
        self.target_model.save_weights(self.save_model + name + '.weights.h5')
        self.target_model.save(self.save_model + name + '.modelall.h5')
        logger.info('### SAVING MODEL ' + abspath + '###')
