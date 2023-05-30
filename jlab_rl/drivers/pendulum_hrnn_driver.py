import gym
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
#
import tensorflow as tf

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

def integrate_model(model, t_span, y0, **kwargs):
    def fun(t, np_x):
        x = tf.tensor(np_x)
        dx = model.time_derivative(x)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

class BaselineModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu)
    self.dense3 = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)
    self.out = tf.keras.layers.Dense(3, activation='linear')

  def time_derivative(self, x, t=None, separate_fields=False):
    return self(x)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    return self.out(x)

bs_model = BaselineModel()


env = gym.make('Pendulum-v1')#, render_mode="human")
print(env)

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

prev_obs = env.reset()
prev_obs = prev_obs[0]
print("The initial observation is {}".format(prev_obs))
prev_mom = []

## Model ##
# q_inputs = tf.keras.layers.Input(shape=(2,), name='q_input')
# p_inputs = tf.keras.layers.Input(shape=(2,), name='p_input')
# hqp = tf.keras.layers.Concatenate(name='qp')([q_inputs,p_inputs])
# hqp = tf.keras.layers.Dense(32, activation='linear', name='hqp1')(hqp)
# hqp = tf.keras.layers.Dense(16, activation='linear', name='hqp2')(hqp)
# h = tf.keras.layers.Dense(1, activation='linear', name='l2')(hqp)
# hqpdot = tf.keras.layers.Dense(16, activation='linear', name='hqpdot1')(h)
# hqpdot = tf.keras.layers.Dense(32, activation='linear', name='hqpdot2')(hqpdot)
# qdot_output = tf.keras.layers.Dense(2, name='qdot_output')(hqpdot)
# pdot_output = tf.keras.layers.Dense(2, name='pdot_output')(hqpdot)
# hnn = tf.keras.Model(inputs=[q_inputs,p_inputs], outputs=[qdot_output, pdot_output])
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

q_list = []
p_list = []
qdot_list = []
pdot_list = []

prev_obs_list = []
next_obs_list = []
state_action_list = []


for i in tqdm(range(10000)):
    if i%200==0:
        prev_obs = env.reset()
        prev_obs = prev_obs[0]
    # Sample a random action from the entire action space
    #random_action = [0]
    random_action = env.action_space.sample()
    prev_obs_list.append(prev_obs)
    # # Take the action and get the new observation space
    new_obs, reward, done, info, _ = env.step(random_action)
    next_obs_list.append(new_obs)
    state_action_list.append(np.append(new_obs, random_action))
    prev_obs = new_obs
    #print("The new observation is {}".format(new_obs))
    x = new_obs[0]
    y = new_obs[1]
    px = (new_obs[0] - prev_obs[0])/0.05
    py = (new_obs[1] - prev_obs[1])/0.05
    new_mom = [px,py]

    if i>0:
        # We need one calculation before
        pdot_x = (new_mom[0] - prev_mom[0]) / 0.05
        pdot_y = (new_mom[1] - prev_mom[1]) / 0.05
        #
        q_list.append([x, y])
        p_list.append([px, py])
        qdot_list.append([px, py])
        pdot_list.append([pdot_x, pdot_y])
    #
    prev_obs = new_obs
    prev_mom = [px,py]
    #env.render()


    # plt.show()
# plt.figure(figsize=(9, 2))
# plt.subplot(121)
# plt.scatter(q_list[:,0], p_list[:,0])
# plt.subplot(122)
# plt.scatter(qdot_list[0], pdot_list[0])
# plt.savefig('dynamics.png')

# Train
bs_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanSquaredError()])
prev_obs_list = np.array(prev_obs_list)
next_obs_list = np.array(next_obs_list)
history = bs_model.fit(x=prev_obs_list, y=next_obs_list, epochs=100)

next_obs_list_hat = bs_model(prev_obs_list)
print(next_obs_list_hat.shape)
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.scatter(next_obs_list[:,0], next_obs_list_hat[:,0])
plt.subplot(132)
plt.scatter(next_obs_list[:,1], next_obs_list_hat[:,1])
plt.subplot(133)
plt.scatter(next_obs_list[:,2], next_obs_list_hat[:,2])
plt.savefig('nn_fpred_results.png')
sys.exit()
# ## Model ##
# qp_inputs = tf.keras.layers.Input(shape=(4,), name='q_input')
# #q_inputs = tf.keras.layers.BatchNormalization()(q_inputs)
# #p_inputs = tf.keras.layers.Input(shape=(2,), name='p_input')
# #p_inputs = tf.keras.layers.BatchNormalization()(p_inputs)
# #hqp = tf.keras.layers.Concatenate(name='qp')([q_inputs,p_inputs])
# hqp = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu, name='hqp1')(qp_inputs)
# hqp = tf.keras.layers.Dropout(0.15, name='dhqp1')(hqp)
# hqp = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu, name='hqp2')(hqp)
# hqp = tf.keras.layers.Dropout(0.15, name='dhqp2')(hqp)
# #h = tf.keras.layers.Dense(1, activation='tanh', name='h')(hqp)
# #hqpdot = tf.keras.layers.Dense(32, activation='tanh', name='hqpdot1')(h)
# #hqpdot = tf.keras.layers.Dense(64, activation='tanh', name='hqpdot2')(hqpdot)
# qpdot_output = tf.keras.layers.Dense(4, activation='linear',name='qdot_output')(hqp)
# #pdot_output = tf.keras.layers.Dense(2, activation='linear', name='pdot_output')(hqpdot)
# #hnn = tf.keras.Model(inputs=[q_inputs, p_inputs], outputs=[qdot_output, pdot_output])
# hnn = tf.keras.Model(inputs=[qp_inputs], outputs=[qpdot_output])
#
# #
# hnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#             loss=tf.keras.losses.MeanSquaredError(),
#             metrics=[tf.keras.metrics.MeanSquaredError()])

q_list = np.array(q_list)
p_list = np.array(p_list)
qdot_list = np.array(qdot_list)
pdot_list = np.array(pdot_list)
print('Sample size:', q_list.shape)


# Use concatenate() with axis
qp_con = np.concatenate((q_list, p_list), axis=1)
print(qp_con.shape)
qpdot_con = np.concatenate((qdot_list, pdot_list), axis=1)
print(qpdot_con.shape)

bs_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanSquaredError()])
history = bs_model.fit(x=qp_con, y=qpdot_con, epochs=100)

qpdot_list_hat = bs_model(qp_con)
print(qpdot_list_hat.shape)
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.scatter(q_list, p_list)
plt.subplot(132)
plt.scatter(qdot_list, pdot_list)
plt.subplot(133)
plt.scatter(qpdot_list_hat[:,2], qpdot_list_hat[:,3])
#plt.show()
plt.savefig('nn_results.png')

env.close()
