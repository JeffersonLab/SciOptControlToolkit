import gym
from jlab_rl.models.keras_hnn import MLP

env = gym.make('Pendulum-v1')

# Create data
q_list = []
p_list = []
qdot_list = []
pdot_list = []

prev_obs_list = []
next_obs_list = []
state_action_list = []

for i in tqdm(range(10000)):
    if i%200 == 0:
        prev_obs = env.reset()
    # Sample a random action from the entire action space
    random_action = [0]
    # # Take the action and get the new observation space
    new_obs, reward, done, info, _ = env.step(random_action)
    q = env.th
    qdot = env.thdot

    th, thdot
    x = new_obs[0]
    y = new_obs[1]
    px = (new_obs[0] - prev_obs[0])/env.dt
    py = (new_obs[1] - prev_obs[1])/env.dt
    new_mom = [px,py]
    if i>0:
        # We need one calculation before
        pdot_x = (new_mom[0] - prev_mom[0]) / env.dt
        pdot_y = (new_mom[1] - prev_mom[1]) / env.dt
        #
        q_list.append([x, y])
        p_list.append([px, py])
        qdot_list.append([px, py])
        pdot_list.append([pdot_x, pdot_y])
    #
    prev_mom = [px, py]
    prev_obs = new_obs

# Build model
mlp_model = MLP(hidden_dim=200, output_dim)

# Train the model
mlp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanSquaredError()])
prev_obs_list = np.array(prev_obs_list)
next_obs_list = np.array(next_obs_list)
history = bs_model.fit(x=prev_obs_list, y=next_obs_list, epochs=100)