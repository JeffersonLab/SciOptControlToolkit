import numpy as np
import jlab_rl.envs as gym

def plot_states(env_id, states):
    # Plot
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_title(env_id, fontsize=14)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("X2", fontsize=12)
    # scatter with colormap mapping to z value
    ax.scatter(states[:,0], states[:,1], s=20, marker='o')
    plt.xlim(np.min(states[:,0]), np.max(states[:,0]))
    plt.ylim(np.min(states[:,1]), np.max(states[:,1]))
    plt.savefig('./utest_scatter_reset_{}.png'.format(env_id))

end_id = 'CEBAF2DEnv-v0'
env = gym.make(end_id)
print(env.reset())
reset_states = np.array([env.reset()[1] for _ in range(2)])
print('reset states:\n', reset_states)

reset_states = np.array([env.reset()[0] for _ in range(10000)])
plot_states(end_id+'-Normalized', reset_states)

reset_states = np.array([env.reset()[1] for _ in range(10000)])
print(reset_states.shape)
plot_states(end_id+'-Actual', reset_states)
sqrt_states = np.square(reset_states)
print(sqrt_states.shape)
radius = np.sqrt(np.sum(sqrt_states, axis=1))
print(radius.shape)
print('Energy min/max: {}/{}'.format(np.min(radius), np.max(radius)))
# empty_action = np.array([0,0])
# env.step(empty_action)
#
# # Check reset sampling
# end_ids = ['Circle2DEnv-v1', 'UniformCircle2DEnv-v1']
# for end_id in end_ids:
#     env = gym.make(end_id)
#     reset_states = np.array([env.reset()[0] for _ in range(10000)])
#     plot_states(end_id, reset_states)
