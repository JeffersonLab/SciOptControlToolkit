from jlab_opt_control.envs.registration import register, make, list_registered_modules
from jlab_opt_control.envs.circle_env import Circle2D

register(
    id='DnC2s-Circle2D-Statefull-v0',
    entry_point='jlab_opt_control.envs:Circle2D',
    kwargs={'rdm_reset_mode': 'fixed',
            'statefull': True, 'max_episode_steps': 1}
)

register(
    id='DnC2s-Circle2D-Stateless-v0',
    entry_point='jlab_opt_control.envs:Circle2D',
    kwargs={'rdm_reset_mode': 'fixed',
            'statefull': False, 'max_episode_steps': 1}
)

# Differentiable backends for DEPO-family agents -- uniform reset so a
# rollout_batch_size > 1 training batch isn't just N copies of one rollout.
register(
    id='DnC2s-Circle2D-Diff-TF-v0',
    entry_point='jlab_opt_control.envs:Circle2D',
    kwargs={'rdm_reset_mode': 'uniform', 'statefull': True,
            'max_episode_steps': 1, 'backend': 'tensorflow'}
)

register(
    id='DnC2s-Circle2D-Diff-Torch-v0',
    entry_point='jlab_opt_control.envs:Circle2D',
    kwargs={'rdm_reset_mode': 'uniform', 'statefull': True,
            'max_episode_steps': 1, 'backend': 'torch'}
)
