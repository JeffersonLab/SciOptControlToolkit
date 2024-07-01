from jlab_opt_control.envs.registration import register, make, list_registered_modules
from jlab_opt_control.envs.circle_env import Circle2D
from jlab_opt_control.envs.goni_env import PolarizedBeamEnv
 
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

register(
    id='Polarized-Beam-v0',
    entry_point='jlab_opt_control.envs:PolarizedBeamEnv',
)
