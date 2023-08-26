from jlab_rl.agents.registration import register, make, list_registered_modules
from jlab_rl.envs.proxyapp_v0 import proxy_app
from jlab_rl.envs.circle_constraint_v0 import circle_constraint_env as circle_constraint_stateless_env
from jlab_rl.envs.circle_constraint_v1 import circle_constraint_env as circle_constraint_statefull_env
from jlab_rl.envs.cebaf_env_v0 import cebaf_env
from jlab_rl.envs.gaussian_gan_env_v0 import ProxyGaussian

# Register the proxy app
register(
    id='DnC2s-ProxyApp-v0',
    entry_point='jlab_rl.envs:proxy_app',
    kwargs={'loss_type': 'default'},
)

register(
    id='DnC2s-ProxyApp-v1',
    entry_point='jlab_rl.envs:proxy_app',
    kwargs={'loss_type':'emil'},
)

# Register the circle constraint app
# This environment is stateless: meaning the action overides the state
register(
    id='DnC2s-Circle2DEnv-v0',
    entry_point='jlab_rl.envs:circle_constraint_stateless_env',
)

register(
    id='DnC2s-Circle6DEnv-v0',
    entry_point='jlab_rl.envs:circle_constraint_stateless_env',
    kwargs={'ndim': 6},
)

register(
    id='DnC2s-StatelessCircle2DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 2, 'rdm_reset_mode': 'circle', 'statefull': False},
)

register(
    id='DnC2s-StatelessCircle4DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 4, 'rdm_reset_mode': 'circle', 'statefull' : False},
)

register(
    id='DnC2s-StatelessCircle6DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 6, 'rdm_reset_mode': 'circle', 'statefull' : False},
)

register(
    id='DnC2s-StatelessCircle12DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 12, 'rdm_reset_mode': 'circle', 'statefull' : False},
)

# This environment is stateless: meaning the action overides the state
register(
    id='DnC2s-StatefullCircle2DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 2, 'rdm_reset_mode': 'circle', 'statefull': True},
)

register(
    id='DnC2s-FixedStatefullCircle2DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 2, 'rdm_reset_mode': 'fixed', 'statefull': True},
)

# This environment is stateless: meaning the action overides the state
register(
    id='DnC2s-StatefullCircle4DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 4, 'rdm_reset_mode': 'circle', 'statefull': True},
)

# This environment is stateless: meaning the action overides the state
register(
    id='DnC2s-StatefullCircle6DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 6, 'rdm_reset_mode': 'circle', 'statefull': True},
)

# This environment is stateless: meaning the action overides the state
register(
    id='DnC2s-StatefullCircle12DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 12, 'rdm_reset_mode': 'circle', 'statefull': True},
)

# This environment is stateless: meaning the action overides the state
register(
    id='DnC2s-UniformCircle2DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 2, 'rdm_reset_mode': 'uniform'},
)

# This environment is stateless: meaning the action overides the state
register(
    id='DnC2s-CEBAF2DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test2'}
)

register(
    id='DnC2s-CEBAF4DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test4'}
)

# For 8 cavities in 1L10, the heat was below 21.4
register(
    id='DnC2s-CEBAF8DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test8'}
)

register(
    id='DnC2s-HeatObj-CEBAF8DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test8', 'objective': 'heat'}
)

register(
    id='DnC2s-TripObj-CEBAF8DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test8', 'objective': 'trip'},
)

register(
    id='DnC2s-MixedObj-CEBAF8DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test8', 'objective': 'mixed'},
)

register(
    id='DnC2s-MultObj-CEBAF8DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test8', 'objective': 'multi-obj'},
)

register(
    id='DnC2s-CEBAFSouthEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': 'south'},
)

register(
    id='DnC2s-CEBAFNorthEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': 'north'},
)


register(
    id='DnC2s-GaussianEnv-v0',
    entry_point='jlab_rl.envs:ProxyGaussian',
)