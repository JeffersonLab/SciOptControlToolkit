from jlab_rl.agents.registration import register, make, list_registered_modules
from jlab_rl.envs.proxyapp_v0 import proxy_app
from jlab_rl.envs.circle_constraint_v0 import circle_constraint_env as circle_constraint_stateless_env
from jlab_rl.envs.circle_constraint_v1 import circle_constraint_env as circle_constraint_statefull_env
from jlab_rl.envs.cebaf_env_v0 import cebaf_env

# Register the proxy app
register(
    id='ProxyApp-v0',
    entry_point='jlab_rl.envs:proxy_app',
    kwargs={'loss_type': 'default'},
)

register(
    id='ProxyApp-v1',
    entry_point='jlab_rl.envs:proxy_app',
    kwargs={'loss_type':'emil'},
)

# Register the circle constraint app
# This environment is stateless: meaning the action overides the state
register(
    id='Circle2DEnv-v0',
    entry_point='jlab_rl.envs:circle_constraint_stateless_env'
)

# This environment is stateless: meaning the action overides the state
register(
    id='Circle2DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
)

# This environment is stateless: meaning the action overides the state
register(
    id='UniformCircle2DEnv-v1',
    entry_point='jlab_rl.envs:circle_constraint_statefull_env',
    kwargs={'ndim': 2, 'rdm_reset_mode': 'uniform'},
)

# This environment is stateless: meaning the action overides the state
register(
    id='CEBAF2DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test2'},
)

register(
    id='CEBAF4DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test4'},
)

register(
    id='CEBAF8DEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': '1l10_test8'},
)

register(
    id='CEBAFSouthEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': 'south'},
)

register(
    id='CEBAFNorthEnv-v0',
    entry_point='jlab_rl.envs:cebaf_env',
    kwargs={'linac': 'north'},
)
