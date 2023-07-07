from jlab_rl.agents.registration import register, make, list_registered_modules
from jlab_rl.envs.proxyapp_v0 import proxy_app
from jlab_rl.envs.circle_constraint_v0 import circle_constraint_env as circle_constraint_stateless_env
from jlab_rl.envs.circle_constraint_v1 import circle_constraint_env as circle_constraint_statefull_env
from jlab_rl.envs.cebaf_env_v0 import cebaf_env as cebaf2d

# Register the proxy app
register(
    id='ProxyApp-v0',
    entry_point='jlab_rl.envs:proxy_app'
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
    id='CEBAF2D-v0',
    entry_point='jlab_rl.envs:cebaf2d',
    kwargs={'linac': '1l06_test2'},
)