from jlab_rl.agents.registration import register, make, list_registered_modules
from jlab_rl.envs.proxyapp_v0 import proxy_app

# Register the proxy app
register(
    id='ProxyApp-v0',
    entry_point='jlab_rl.envs:proxy_app'
)
