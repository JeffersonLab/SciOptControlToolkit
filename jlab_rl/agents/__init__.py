from jlab_rl.agents.registration import register, make, list_registered_modules

# Import agents
from jlab_rl.agents.keras_td3 import KerasTD3

# Register TD3 agent
register(
    id='KerasTD3-v0',
    entry_point='jlab_rl.agents:KerasTD3'
)
