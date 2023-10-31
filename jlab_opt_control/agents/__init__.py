from jlab_opt_control.agents.registration import register, make, list_registered_modules
from jlab_opt_control.agents.keras_td3 import KerasTD3

# Single objective agents
register(
    id='KerasTD3-v0',
    entry_point='jlab_opt_control.agents:KerasTD3',
    kwargs={'cfg': 'keras_td3.cfg'},
)