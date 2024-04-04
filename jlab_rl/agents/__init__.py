from jlab_rl.agents.registration import register, make, list_registered_modules

# Import agents
from jlab_rl.agents.keras_td3 import KerasTD3
from jlab_rl.agents.keras_gan_td3 import KerasGenerativeTD3
from jlab_rl.agents.keras_gan_kernel_dist_td3 import KerasKernelDistGenerativeTD3
from jlab_rl.agents.keras_gan_kernel_sampling_td3 import KerasKernelSamplingGenerativeTD3

# Single objective agents
register(
    id='KerasTD3-v0',
    entry_point='jlab_rl.agents:KerasTD3'
)

register(
    id='KerasGenerativeTD3-v0',
    entry_point='jlab_rl.agents:KerasGenerativeTD3'
)

register(
    id='KerasKernelDistGenerativeTD3-v0',
    entry_point='jlab_rl.agents:KerasKernelDistGenerativeTD3'
)