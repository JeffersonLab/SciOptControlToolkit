from jlab_rl.agents.registration import register, make, list_registered_modules

# Import agents
from jlab_rl.agents.keras_td3 import KerasTD3
from jlab_rl.agents.keras_gan_td3 import KerasGenerativeTD3
from jlab_rl.agents.keras_ensemble_critic_td3 import KerasECGTD3
from jlab_rl.agents.keras_contraint_gan_td3 import KerasTD3 as KerasConstraintGenerativeTD3
# from jlab_rl.agents.keras_td3_critic_dgpa import KerasTD3CriticDGPA
from jlab_rl.agents.keras_td3_action_dgpa_v1 import KerasTD3ActorDGPA as KerasTD3ActorDGPA_v1
from jlab_rl.agents.keras_td3_action_dgpa_v2 import KerasTD3ActorDGPA as KerasTD3ActorDGPA_v2
from jlab_rl.agents.keras_td3_critic_dgpa_v2 import KerasTD3CriticDGPA as KerasTD3CriticDGPA_v2
from jlab_rl.agents.keras_mo_td3 import KerasTD3 as KerasMultiObjTD3
from jlab_rl.agents.keras_gdmb_td3 import KerasGenerativeDynamicModelBased
from jlab_rl.agents.keras_ensemble_grl import KerasEnsembleGenerativeTD3
from jlab_rl.agents.keras_gtd3 import KerasGTD3

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
    id='KerasGTD3-v0',
    entry_point='jlab_rl.agents:KerasGTD3'
)

register(
    id='KerasECGTD3-v0',
    entry_point='jlab_rl.agents:KerasECGTD3'
)

register(
    id='KerasEnsembleGenerativeTD3-v0',
    entry_point='jlab_rl.agents:KerasEnsembleGenerativeTD3'
)

register(
    id='KerasGenerativeDynamicModelBased-v0',
    entry_point='jlab_rl.agents:KerasGenerativeDynamicModelBased'
)
# Multi-objective agents
register(
    id='KerasMultiObjTD3-v0',
    entry_point='jlab_rl.agents:KerasMultiObjTD3'
)