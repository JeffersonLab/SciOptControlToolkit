from jlab_rl.agents.registration import register, make, list_registered_modules

# Import agents
from jlab_rl.agents.keras_td3 import KerasTD3
#from jlab_rl.agents.keras_cebaf_td3 import KerasTD3 as KerasCEBAFTD3
from jlab_rl.agents.keras_gan_td3 import KerasGenerativeTD3
from jlab_rl.agents.keras_contraint_gan_td3 import KerasTD3 as KerasConstraintGenerativeTD3
# from jlab_rl.agents.keras_td3_critic_dgpa import KerasTD3CriticDGPA
from jlab_rl.agents.keras_td3_action_dgpa_v1 import KerasTD3ActorDGPA as KerasTD3ActorDGPA_v1
from jlab_rl.agents.keras_td3_action_dgpa_v2 import KerasTD3ActorDGPA as KerasTD3ActorDGPA_v2
from jlab_rl.agents.keras_td3_critic_dgpa_v2 import KerasTD3CriticDGPA as KerasTD3CriticDGPA_v2
from jlab_rl.agents.keras_modelbased_agent import KerasGenericModelBasedAgent
from jlab_rl.agents.keras_mo_gan_td3 import KerasMultiObjectiveGenerativeTD3

# Register TD3 agent
register(
    id='KerasTD3-v0',
    entry_point='jlab_rl.agents:KerasTD3'
)

register(
    id='KerasGenerativeTD3-v0',
    entry_point='jlab_rl.agents:KerasGenerativeTD3'
)

register(
    id='KerasMultiObjectiveGenerativeTD3-v0',
    entry_point='jlab_rl.agents:KerasMultiObjectiveGenerativeTD3'
)

register(
    id='KerasConstraintGenerativeTD3-v0',
    entry_point='jlab_rl.agents:KerasConstraintGenerativeTD3'
)

register(
    id='KerasCEBAFTD3-v0',
    entry_point='jlab_rl.agents:KerasCEBAFTD3'
)

register(
    id='KerasTD3-RFF-v0',
    entry_point='jlab_rl.agents:KerasTD3',
    kwargs={'nrff': 128}
)
# Register TD3 w/ DGPA Actor using Steven model
register(
    id='KerasTD3ActorDGPA-v2',
    entry_point='jlab_rl.agents:KerasTD3ActorDGPA_v2'
)

# Register TD3 w/ DGPA Actor using the original model
register(
    id='KerasTD3ActorDGPA-v1',
    entry_point='jlab_rl.agents:KerasTD3ActorDGPA_v1'
)

# Register TD3 w/ DGPA Critic using Steven model
register(
    id='KerasTD3CriticDGPA-v2',
    entry_point='jlab_rl.agents:KerasTD3CriticDGPA_v2'
)

register(
    id='KerasGenericModelBaseAdgent-v0',
    entry_point='jlab_rl.agents:KerasGenericModelBasedAgent'
)