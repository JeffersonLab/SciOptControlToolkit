from jlab_rl.agents.registration import register, make, list_registered_modules

# Import agents
from jlab_rl.agents.keras_td3 import KerasTD3
# from jlab_rl.agents.keras_td3_critic_dgpa import KerasTD3CriticDGPA
from jlab_rl.agents.keras_td3_action_dgpa import KerasTD3ActorDGPA

# Register TD3 agent
register(
    id='KerasTD3-v0',
    entry_point='jlab_rl.agents:KerasTD3'
)

# # Register TD3 w/ DGPA Critic
# register(
#     id='KerasTD3CriticDGPA-v0',
#     entry_point='jlab_rl.agents:KerasTD3CriticDGPA'
# )

# Register TD3 w/ DGPA Actor
register(
    id='KerasTD3ActorDGPA-v0',
    entry_point='jlab_rl.agents:KerasTD3ActorDGPA'
)