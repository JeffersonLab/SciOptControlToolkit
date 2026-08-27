from jlab_opt_control.core.registry_core import Registry

agent_registry = Registry("Agent")


def register(id, **kwargs):
    return agent_registry.register(id, **kwargs)


def make(id, **kwargs):
    return agent_registry.make(id, **kwargs)


def spec(id):
    return agent_registry.spec(id)


def list_registered_modules():
    return agent_registry.list_registered_modules()
