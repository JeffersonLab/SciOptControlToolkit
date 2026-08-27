from jlab_opt_control.core.registry_core import Registry

replay_registry = Registry("Replay Buffer")


def register(id, **kwargs):
    return replay_registry.register(id, **kwargs)


def make(id, **kwargs):
    return replay_registry.make(id, **kwargs)


def spec(id):
    return replay_registry.spec(id)


def list_registered_modules():
    return replay_registry.list_registered_modules()
