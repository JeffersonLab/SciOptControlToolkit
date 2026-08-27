from jlab_opt_control.core.registry_core import Registry

model_registry = Registry("Model")


def register(id, **kwargs):
    return model_registry.register(id, **kwargs)


def make(id, **kwargs):
    return model_registry.make(id, **kwargs)


def spec(id):
    return model_registry.spec(id)


def list_registered_modules():
    return model_registry.list_registered_modules()
