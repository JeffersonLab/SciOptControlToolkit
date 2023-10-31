import importlib
import logging

disc_log = logging.getLogger("Agent Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class AgentSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the agent with appropriate kwargs"""
        if self.entry_point is None:
            raise disc_log.error('Attempting to make deprecated Agent {}. \
                               (HINT: is there a newer registered version \
                               of this Agent?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class AgentRegistry(object):
    def __init__(self):
        self.disc_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            disc_log.info('Making new agent: %s (%s)', path, kwargs)
        else:
            disc_log.info('Making new agent: %s', path)
        disc_spec = self.spec(path)
        disc = disc_spec.make(**kwargs)

        return disc

    def all(self):
        return self.disc_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise disc_log.error('A module ({}) was specified for the agent but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `agent_module.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.disc_specs[id]
        except KeyError:
            raise disc_log.error('No registered agent with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.disc_specs:
            raise disc_log.error('Cannot re-register id: {}'.format(id))
        self.disc_specs[id] = AgentSpec(id, **kwargs)


# Global agent registry
disc_registry = AgentRegistry()


def register(id, **kwargs):
    return disc_registry.register(id, **kwargs)


def make(id, **kwargs):
    return disc_registry.make(id, **kwargs)


def spec(id):
    return disc_registry.spec(id)


def list_registered_modules():
    return list(disc_registry.disc_specs.keys())
