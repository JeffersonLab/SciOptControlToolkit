import importlib
import logging

replay_log = logging.getLogger("Replay Buffer Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class ReplaySpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the Replay Buffer with appropriate kwargs"""
        if self.entry_point is None:
            raise replay_log.error('Attempting to make deprecated Replay Buffer {}. \
                               (HINT: is there a newer registered version \
                               of this Replay Buffer?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class ReplayRegistry(object):
    def __init__(self):
        self.replay_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            replay_log.info('Making new replay buffer: %s (%s)', path, kwargs)
        else:
            replay_log.info('Making new replay buffer: %s', path)
        replay_spec = self.spec(path)
        replay = replay_spec.make(**kwargs)

        return replay

    def all(self):
        return self.replay_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise replay_log.error('A module ({}) was specified for the replay buffer but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `replay_buffer_module.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.replay_specs[id]
        except KeyError:
            raise replay_log.error(
                'No registered replay buffer with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.replay_specs:
            raise replay_log.error('Cannot re-register id: {}'.format(id))
        self.replay_specs[id] = ReplaySpec(id, **kwargs)


# Global Replay Buffer registry
replay_registry = ReplayRegistry()


def register(id, **kwargs):
    return replay_registry.register(id, **kwargs)


def make(id, **kwargs):
    return replay_registry.make(id, **kwargs)


def spec(id):
    return replay_registry.spec(id)


def list_registered_modules():
    return list(replay_registry.replay_specs.keys())
