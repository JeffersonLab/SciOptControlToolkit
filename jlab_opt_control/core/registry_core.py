import importlib
import logging


def load(name):
    """Resolve a "module:attr" entry_point string into the attribute itself."""
    mod_name, attr_name = name.split(":")
    logging.getLogger("Registry").debug(
        f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)


class Spec(object):
    """Describes how to build one registered component."""

    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiate this spec with the given kwargs (call-site kwargs win)."""
        if self.entry_point is None:
            raise ValueError(
                'Attempting to make deprecated entry {}. '
                '(HINT: is there a newer registered version?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            return self.entry_point(**_kwargs)
        cls = load(self.entry_point)
        return cls(**_kwargs)


class Registry(object):
    """Generic id -> Spec registry shared by agents/models/buffers/envs.

    `kind` names what this registry holds (e.g. "Agent", "Model",
    "Replay Buffer", "Environment") and is used for both the logger name
    and error messages.
    """

    def __init__(self, kind):
        self.kind = kind
        self.log = logging.getLogger(f"{kind} Registry")
        self.specs = {}

    def make(self, path, **kwargs):
        if kwargs:
            self.log.info('Making new %s: %s (%s)', self.kind, path, kwargs)
        else:
            self.log.info('Making new %s: %s', self.kind, path)
        return self.spec(path).make(**kwargs)

    def all(self):
        return self.specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                msg = ('A module ({}) was specified for the {} but was not found, '
                       'make sure the package is installed with `pip install` '
                       'before calling `make()`').format(mod_name, self.kind.lower())
                self.log.error(msg)
                raise ImportError(msg) from e
        else:
            id = path

        try:
            return self.specs[id]
        except KeyError as e:
            msg = 'No registered {} with id: {}'.format(self.kind.lower(), id)
            self.log.error(msg)
            raise KeyError(msg) from e

    def register(self, id, **kwargs):
        if id in self.specs:
            msg = 'Cannot re-register id: {}'.format(id)
            self.log.error(msg)
            raise ValueError(msg)
        self.specs[id] = Spec(id, **kwargs)

    def list_registered_modules(self):
        return list(self.specs.keys())
