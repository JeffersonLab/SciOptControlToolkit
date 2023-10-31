# Copyright (c) 2020, Jefferson Science Associates, LLC. All Rights Reserved. Redistribution
# and use in source and binary forms, with or without modification, are permitted as a
# licensed user provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# This material resulted from work developed under a United States Government Contract.
# The Government retains a paid-up, nonexclusive, irrevocable worldwide license in such
# copyrighted data to reproduce, distribute copies to the public, prepare derivative works,
# perform publicly and display publicly and to permit others to do so.
#
# THIS SOFTWARE IS PROVIDED BY JEFFERSON SCIENCE ASSOCIATES LLC "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# JEFFERSON SCIENCE ASSOCIATES, LLC OR THE U.S. GOVERNMENT BE LIABLE TO LICENSEE OR ANY
# THIRD PARTES FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import importlib
import logging

env_reg_log = logging.getLogger("Environment Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class EnvSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the agent with appropriate kwargs"""
        if self.entry_point is None:
            raise env_reg_log.error('Attempting to make deprecated Env {}. \
                               (HINT: is there a newer registered version \
                               of this Env?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class EnvRegistry(object):
    def __init__(self):
        self.disc_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            env_reg_log.info('Making new agent: %s (%s)', path, kwargs)
        else:
            env_reg_log.info('Making new agent: %s', path)
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
                raise env_reg_log.error('A module ({}) was specified for the agent but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `agent_module.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.disc_specs[id]
        except KeyError:
            raise env_reg_log.error('No registered agent with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.disc_specs:
            raise env_reg_log.error('Cannot re-register id: {}'.format(id))
        self.disc_specs[id] = EnvSpec(id, **kwargs)


# Global agent registry
env_registry = EnvRegistry()


def register(id, **kwargs):
    return env_registry.register(id, **kwargs)


def make(id, **kwargs):
    return env_registry.make(id, **kwargs)


def spec(id):
    return env_registry.spec(id)


def list_registered_modules():
    return list(env_registry.disc_specs.keys())
