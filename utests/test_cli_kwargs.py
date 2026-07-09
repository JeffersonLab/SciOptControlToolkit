import unittest
import tempfile
from unittest.mock import patch, MagicMock

import gymnasium as gym

import jlab_opt_control.agents as agents
from jlab_opt_control.agents.keras_td3 import KerasTD3
from jlab_opt_control.agents.keras_ddpg import KerasDDPG
from jlab_opt_control.agents.keras_sac import KerasSAC
from jlab_opt_control.drivers.run_continuous import run_opt


def make_env():
    return gym.make('Pendulum-v1')


class AgentKwargsFallbackMixin:
    """Every refactored agent must resolve buffer_type/buffer_size/load_model
    with CLI (kwargs) taking precedence over the cfg file, falling back to
    the cfg value (or its existing default) when no kwarg is supplied."""

    agent_cls = None
    cfg_buffer_type = 'ER-v0'

    def setUp(self):
        self.env = make_env()
        self.logdir = tempfile.mkdtemp()

    def test_signature_is_env_logdir_cfg_kwargs(self):
        import inspect
        sig = inspect.signature(self.agent_cls.__init__)
        params = list(sig.parameters.values())
        names = [p.name for p in params]
        self.assertEqual(names[0], 'self')
        self.assertEqual(names[1], 'env')
        self.assertEqual(names[2], 'logdir')
        self.assertEqual(names[3], 'cfg')
        self.assertEqual(params[-1].kind, inspect.Parameter.VAR_KEYWORD)

    def test_buffer_type_kwarg_overrides_cfg(self):
        agent = self.agent_cls(env=self.env, logdir=tempfile.mkdtemp(),
                                buffer_type='PER-v0')
        self.assertEqual(agent.buffer_type, 'PER-v0')

    def test_buffer_type_falls_back_to_cfg(self):
        agent = self.agent_cls(env=self.env, logdir=tempfile.mkdtemp())
        self.assertEqual(agent.buffer_type, self.cfg_buffer_type)

    def test_buffer_size_kwarg_overrides_default(self):
        agent = self.agent_cls(env=self.env, logdir=tempfile.mkdtemp(),
                                buffer_size=321)
        self.assertEqual(agent.buffer.buffer_capacity, 321)

    def test_buffer_size_falls_back_when_absent(self):
        agent = self.agent_cls(env=self.env, logdir=tempfile.mkdtemp())
        # No buffer_size key in the agent cfg -> buffer falls back to its
        # own cfg's buffer_capacity (er.cfg / per.cfg default to 1000000)
        self.assertEqual(agent.buffer.buffer_capacity, 1000000)

    def test_load_model_kwarg_overrides_cfg(self):
        load_path = tempfile.mkdtemp()
        agent = self.agent_cls(env=self.env, logdir=tempfile.mkdtemp(),
                                load_model=load_path)
        self.assertEqual(agent.model_load_path, load_path)

    def test_load_model_falls_back_to_none(self):
        agent = self.agent_cls(env=self.env, logdir=tempfile.mkdtemp())
        self.assertIsNone(agent.model_load_path)


class TestKerasTD3KwargsFallback(AgentKwargsFallbackMixin, unittest.TestCase):
    agent_cls = KerasTD3


class TestKerasDDPGKwargsFallback(AgentKwargsFallbackMixin, unittest.TestCase):
    agent_cls = KerasDDPG


class TestKerasSACKwargsFallback(AgentKwargsFallbackMixin, unittest.TestCase):
    agent_cls = KerasSAC


class TestRunOptAgentKwargs(unittest.TestCase):
    """run_opt() must only forward buffer_type/buffer_size/load_model to
    agents.make() when they are not None; env/logdir are always passed."""

    def _run(self, buffer_type, buffer_size, model_load_path):
        mock_agent = MagicMock()
        with patch('jlab_opt_control.agents.make', return_value=mock_agent) as mock_make:
            run_opt(
                index=0, max_nepisodes=0, max_nsteps=-1,
                agent_id='KerasTD3-v0', env_id='Pendulum-v1',
                logdir=tempfile.mkdtemp(), buffer_type=buffer_type,
                buffer_size=buffer_size, inference_flag=False,
                difficulty=0.08, nepisode_avg=20, model_save_threshold=0.05,
                model_load_path=model_load_path,
            )
        return mock_make

    def test_all_none_omits_optional_kwargs(self):
        mock_make = self._run(None, None, None)
        _, kwargs = mock_make.call_args
        self.assertIn('env', kwargs)
        self.assertIn('logdir', kwargs)
        self.assertNotIn('buffer_type', kwargs)
        self.assertNotIn('buffer_size', kwargs)
        self.assertNotIn('load_model', kwargs)

    def test_buffer_type_forwarded_when_set(self):
        mock_make = self._run('PER-v0', None, None)
        _, kwargs = mock_make.call_args
        self.assertEqual(kwargs['buffer_type'], 'PER-v0')
        self.assertNotIn('buffer_size', kwargs)
        self.assertNotIn('load_model', kwargs)

    def test_buffer_size_forwarded_when_set(self):
        mock_make = self._run(None, 777, None)
        _, kwargs = mock_make.call_args
        self.assertEqual(kwargs['buffer_size'], 777)
        self.assertNotIn('buffer_type', kwargs)
        self.assertNotIn('load_model', kwargs)

    def test_load_model_forwarded_as_load_model_kwarg(self):
        mock_make = self._run(None, None, '/some/model/path')
        _, kwargs = mock_make.call_args
        self.assertEqual(kwargs['load_model'], '/some/model/path')
        self.assertNotIn('buffer_type', kwargs)
        self.assertNotIn('buffer_size', kwargs)

    def test_all_set_forwarded_together(self):
        mock_make = self._run('PER-v0', 555, '/another/path')
        _, kwargs = mock_make.call_args
        self.assertEqual(kwargs['buffer_type'], 'PER-v0')
        self.assertEqual(kwargs['buffer_size'], 555)
        self.assertEqual(kwargs['load_model'], '/another/path')


if __name__ == '__main__':
    unittest.main()
