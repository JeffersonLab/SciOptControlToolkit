import unittest
import numpy as np
import os
import tempfile
import gymnasium as gym
import jlab_opt_control.agents as agents
from jlab_opt_control.agents.keras_td3 import KerasTD3
from jlab_opt_control.agents.keras_ddpg import KerasDDPG
from jlab_opt_control.agents.keras_sac import KerasSAC


def make_env():
    return gym.make('Pendulum-v1')


def make_agent(cls, env, logdir, buffer_type='ER-v0', buffer_size=500):
    return cls(env=env, logdir=logdir,
               buffer_type=buffer_type, buffer_size=buffer_size)


def random_transition(env):
    state, _ = env.reset()
    action = env.action_space.sample()
    next_state, reward, term, trunc, _ = env.step(action)
    done = term or trunc
    return (state, action, float(reward), next_state, float(done))


class AgentTestMixin:
    """Shared tests run against every agent class."""

    agent_cls = None  # set in subclass

    def setUp(self):
        self.env     = make_env()
        self.logdir  = tempfile.mkdtemp()
        self.agent   = make_agent(self.agent_cls, self.env, self.logdir)

    # --- action() ---

    def test_action_returns_tuple(self):
        state, _ = self.env.reset()
        result = self.agent.action(state)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_action_shape_during_warmup(self):
        # Buffer is empty → warmup random action
        state, _ = self.env.reset()
        action, noise = self.agent.action(state, train=True)
        self.assertEqual(action.shape, (self.env.action_space.shape[0],))

    def test_action_within_bounds(self):
        state, _ = self.env.reset()
        # Fill past warmup so actor is used
        for _ in range(self.agent.warmup_size + 1):
            self.agent.memory(random_transition(self.env) + (1.0,))
        action, _ = self.agent.action(state, train=False)
        np.testing.assert_array_less(
            self.env.action_space.low  - 1e-5, action)
        np.testing.assert_array_less(
            action, self.env.action_space.high + 1e-5)

    # --- memory() ---

    def test_memory_increments_buffer(self):
        before = self.agent.buffer.size()
        self.agent.memory(random_transition(self.env))
        self.assertEqual(self.agent.buffer.size(), before + 1)

    def test_memory_stores_correct_state(self):
        state = np.ones(self.env.observation_space.shape[0], dtype=np.float32) * 5.0
        transition = (state,
                      self.env.action_space.sample(),
                      1.0,
                      state * 2,
                      0.0)
        self.agent.memory(transition)
        idx = self.agent.buffer.pointer - 1
        np.testing.assert_array_almost_equal(
            self.agent.buffer.states[idx], state)

    # --- train() does not crash after warmup ---

    def test_train_does_not_crash_during_warmup(self):
        # Should silently do nothing — buffer not full enough
        self.agent.train()

    def test_train_runs_after_warmup(self):
        for _ in range(self.agent.warmup_size + self.agent.batch_size + 1):
            self.agent.memory(random_transition(self.env))
        self.agent.train()  # should not raise

    # --- soft_update ---

    def test_soft_update_changes_target_weights(self):
        # Initialise weights with a forward pass
        import tensorflow as tf
        state = tf.zeros([1, self.env.observation_space.shape[0]])
        self.agent.actor_model(state)
        self.agent.target_actor(state)

        # Diverge target from source
        new_weights = [np.ones_like(w) * 99.0
                       for w in self.agent.actor_model.get_weights()]
        self.agent.actor_model.set_weights(new_weights)

        old_target = [w.copy()
                      for w in self.agent.target_actor.get_weights()]
        self.agent.soft_update(self.agent.target_actor, self.agent.actor_model)
        new_target = self.agent.target_actor.get_weights()

        changed = any(
            not np.allclose(o, n)
            for o, n in zip(old_target, new_target)
        )
        self.assertTrue(changed)

    def test_soft_update_tau_zero_leaves_target_unchanged(self):
        import tensorflow as tf
        state = tf.zeros([1, self.env.observation_space.shape[0]])
        self.agent.actor_model(state)
        self.agent.target_actor(state)

        original_tau = self.agent.tau
        self.agent.tau = 0.0
        old_target = [w.copy()
                      for w in self.agent.target_actor.get_weights()]
        self.agent.soft_update(self.agent.target_actor, self.agent.actor_model)
        new_target = self.agent.target_actor.get_weights()
        self.agent.tau = original_tau

        for o, n in zip(old_target, new_target):
            np.testing.assert_array_almost_equal(o, n)

    def test_soft_update_tau_one_copies_source(self):
        import tensorflow as tf
        state = tf.zeros([1, self.env.observation_space.shape[0]])
        self.agent.actor_model(state)
        self.agent.target_actor(state)

        original_tau = self.agent.tau
        self.agent.tau = 1.0
        source_weights = [w.copy()
                          for w in self.agent.actor_model.get_weights()]
        self.agent.soft_update(self.agent.target_actor, self.agent.actor_model)
        self.agent.tau = original_tau

        for s, t in zip(source_weights,
                        self.agent.target_actor.get_weights()):
            np.testing.assert_array_almost_equal(s, t)

    # --- save / load round-trip ---

    def test_save_creates_files(self):
        self.agent.save('test')
        model_dir = os.path.join(self.logdir, 'models', 'test')
        self.assertTrue(os.path.isdir(model_dir))
        files = os.listdir(model_dir)
        self.assertGreater(len(files), 0)

    def test_save_cfg(self):
        self.agent.save_cfg()
        cfg_dir = os.path.join(self.logdir, 'cfgs')
        self.assertTrue(os.path.isdir(cfg_dir))
        self.assertGreater(len(os.listdir(cfg_dir)), 0)


class TestKerasTD3Agent(AgentTestMixin, unittest.TestCase):
    agent_cls = KerasTD3


class TestKerasDDPGAgent(AgentTestMixin, unittest.TestCase):
    agent_cls = KerasDDPG

    def test_soft_update_changes_target_weights(self):
        # DDPG has same actor structure — override only to confirm critic too
        super().test_soft_update_changes_target_weights()

        import tensorflow as tf
        state  = tf.zeros([1, self.env.observation_space.shape[0]])
        action = tf.zeros([1, self.env.action_space.shape[0]])
        self.agent.critic_model1(state, action)
        self.agent.target_critic1(state, action)

        new_weights = [np.ones_like(w) * 42.0
                       for w in self.agent.critic_model1.get_weights()]
        self.agent.critic_model1.set_weights(new_weights)
        old_target = [w.copy()
                      for w in self.agent.target_critic1.get_weights()]
        self.agent.soft_update(self.agent.target_critic1,
                               self.agent.critic_model1)
        new_target = self.agent.target_critic1.get_weights()
        changed = any(not np.allclose(o, n)
                      for o, n in zip(old_target, new_target))
        self.assertTrue(changed)


class TestKerasSACAgent(AgentTestMixin, unittest.TestCase):
    agent_cls = KerasSAC

    def test_soft_update_changes_target_weights(self):
        # SAC has no target_actor — override to test critics instead
        import tensorflow as tf
        state  = tf.zeros([1, self.env.observation_space.shape[0]])
        action = tf.zeros([1, self.env.action_space.shape[0]])
        self.agent.critic_model1(state, action)
        self.agent.target_critic1(state, action)

        new_weights = [np.ones_like(w) * 7.0
                       for w in self.agent.critic_model1.get_weights()]
        self.agent.critic_model1.set_weights(new_weights)
        old_target = [w.copy()
                      for w in self.agent.target_critic1.get_weights()]
        self.agent.soft_update(self.agent.target_critic1,
                               self.agent.critic_model1)
        new_target = self.agent.target_critic1.get_weights()
        changed = any(not np.allclose(o, n)
                      for o, n in zip(old_target, new_target))
        self.assertTrue(changed)

    def test_dual_critic_exist(self):
        self.assertIsNotNone(self.agent.critic_model1)
        self.assertIsNotNone(self.agent.critic_model2)
        self.assertIsNotNone(self.agent.target_critic1)
        self.assertIsNotNone(self.agent.target_critic2)

    def test_soft_update_tau_zero_leaves_target_unchanged(self):
        import tensorflow as tf
        state  = tf.zeros([1, self.env.observation_space.shape[0]])
        action = tf.zeros([1, self.env.action_space.shape[0]])
        self.agent.critic_model1(state, action)
        self.agent.target_critic1(state, action)

        original_tau = self.agent.tau
        self.agent.tau = 0.0
        old_target = [w.copy() for w in self.agent.target_critic1.get_weights()]
        self.agent.soft_update(self.agent.target_critic1, self.agent.critic_model1)
        new_target = self.agent.target_critic1.get_weights()
        self.agent.tau = original_tau

        for o, n in zip(old_target, new_target):
            np.testing.assert_array_almost_equal(o, n)

    def test_soft_update_tau_one_copies_source(self):
        import tensorflow as tf
        state  = tf.zeros([1, self.env.observation_space.shape[0]])
        action = tf.zeros([1, self.env.action_space.shape[0]])
        self.agent.critic_model1(state, action)
        self.agent.target_critic1(state, action)

        original_tau = self.agent.tau
        self.agent.tau = 1.0
        source_weights = [w.copy() for w in self.agent.critic_model1.get_weights()]
        self.agent.soft_update(self.agent.target_critic1, self.agent.critic_model1)
        self.agent.tau = original_tau

        for s, t in zip(source_weights, self.agent.target_critic1.get_weights()):
            np.testing.assert_array_almost_equal(s, t)


class TestAgentRegistry(unittest.TestCase):

    def setUp(self):
        self.env    = make_env()
        self.logdir = tempfile.mkdtemp()

    def test_make_td3(self):
        agent = agents.make('KerasTD3-v0', env=self.env, logdir=self.logdir)
        self.assertIsInstance(agent, KerasTD3)

    def test_make_ddpg(self):
        agent = agents.make('KerasDDPG-v0', env=self.env, logdir=self.logdir)
        self.assertIsInstance(agent, KerasDDPG)

    def test_make_sac(self):
        agent = agents.make('KerasSAC-v0', env=self.env, logdir=self.logdir)
        self.assertIsInstance(agent, KerasSAC)

    def test_invalid_id_raises(self):
        with self.assertRaises(Exception):
            agents.make('NotAnAgent-v0', env=self.env, logdir=self.logdir)

    def test_list_registered(self):
        registered = agents.list_registered_modules()
        self.assertIn('KerasTD3-v0',  registered)
        self.assertIn('KerasDDPG-v0', registered)
        self.assertIn('KerasSAC-v0',  registered)


class TestAgentPERBuffer(unittest.TestCase):
    """Agents should work correctly with a PER buffer."""

    def setUp(self):
        self.env    = make_env()
        self.logdir = tempfile.mkdtemp()

    def test_td3_with_per(self):
        agent = KerasTD3(env=self.env, logdir=self.logdir,
                         buffer_type='PER-v0', buffer_size=500)
        for _ in range(10):
            agent.memory(random_transition(self.env))
        agent.train()  # warmup — should not raise

    def test_ddpg_with_per(self):
        agent = KerasDDPG(env=self.env, logdir=self.logdir,
                          buffer_type='PER-v0', buffer_size=500)
        for _ in range(10):
            agent.memory(random_transition(self.env))
        agent.train()


if __name__ == '__main__':
    unittest.main()
