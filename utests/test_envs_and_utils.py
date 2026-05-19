import unittest
import numpy as np
import jlab_opt_control.envs as envs
from jlab_opt_control.envs.circle_env import Circle2D
from jlab_opt_control.utils.cfg_utils import cfg_get


class TestCfgUtils(unittest.TestCase):

    def test_get_existing_key(self):
        data = {'learning_rate': '0.001', 'batch_size': 64}
        self.assertEqual(cfg_get(data, 'batch_size', 0), 64)

    def test_get_string_value(self):
        data = {'name': 'KerasTD3'}
        self.assertEqual(cfg_get(data, 'name', ''), 'KerasTD3')

    def test_get_missing_key_returns_default(self):
        data = {'a': 1}
        self.assertEqual(cfg_get(data, 'missing', 42), 42)

    def test_get_missing_key_default_none(self):
        data = {}
        self.assertIsNone(cfg_get(data, 'missing'))

    def test_get_falsy_value_not_replaced_by_default(self):
        data = {'flag': 0}
        self.assertEqual(cfg_get(data, 'flag', 99), 0)

    def test_get_nested_dict_value(self):
        data = {'layers': [256, 256]}
        result = cfg_get(data, 'layers', [])
        self.assertEqual(result, [256, 256])


class TestCircle2DEnv(unittest.TestCase):

    def _make_env(self, rdm_reset_mode='fixed', statefull=True, max_episode_steps=1):
        return Circle2D(rdm_reset_mode=rdm_reset_mode, statefull=statefull, max_episode_steps=max_episode_steps)

    # --- spaces ---

    def test_action_space_shape(self):
        env = self._make_env()
        self.assertEqual(env.action_space.shape, (2,))

    def test_observation_space_shape(self):
        env = self._make_env()
        self.assertEqual(env.observation_space.shape, (2,))

    def test_action_bounds(self):
        env = self._make_env()
        np.testing.assert_array_equal(env.action_space.low,  [-1., -1.])
        np.testing.assert_array_equal(env.action_space.high, [ 1.,  1.])

    # --- reset ---

    def test_reset_returns_obs_and_info(self):
        env = self._make_env()
        result = env.reset()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_reset_fixed_mode_returns_zeros(self):
        env = self._make_env(rdm_reset_mode='fixed')
        state, _ = env.reset()
        np.testing.assert_array_equal(state, np.zeros(2))

    def test_reset_uniform_mode_returns_array(self):
        env = self._make_env(rdm_reset_mode='uniform')
        state, _ = env.reset()
        self.assertEqual(state.shape, (2,))

    # --- step ---

    def test_step_returns_five_tuple(self):
        env = self._make_env()
        env.reset()
        result = env.step(env.action_space.sample())
        self.assertEqual(len(result), 5)

    def test_step_reward_is_positive(self):
        # Reward is always exp-based, so always > 0
        env = self._make_env()
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.67, 0.67]))
        self.assertGreater(reward, 0.0)

    def test_step_reward_at_target_is_high(self):
        # Moving close to radius=0.95 from origin should yield a high reward
        env = self._make_env(statefull=False)
        env.reset()
        # Action that places agent near circle of radius 0.95
        action = np.array([0.95 / np.sqrt(2), 0.95 / np.sqrt(2)])
        _, reward, _, _, _ = env.step(action)
        self.assertGreater(reward, 1.0)

    def test_max_episode_steps_triggers_done(self):
        env = self._make_env(max_episode_steps=3)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
            if steps > 10:
                break
        self.assertLessEqual(steps, 3)

    def test_statefull_accumulates(self):
        env = self._make_env(statefull=True)
        env.reset()
        action = np.array([0.5, 0.0])
        state1, _, _, _, _ = env.step(action)
        state2, _, _, _, _ = env.step(action)
        # Each step adds the action to state
        self.assertAlmostEqual(float(state2[0]), float(state1[0]) + 0.5, places=5)

    def test_stateless_ignores_history(self):
        env = self._make_env(statefull=False)
        env.reset()
        action = np.array([0.3, 0.4])
        state1, _, _, _, _ = env.step(action)
        state2, _, _, _, _ = env.step(action)
        # Stateless: state == last action, both steps same action → same state
        np.testing.assert_array_almost_equal(state1, state2)

    def test_nsteps_resets_after_env_reset(self):
        env = self._make_env(max_episode_steps=2)
        env.reset()
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.reset()
        # After reset, a new episode should run to 2 steps again
        _, _, term1, trunc1, _ = env.step(env.action_space.sample())
        self.assertFalse(term1 or trunc1)
        _, _, term2, trunc2, _ = env.step(env.action_space.sample())
        self.assertTrue(term2 or trunc2)

    # --- registry ---

    def test_registry_statefull(self):
        env = envs.make('DnC2s-Circle2D-Statefull-v0')
        self.assertIsInstance(env, Circle2D)
        self.assertTrue(env.statefull)

    def test_registry_stateless(self):
        env = envs.make('DnC2s-Circle2D-Stateless-v0')
        self.assertIsInstance(env, Circle2D)
        self.assertFalse(env.statefull)

    def test_registry_invalid_id_raises(self):
        with self.assertRaises(Exception):
            envs.make('NotAnEnv-v0')

    def test_list_registered_envs(self):
        registered = envs.list_registered_modules()
        self.assertIn('DnC2s-Circle2D-Statefull-v0', registered)
        self.assertIn('DnC2s-Circle2D-Stateless-v0', registered)

    # --- action space sampling stays in bounds ---

    def test_sampled_action_in_bounds(self):
        env = self._make_env()
        for _ in range(20):
            action = env.action_space.sample()
            self.assertTrue(np.all(action >= -1.0))
            self.assertTrue(np.all(action <=  1.0))


if __name__ == '__main__':
    unittest.main()
