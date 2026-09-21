import tempfile
import unittest

import numpy as np
import tensorflow as tf
from gymnasium import spaces

import jlab_opt_control.agents as agents
from jlab_opt_control.agents.keras_depo import KerasDEPO


class _FakeDiffEnv:
    """Minimal stand-in for a differentiable env (see
    jlab_opt_control/envs/circle_env.py for a plain, non-differentiable
    Gym env, and keras_depo.py's module docstring for the contract this
    fakes) -- just enough surface for KerasDEPO (action_space/
    observation_space as Box, reset(batch_size=...)/step(actions)
    returning TF tensors with gradients preserved) without depending on a
    real trained differentiable model. reward = -sum(state^2), so an actor
    that pushes state toward 0 has something real to learn; state = action
    (stateless, one-step-per-reset is enough to exercise KerasDEPO's
    unroll loop). reset() returns a nonzero state -- Keras Dense layers
    zero-init their biases, so with an all-zero state an untrained
    ActorFCNN deterministically outputs exactly 0, which is already this
    env's optimum and leaves no gradient signal to learn from."""

    def __init__(self, n=3):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self._batch_size = None

    def reset(self, *, batch_size, seed=None, options=None):
        self._batch_size = batch_size
        return tf.fill([batch_size, self.action_space.shape[0]], 0.5), {}

    def step(self, actions):
        state = actions
        reward = -tf.reduce_sum(tf.square(state), axis=-1)
        terminated = tf.zeros([self._batch_size], dtype=tf.bool)
        truncated = tf.ones([self._batch_size], dtype=tf.bool)
        return state, reward, terminated, truncated, {}


class TestKerasDEPO(unittest.TestCase):

    def setUp(self):
        self.env = _FakeDiffEnv(n=3)
        self.logdir = tempfile.mkdtemp()
        self.agent = agents.make(
            'KerasDEPO-v0', env=self.env, logdir=self.logdir,
            unroll_steps=1, discount=0.99,
        )

    def test_action_returns_a_tuple_within_bounds(self):
        state = np.zeros(3, dtype=np.float32)
        action, noise = self.agent.action(state, train=False)
        self.assertEqual(action.shape, (3,))
        np.testing.assert_array_less(self.env.action_space.low - 1e-5, action)
        np.testing.assert_array_less(action, self.env.action_space.high + 1e-5)

    def test_train_returns_a_tensor_and_moves_the_actor_weights(self):
        before = [w.numpy().copy() for w in self.agent.actor_model.trainable_variables]

        result = self.agent.train()

        self.assertTrue(tf.is_tensor(result) or isinstance(result, np.floating))
        after = self.agent.actor_model.trainable_variables
        self.assertTrue(any(
            not np.array_equal(b, a.numpy()) for b, a in zip(before, after)
        ))

    def test_repeated_training_reduces_state_squared_norm(self):
        # reward = -sum(state^2), state = action -- a working actor should
        # learn to output actions near 0, i.e. the discounted return
        # (== reward here, single-step) should trend toward 0 (its max).
        returns = [float(self.agent.train()) for _ in range(30)]
        self.assertGreater(np.mean(returns[-5:]), np.mean(returns[:5]))

    def test_memory_and_soft_update_are_safe_no_ops(self):
        self.agent.memory((None, None, None, None, None))
        self.agent.soft_update()  # must not raise

    def test_save_then_load_round_trips_actor_weights(self):
        # Regression test for the save()/load() filename mismatch: save()
        # writes into <logdir>/models/<post_fix>/actor_model_<post_fix>.weights.h5,
        # so load() must be pointed at that directory and must actually
        # find and load the file it wrote there.
        self.agent.train()
        self.agent.save(post_fix="ckpt")
        saved_weights = [w.numpy().copy() for w in self.agent.actor_model.trainable_variables]

        other_env = _FakeDiffEnv(n=3)
        loaded_agent = KerasDEPO(
            other_env, tempfile.mkdtemp(),
            load_model=f"{self.logdir}/models/ckpt/",
            unroll_steps=1, discount=0.99,
        )

        loaded_weights = loaded_agent.actor_model.trainable_variables
        for saved, loaded in zip(saved_weights, loaded_weights):
            np.testing.assert_array_equal(saved, loaded.numpy())


if __name__ == '__main__':
    unittest.main()
