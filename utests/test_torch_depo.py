import tempfile
import unittest

import numpy as np
import torch
from gymnasium import spaces

import jlab_opt_control.agents as agents
import jlab_opt_control.models as models
from jlab_opt_control.agents.torch_depo import TorchDEPO
from jlab_opt_control.models.torch_actor_fcnn import TorchActorFCNN


class _FakeDiffEnv:
    """Minimal stand-in for fel_dt.envs.FELDigitalTwinBatchEnv -- just
    enough surface for TorchDEPO (action_space/observation_space as Box,
    reset(batch_size=...)/step(actions) returning torch tensors with
    gradients preserved) without depending on fel_dt or a real trained
    model. reward = -sum(state^2), so an actor that pushes state toward 0
    has something real to learn; state = action (stateless, one-step-per-
    reset is enough to exercise TorchDEPO's unroll loop)."""

    def __init__(self, n=3):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self._batch_size = None

    def reset(self, *, batch_size, seed=None, options=None):
        self._batch_size = batch_size
        return torch.zeros(batch_size, self.action_space.shape[0]), {}

    def step(self, actions):
        state = actions
        reward = -(state ** 2).sum(dim=-1)
        terminated = torch.zeros(self._batch_size, dtype=torch.bool)
        truncated = torch.ones(self._batch_size, dtype=torch.bool)
        return state, reward, terminated, truncated, {}


class TestTorchActorFCNN(unittest.TestCase):

    def test_output_shape_and_bounds(self):
        actor = models.make(
            'actor_fcnn_torch-v0', state_dim=4, action_dim=2,
            min_action=np.array([-0.5, -0.2]), max_action=np.array([0.5, 0.2]),
            logdir=tempfile.mkdtemp(),
        )
        out = actor(torch.randn(5, 4))
        self.assertEqual(tuple(out.shape), (5, 2))
        self.assertTrue(torch.all(out >= -0.5 - 1e-5))
        self.assertTrue(torch.all(out[:, 1] <= 0.2 + 1e-5))


class TestTorchDEPO(unittest.TestCase):

    def setUp(self):
        self.env = _FakeDiffEnv(n=3)
        self.logdir = tempfile.mkdtemp()
        self.agent = agents.make(
            'TorchDEPO-v0', env=self.env, logdir=self.logdir,
            unroll_steps=1, discount=0.99,
        )

    def test_action_returns_a_tuple_within_bounds(self):
        state = np.zeros(3, dtype=np.float32)
        action, noise = self.agent.action(state, train=False)
        self.assertEqual(action.shape, (3,))
        np.testing.assert_array_less(self.env.action_space.low - 1e-5, action)
        np.testing.assert_array_less(action, self.env.action_space.high + 1e-5)

    def test_train_returns_a_tensor_and_moves_the_actor_weights(self):
        before = {k: v.clone() for k, v in self.agent.actor_model.state_dict().items()}

        result = self.agent.train()

        self.assertIsInstance(result, torch.Tensor)
        after = self.agent.actor_model.state_dict()
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))

    def test_repeated_training_reduces_state_squared_norm(self):
        # reward = -sum(state^2), state = action -- a working actor should
        # learn to output actions near 0, i.e. the discounted return
        # (== reward here, single-step) should trend toward 0 (its max).
        returns = [self.agent.train().item() for _ in range(30)]
        self.assertGreater(np.mean(returns[-5:]), np.mean(returns[:5]))

    def test_memory_and_soft_update_are_safe_no_ops(self):
        self.agent.memory((None, None, None, None, None))
        self.agent.soft_update()  # must not raise

    def test_save_then_load_round_trips_actor_weights_into_a_fresh_agent(self):
        for _ in range(5):
            self.agent.train()  # move weights away from fresh-init defaults
        self.agent.save(post_fix="final")
        saved_state = {k: v.clone() for k, v in self.agent.actor_model.state_dict().items()}

        loaded_agent = agents.make(
            'TorchDEPO-v0', env=self.env, logdir=tempfile.mkdtemp(),
            unroll_steps=1, discount=0.99, load_model=f"{self.logdir}/models/final",
        )

        loaded_state = loaded_agent.actor_model.state_dict()
        for key in saved_state:
            self.assertTrue(torch.equal(saved_state[key], loaded_state[key]), f"mismatch at {key}")

    def test_load_raises_instead_of_silently_keeping_random_weights(self):
        with self.assertRaises(Exception):
            agents.make(
                'TorchDEPO-v0', env=self.env, logdir=tempfile.mkdtemp(),
                unroll_steps=1, discount=0.99, load_model="/no/such/checkpoint/dir",
            )


if __name__ == "__main__":
    unittest.main()