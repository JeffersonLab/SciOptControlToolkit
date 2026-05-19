import unittest
import numpy as np
import os
import tempfile
import tensorflow as tf
import jlab_opt_control.models as models
from jlab_opt_control.models.actor_fcnn import ActorFCNN
from jlab_opt_control.models.actor_gaussian_v0 import ActorGaussian
from jlab_opt_control.models.critic_fcnn import CriticFCNN


STATE_DIM  = 4
ACTION_DIM = 2
MIN_ACTION = np.array([-1.0, -1.0], dtype=np.float32)
MAX_ACTION = np.array([ 1.0,  1.0], dtype=np.float32)
BATCH_SIZE = 8


def zeros_state(batch=BATCH_SIZE):
    return tf.zeros([batch, STATE_DIM])


def zeros_action(batch=BATCH_SIZE):
    return tf.zeros([batch, ACTION_DIM])


class TestActorFCNN(unittest.TestCase):

    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.actor = ActorFCNN(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            min_action=MIN_ACTION,
            max_action=MAX_ACTION,
            logdir=self.logdir,
        )

    def test_output_shape(self):
        out = self.actor(zeros_state())
        self.assertEqual(out.shape, (BATCH_SIZE, ACTION_DIM))

    def test_output_within_bounds(self):
        state = tf.random.uniform([BATCH_SIZE, STATE_DIM], -1, 1)
        out = self.actor(state).numpy()
        self.assertTrue(np.all(out >= MIN_ACTION - 1e-5))
        self.assertTrue(np.all(out <= MAX_ACTION + 1e-5))

    def test_output_shape_single(self):
        out = self.actor(zeros_state(batch=1))
        self.assertEqual(out.shape, (1, ACTION_DIM))

    def test_action_scale_bias(self):
        # For symmetric bounds [-1, 1]: scale=1, bias=0
        np.testing.assert_array_almost_equal(
            self.actor.action_scale.numpy(), np.ones(ACTION_DIM))
        np.testing.assert_array_almost_equal(
            self.actor.action_bias.numpy(), np.zeros(ACTION_DIM))

    def test_asymmetric_bounds(self):
        actor = ActorFCNN(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            min_action=np.array([0.0, -2.0], dtype=np.float32),
            max_action=np.array([2.0,  0.0], dtype=np.float32),
            logdir=self.logdir,
        )
        state = tf.random.uniform([BATCH_SIZE, STATE_DIM], -1, 1)
        out = actor(state).numpy()
        self.assertTrue(np.all(out[:, 0] >= 0.0 - 1e-5))
        self.assertTrue(np.all(out[:, 0] <= 2.0 + 1e-5))
        self.assertTrue(np.all(out[:, 1] >= -2.0 - 1e-5))
        self.assertTrue(np.all(out[:, 1] <=  0.0 + 1e-5))

    def test_training_flag_does_not_crash(self):
        out_train = self.actor(zeros_state(), training=True)
        out_infer = self.actor(zeros_state(), training=False)
        self.assertEqual(out_train.shape, out_infer.shape)

    def test_save_cfg(self):
        self.actor.save_cfg()
        cfg_path = os.path.join(self.logdir, 'cfgs', 'actor_fcnn.cfg')
        self.assertTrue(os.path.exists(cfg_path))

    def test_save_cfg_idempotent(self):
        self.actor.save_cfg()
        self.actor.save_cfg()  # second call should not raise
        cfg_path = os.path.join(self.logdir, 'cfgs', 'actor_fcnn.cfg')
        self.assertTrue(os.path.exists(cfg_path))

    def test_registry(self):
        actor = models.make(
            'actor_fcnn-v0',
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            min_action=MIN_ACTION,
            max_action=MAX_ACTION,
            logdir=self.logdir,
        )
        self.assertIsInstance(actor, ActorFCNN)


class TestActorGaussian(unittest.TestCase):

    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.actor = ActorGaussian(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            min_action=MIN_ACTION,
            max_action=MAX_ACTION,
            logdir=self.logdir,
        )

    def test_output_is_tuple(self):
        out = self.actor(zeros_state())
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_action_shape(self):
        action, log_pi = self.actor(zeros_state())
        self.assertEqual(action.shape, (BATCH_SIZE, ACTION_DIM))

    def test_log_pi_shape(self):
        action, log_pi = self.actor(zeros_state())
        self.assertEqual(log_pi.shape, (BATCH_SIZE, 1))

    def test_action_within_bounds(self):
        state = tf.random.uniform([BATCH_SIZE, STATE_DIM], -1, 1)
        action, _ = self.actor(state)
        action = action.numpy()
        self.assertTrue(np.all(action >= MIN_ACTION - 1e-5))
        self.assertTrue(np.all(action <= MAX_ACTION + 1e-5))

    def test_log_pi_is_finite(self):
        state = tf.random.uniform([BATCH_SIZE, STATE_DIM], -1, 1)
        _, log_pi = self.actor(state)
        self.assertTrue(np.all(np.isfinite(log_pi.numpy())))

    def test_stochastic_output(self):
        # Two calls with the same input should generally differ (stochastic)
        state = tf.ones([BATCH_SIZE, STATE_DIM])
        a1, _ = self.actor(state)
        a2, _ = self.actor(state)
        self.assertFalse(np.allclose(a1.numpy(), a2.numpy()))

    def test_save_cfg(self):
        self.actor.save_cfg()
        cfg_path = os.path.join(self.logdir, 'cfgs', 'actor_gaussian.cfg')
        self.assertTrue(os.path.exists(cfg_path))

    def test_registry(self):
        actor = models.make(
            'actor_gaussian-v0',
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            min_action=MIN_ACTION,
            max_action=MAX_ACTION,
            logdir=self.logdir,
        )
        self.assertIsInstance(actor, ActorGaussian)


class TestCriticFCNN(unittest.TestCase):

    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.critic = CriticFCNN(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            logdir=self.logdir,
        )

    def test_output_shape(self):
        out = self.critic(zeros_state(), zeros_action())
        self.assertEqual(out.shape, (BATCH_SIZE, 1))

    def test_output_shape_single(self):
        out = self.critic(zeros_state(1), zeros_action(1))
        self.assertEqual(out.shape, (1, 1))

    def test_output_is_scalar_per_sample(self):
        state  = tf.random.uniform([BATCH_SIZE, STATE_DIM],  -1, 1)
        action = tf.random.uniform([BATCH_SIZE, ACTION_DIM], -1, 1)
        out = self.critic(state, action)
        self.assertEqual(out.shape, (BATCH_SIZE, 1))

    def test_output_is_finite(self):
        state  = tf.random.uniform([BATCH_SIZE, STATE_DIM],  -1, 1)
        action = tf.random.uniform([BATCH_SIZE, ACTION_DIM], -1, 1)
        out = self.critic(state, action).numpy()
        self.assertTrue(np.all(np.isfinite(out)))

    def test_training_flag_does_not_crash(self):
        out_train = self.critic(zeros_state(), zeros_action(), training=True)
        out_infer = self.critic(zeros_state(), zeros_action(), training=False)
        self.assertEqual(out_train.shape, out_infer.shape)

    def test_save_cfg(self):
        self.critic.save_cfg()
        cfg_path = os.path.join(self.logdir, 'cfgs', 'critic_fcnn.cfg')
        self.assertTrue(os.path.exists(cfg_path))

    def test_registry(self):
        critic = models.make(
            'critic_fcnn-v0',
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            logdir=self.logdir,
        )
        self.assertIsInstance(critic, CriticFCNN)

    def test_registry_invalid_id_raises(self):
        with self.assertRaises(Exception):
            models.make('not_a_model-v0', state_dim=STATE_DIM,
                        action_dim=ACTION_DIM, logdir=self.logdir)


class TestModelWeightSaveLoad(unittest.TestCase):
    """Round-trip weight save/load for actor and critic."""

    def setUp(self):
        self.logdir = tempfile.mkdtemp()

    def test_actor_fcnn_weight_roundtrip(self):
        actor = ActorFCNN(STATE_DIM, ACTION_DIM, MIN_ACTION, MAX_ACTION, self.logdir)
        actor(zeros_state())  # initialise weights
        path = os.path.join(self.logdir, 'actor.weights.h5')
        actor.save_weights(path)

        actor2 = ActorFCNN(STATE_DIM, ACTION_DIM, MIN_ACTION, MAX_ACTION, self.logdir)
        actor2(zeros_state())
        actor2.load_weights(path)

        for w1, w2 in zip(actor.get_weights(), actor2.get_weights()):
            np.testing.assert_array_equal(w1, w2)

    def test_critic_fcnn_weight_roundtrip(self):
        critic = CriticFCNN(STATE_DIM, ACTION_DIM, self.logdir)
        critic(zeros_state(), zeros_action())
        path = os.path.join(self.logdir, 'critic.weights.h5')
        critic.save_weights(path)

        critic2 = CriticFCNN(STATE_DIM, ACTION_DIM, self.logdir)
        critic2(zeros_state(), zeros_action())
        critic2.load_weights(path)

        for w1, w2 in zip(critic.get_weights(), critic2.get_weights()):
            np.testing.assert_array_equal(w1, w2)


if __name__ == '__main__':
    unittest.main()
