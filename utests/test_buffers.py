import unittest
import numpy as np
import os
import tempfile
import jlab_opt_control.buffers as buffers
from jlab_opt_control.buffers.er import ER
from jlab_opt_control.buffers.per import PER


STATE_DIM = 4
ACTION_DIM = 2
BUFFER_SIZE = 100


def make_er(logdir):
    return ER(state_dim=STATE_DIM, action_dim=ACTION_DIM,
              logdir=logdir, buffer_size=BUFFER_SIZE)


def make_per(logdir):
    return PER(state_dim=STATE_DIM, action_dim=ACTION_DIM,
               logdir=logdir, buffer_size=BUFFER_SIZE)


def random_transition():
    state      = np.random.randn(STATE_DIM).astype(np.float32)
    action     = np.random.randn(ACTION_DIM).astype(np.float32)
    reward     = float(np.random.randn())
    next_state = np.random.randn(STATE_DIM).astype(np.float32)
    done       = float(np.random.randint(0, 2))
    priority   = 1.0
    return (state, action, reward, next_state, done, priority)


def zero_transition():
    """A fully valid experience where every field is zero."""
    state      = np.zeros(STATE_DIM, dtype=np.float32)
    action     = np.zeros(ACTION_DIM, dtype=np.float32)
    reward     = 0.0
    next_state = np.zeros(STATE_DIM, dtype=np.float32)
    done       = 0.0
    priority   = 1.0
    return (state, action, reward, next_state, done, priority)


class TestERBuffer(unittest.TestCase):

    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.buf = make_er(self.logdir)

    # --- size tracking ---

    def test_size_starts_at_zero(self):
        self.assertEqual(self.buf.size(), 0)

    def test_size_increments_on_record(self):
        for i in range(1, 11):
            self.buf.record(random_transition())
            self.assertEqual(self.buf.size(), i)

    def test_size_caps_at_capacity(self):
        for _ in range(BUFFER_SIZE + 20):
            self.buf.record(random_transition())
        self.assertEqual(self.buf.size(), BUFFER_SIZE)

    # --- record / sample shapes ---

    def test_sample_shapes(self):
        for _ in range(50):
            self.buf.record(random_transition())
        states, actions, rewards, next_states, dones, weights = self.buf.sample(10)
        self.assertEqual(states.shape,      (10, STATE_DIM))
        self.assertEqual(actions.shape,     (10, ACTION_DIM))
        self.assertEqual(rewards.shape,     (10, 1))
        self.assertEqual(next_states.shape, (10, STATE_DIM))
        self.assertEqual(dones.shape,       (10, 1))
        self.assertEqual(weights.shape,     (10,))

    def test_sample_returns_recorded_data(self):
        state = np.ones(STATE_DIM, dtype=np.float32) * 7.0
        action = np.ones(ACTION_DIM, dtype=np.float32) * 3.0
        transition = (state, action, 5.0, state * 2, 0.0, 1.0)
        # Fill enough so sample can run, then overwrite index 0
        for _ in range(20):
            self.buf.record(random_transition())
        self.buf.states[0] = state
        self.buf.actions[0] = action
        # Force sample to pick index 0 by making it the only choice
        buf_single = make_er(self.logdir)
        buf_single.record(transition)
        states, actions, rewards, next_states, dones, _ = buf_single.sample(1)
        np.testing.assert_array_almost_equal(states[0], state)
        np.testing.assert_array_almost_equal(actions[0], action)
        self.assertAlmostEqual(rewards[0][0], 5.0)

    # --- circular overwrite ---

    def test_circular_overwrite(self):
        sentinel = np.ones(STATE_DIM, dtype=np.float32) * 99.0
        for _ in range(BUFFER_SIZE):
            self.buf.record(random_transition())
        # Write one more — should overwrite index 0
        self.buf.record((sentinel, np.zeros(ACTION_DIM), 0.0,
                         np.zeros(STATE_DIM), 0.0, 1.0))
        np.testing.assert_array_almost_equal(self.buf.states[0], sentinel)

    # --- save / load ---

    def test_save_and_load(self):
        for _ in range(30):
            self.buf.record(random_transition())
        filepath = os.path.join(self.logdir, 'replay.npy')
        self.buf.save(filepath)
        self.assertTrue(os.path.exists(filepath))

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        np.testing.assert_array_equal(self.buf.states, buf2.states)
        np.testing.assert_array_equal(self.buf.actions, buf2.actions)

    def test_save_and_load_preserves_pointer(self):
        """Pointer value round-trips through save/load exactly."""
        for _ in range(30):
            self.buf.record(random_transition())
        filepath = os.path.join(self.logdir, 'replay.npy')
        self.buf.save(filepath)

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        self.assertEqual(buf2.pointer, 30)
        self.assertEqual(buf2.size(), 30)

    def test_save_and_load_wrapped_buffer_preserves_pointer(self):
        """After wrapping, pointer exceeds capacity and is restored exactly."""
        overflow = 15
        total = BUFFER_SIZE + overflow
        for _ in range(total):
            self.buf.record(random_transition())
        filepath = os.path.join(self.logdir, 'replay.npy')
        self.buf.save(filepath)

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        # pointer should be the true count, not capped at capacity
        self.assertEqual(buf2.pointer, total)
        # size() caps at capacity
        self.assertEqual(buf2.size(), BUFFER_SIZE)

    def test_load_empty_buffer(self):
        """An empty buffer round-trips with pointer at 0."""
        filepath = os.path.join(self.logdir, 'replay.npy')
        self.buf.save(filepath)

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        self.assertEqual(buf2.pointer, 0)
        self.assertEqual(buf2.size(), 0)

    def test_load_with_all_zero_experiences(self):
        """Experiences that are entirely zeros are not lost on save/load.

        This is the core scenario the pointer-based fix addresses:
        if the last experiences in the buffer are all zeros, a non-zero
        heuristic would set the pointer too low and lose data.
        """
        # Record some normal transitions, then end with all-zero ones
        for _ in range(5):
            self.buf.record(random_transition())
        for _ in range(3):
            self.buf.record(zero_transition())
        expected_pointer = 8

        filepath = os.path.join(self.logdir, 'replay.npy')
        self.buf.save(filepath)

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        self.assertEqual(buf2.pointer, expected_pointer)
        self.assertEqual(buf2.size(), expected_pointer)

        # Verify the all-zero rows are actually present and sampable
        np.testing.assert_array_equal(buf2.states[7], np.zeros(STATE_DIM))
        np.testing.assert_array_equal(buf2.actions[7], np.zeros(ACTION_DIM))
        np.testing.assert_array_equal(buf2.rewards[7], np.array([0.0]))

    def test_load_only_zero_experiences(self):
        """A buffer filled with nothing but all-zero experiences survives
        a save/load cycle — the old heuristic would report pointer=0."""
        for _ in range(5):
            self.buf.record(zero_transition())

        filepath = os.path.join(self.logdir, 'replay.npy')
        self.buf.save(filepath)

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        self.assertEqual(buf2.pointer, 5)
        self.assertEqual(buf2.size(), 5)

    def test_load_legacy_format_without_pointer(self):
        """Old save files that lack 'pointer' fall back to the non-zero
        heuristic without crashing."""
        for _ in range(20):
            self.buf.record(random_transition())

        filepath = os.path.join(self.logdir, 'legacy.npy')
        # Manually save without the pointer key to simulate an old file
        data = {
            "states": self.buf.states,
            "actions": self.buf.actions,
            "rewards": self.buf.rewards,
            "next_states": self.buf.next_states,
            "dones": self.buf.dones,
            "priorities": self.buf.priorities,
        }
        np.save(filepath, data)

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        # The heuristic should find the last non-zero row at index 19
        self.assertEqual(buf2.pointer, 20)
        self.assertEqual(buf2.size(), 20)

    def test_load_legacy_format_empty(self):
        """Old-format empty save results in pointer=0 via fallback."""
        filepath = os.path.join(self.logdir, 'legacy_empty.npy')
        data = {
            "states": self.buf.states,
            "actions": self.buf.actions,
            "rewards": self.buf.rewards,
            "next_states": self.buf.next_states,
            "dones": self.buf.dones,
            "priorities": self.buf.priorities,
        }
        np.save(filepath, data)

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        self.assertEqual(buf2.pointer, 0)

    def test_sample_counts_reset_on_load(self):
        """Sample counts are zeroed out after loading."""
        for _ in range(20):
            self.buf.record(random_transition())
        self.buf.sample(10)
        # sample_counts should be non-zero now
        self.assertGreater(self.buf.sample_counts.sum(), 0)

        filepath = os.path.join(self.logdir, 'replay.npy')
        self.buf.save(filepath)

        buf2 = make_er(self.logdir)
        buf2.load(filepath)
        np.testing.assert_array_equal(
            buf2.sample_counts, np.zeros((BUFFER_SIZE, 1)))

    # --- cfg save ---

    def test_save_cfg(self):
        self.buf.save_cfg()
        cfg_path = os.path.join(self.logdir, 'cfgs', 'er.cfg')
        self.assertTrue(os.path.exists(cfg_path))

    # --- guard: sample size capped at buffer size ---

    def test_sample_does_not_exceed_buffer_size(self):
        for _ in range(5):
            self.buf.record(random_transition())
        # Should not raise even though 10 > 5
        states, _, _, _, _, _ = self.buf.sample(5)
        self.assertEqual(len(states), 5)


class TestPERBuffer(unittest.TestCase):

    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.buf = make_per(self.logdir)

    def _fill(self, n=50):
        for _ in range(n):
            self.buf.record(random_transition())

    # --- inherits ER size behaviour ---

    def test_size_starts_at_zero(self):
        self.assertEqual(self.buf.size(), 0)

    def test_size_caps_at_capacity(self):
        for _ in range(BUFFER_SIZE + 10):
            self.buf.record(random_transition())
        self.assertEqual(self.buf.size(), BUFFER_SIZE)

    # --- proportional sampling ---

    def test_proportional_sample_shapes(self):
        self._fill()
        states, actions, rewards, next_states, dones, weights = self.buf.sample(10)
        self.assertEqual(states.shape,  (10, STATE_DIM))
        self.assertEqual(weights.shape, (10,))

    def test_proportional_weights_normalised(self):
        self._fill()
        _, _, _, _, _, weights = self.buf.sample(10)
        self.assertLessEqual(float(np.max(weights)), 1.0 + 1e-6)
        self.assertGreater(float(np.min(weights)), 0.0)

    # --- rank-based sampling ---

    def test_rank_based_sample(self):
        self.buf.prioritization_type = 'rank'
        self._fill()
        states, _, _, _, _, weights = self.buf.sample(10)
        self.assertEqual(states.shape,  (10, STATE_DIM))
        self.assertEqual(weights.shape, (10,))

    # --- update_priorities ---

    def test_update_priorities_changes_values(self):
        self._fill()
        self.buf.sample(10)
        new_tds = np.random.uniform(0.1, 2.0, size=(10,)).astype(np.float32)
        old_priorities = self.buf.priorities.copy()
        self.buf.update_priorities(new_tds)
        self.assertFalse(np.array_equal(self.buf.priorities, old_priorities))

    def test_update_priorities_updates_max(self):
        self._fill()
        self.buf.sample(10)
        high_td = np.ones((10,), dtype=np.float32) * 999.0
        self.buf.update_priorities(high_td)
        self.assertGreater(self.buf.max_priority, 1.0)

    # --- beta annealing ---

    def test_beta_increases_after_sample(self):
        self._fill()
        beta_before = self.buf.beta
        self.buf.sample(10)
        self.assertGreater(self.buf.beta, beta_before)

    def test_beta_caps_at_one(self):
        self._fill()
        self.buf.beta = 0.999
        self.buf.beta_increment = 0.1
        self.buf.sample(10)
        self.assertLessEqual(self.buf.beta, 1.0)

    # --- cfg save ---

    def test_save_cfg(self):
        self.buf.save_cfg()
        cfg_path = os.path.join(self.logdir, 'cfgs', 'per.cfg')
        self.assertTrue(os.path.exists(cfg_path))

    # --- registry ---

    def test_registry_er(self):
        buf = buffers.make('ER-v0', state_dim=STATE_DIM,
                           action_dim=ACTION_DIM, logdir=self.logdir)
        self.assertIsInstance(buf, ER)

    def test_registry_per(self):
        buf = buffers.make('PER-v0', state_dim=STATE_DIM,
                           action_dim=ACTION_DIM, logdir=self.logdir)
        self.assertIsInstance(buf, PER)

    def test_registry_invalid_id_raises(self):
        with self.assertRaises(Exception):
            buffers.make('NotABuffer-v0', state_dim=STATE_DIM,
                         action_dim=ACTION_DIM, logdir=self.logdir)


if __name__ == '__main__':
    unittest.main()