"""NoOpBuffer -- a replay buffer that stores nothing and does nothing.

Exists so buffer-free agents (KerasDEPO, TorchDEPO -- see keras_depo.py's
module docstring for why they have no replay buffer) can still expose
self.buffer and satisfy drivers/run_continuous.py's unconditional
`agent.buffer.save(...)` calls, without that shared driver needing to
special-case buffer-free agents itself.
"""

from jlab_opt_control.core.replay_core import Replay


class NoOpBuffer(Replay):
    def __init__(self, state_dim, action_dim, logdir, buffer_size=None):
        super().__init__(None, None, None, None, None, None)
        self.logdir = logdir

    def record(self, memory):
        pass

    def sample(self, nsamples):
        return None

    def save(self, filename='replay_buffer.npy'):
        pass

    def save_cfg(self):
        """No cfg file -- nothing to configure for a buffer with no state."""
        pass

    def load(self, filename):
        pass

    def size(self):
        return 0
