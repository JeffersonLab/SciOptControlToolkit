from abc import ABC, abstractmethod


class Replay(ABC):
    def __init__(self, state, action, reward, next_state, done, probability):
        """ Define all key variables required for all buffers """
        pass

    @abstractmethod
    def record(self, memory):
        """ Add new entries into the buffer """
        pass

    @abstractmethod
    def sample(self, nsamples):
        """ return nsamples based on probability distribution"""
        pass

    @abstractmethod
    def save(self, filename='replay_buffer.npy'):
        """ Save buffer to file """
        pass

    @abstractmethod
    def save_cfg(self):
        """ Save the buffer cfg """
        pass

    @abstractmethod
    def load(self, filename):
        """ Load previous experiences into buffer """
        pass

    @abstractmethod
    def size(self):
        """ Return number of memories in buffer """
        pass
