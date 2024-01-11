from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, **kwargs):
        """ Define all key variables required for all agent """
        pass

    @abstractmethod
    def soft_update(self):
        """ Do a soft update of the target model """
        pass

    @abstractmethod
    def train(self):
        """ Method used to train """
        pass

    @abstractmethod
    def action(self, state):
        """ Method used to provide the next action """
        pass

    @abstractmethod
    def load(self):
        """ Load the ML models """
        pass

    @abstractmethod
    def save(self):
        """ Save the ML models """
        pass
