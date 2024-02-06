from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, **kwargs):
        """ Define all key variables required for all model """
        pass

    @abstractmethod
    def call(self):
        """ forward pass of model """
        pass


