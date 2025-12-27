from abc import ABC, abstractmethod

class Model(ABC):    
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass
