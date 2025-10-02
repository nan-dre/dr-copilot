from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass
        