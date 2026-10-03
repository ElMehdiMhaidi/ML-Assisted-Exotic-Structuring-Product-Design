from abc import ABC, abstractmethod
class BasePricer(ABC):
    @abstractmethod
    def price(self, spec, market, paths=50000, seed=42):
        ...
