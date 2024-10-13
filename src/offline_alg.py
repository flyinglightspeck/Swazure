from abc import ABC, abstractmethod
from pandas import DataFrame
from orchestrator import Orchestrator


class OfflineAlg(ABC):
    orc: Orchestrator
    df: DataFrame

    def __init__(self, orc: Orchestrator):
        self.orc = orc

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def store(self):
        pass
