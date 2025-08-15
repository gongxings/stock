from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    name = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame):
        pass
