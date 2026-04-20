from abc import ABC, abstractmethod

import numpy as np



class ControlPolicy(ABC):

    @abstractmethod
    def sample(self, target: np.ndarray, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    