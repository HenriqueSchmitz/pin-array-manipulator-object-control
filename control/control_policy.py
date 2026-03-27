from abc import ABC, abstractmethod

import torch
from torch import Tensor

from objects.object import Pose



class EnvironmentState():
    def __init__(self, object_pose: Pose, target_pose: Pose):
        self.object_pose = object_pose
        self.target_pose = target_pose


class ControlPolicy(ABC):
    def __init__(self):
        self.control_tensor = torch.empty(0)

    @abstractmethod
    def update(self, state: EnvironmentState) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def get_control_tensor(self) -> Tensor:
        return self.control_tensor