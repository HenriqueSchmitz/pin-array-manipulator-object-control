from abc import ABC, abstractmethod
from typing import Optional

from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Pose



class RewardModel(ABC):

    def __init__(self, manipulator_config: Optional[PinArrayManipulatorConfig] = None):
        if manipulator_config is None:
            manipulator_config = PinArrayManipulatorConfig()
        self.manipulator_config = manipulator_config
        manip_size = self.manipulator_config.manipulator_size
        self.space_size = (manip_size**2 + manip_size**2)**(1/2)

    @abstractmethod
    def get_reward(self, target_pose: Pose, object_pose: Pose) -> float:
        raise NotImplementedError