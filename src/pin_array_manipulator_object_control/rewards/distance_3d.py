from typing import Optional

from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Pose
from pin_array_manipulator_object_control.rewards.base_model import RewardModel



class Distance3DRewardModel(RewardModel):
    def __init__(self, manipulator_config: Optional[PinArrayManipulatorConfig] = None):
        super().__init__(manipulator_config)

    def get_reward(self, target_pose: Pose, object_pose: Pose) -> float:
        distance = object_pose.translation_to(target_pose).length()
        return 1.0 - (distance / self.space_size)
