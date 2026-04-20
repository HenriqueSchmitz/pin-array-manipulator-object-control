from typing import Optional

from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Pose
from pin_array_manipulator_object_control.rewards.base_model import RewardModel



class Distance3DRewardModel(RewardModel):
    def __init__(self, manipulator_config: Optional[PinArrayManipulatorConfig] = None):
        super().__init__(manipulator_config)
        self.last_target_pose = None

    def get_reward(self, target_pose: Pose, object_pose: Pose) -> float:
        # if target_pose != self.last_target_pose:
        #     self.last_target_pose = target_pose
        #     return 100.0
        distance = object_pose.translation_to(target_pose).length()
        return 1.0 - (distance / self.space_size)
