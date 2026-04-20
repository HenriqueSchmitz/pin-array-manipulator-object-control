import numpy as np
import random
from typing import Optional

from pin_array_manipulator_object_control.routines.target_generator import PinArrayTargetGenerator
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Pose, Object


class SingleTargetGenerator(PinArrayTargetGenerator):
    def __init__(self, 
                 simulation_object: Object, 
                 manipulator_config: Optional[PinArrayManipulatorConfig] = None,
                 margin_factor: float = 0.9,
                 distance_threshold: Optional[float] = None):
        super().__init__(simulation_object, manipulator_config)
        self.margin_factor = margin_factor
        self.distance_threshold = distance_threshold
        self.current_target_pose: Optional[Pose] = None
        
    def reset(self) -> None:
        self.current_target_pose = self._create_random_pose()

    def _create_random_pose(self) -> Pose:
        obj_size = self.simulation_object.get_size()
        safe_half_size = (self.manipulator_config.manipulator_size * self.margin_factor) / 2
        limit_x = max(0, safe_half_size - (obj_size.x / 2))
        limit_y = max(0, safe_half_size - (obj_size.y / 2))
        return Pose(
            x=random.uniform(-limit_x, limit_x),
            y=random.uniform(-limit_y, limit_y),
            z=obj_size.z / 2,
            roll=0.0, pitch=0.0, yaw=0.0
        )

    def _generate_target(self, observation: PinArrayEnvObservation) -> Optional[Pose]:
        dist = observation.object_pose.translation_to(self.current_target_pose).length()
        if self._has_fallen_from_manipulator(dist):
            return None
        if self.distance_threshold is not None:
            if dist < self.distance_threshold:
                return None
        return self.current_target_pose
    
    def _has_fallen_from_manipulator(self, dist: float) -> bool:
        return dist > self.manipulator_config.manipulator_size * 20