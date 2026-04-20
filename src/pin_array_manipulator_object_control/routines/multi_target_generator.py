import numpy as np
import random
import time
from typing import Optional, List
from pin_array_manipulator_object_control.routines.target_generator import PinArrayTargetGenerator
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Pose, Object

class MultiTargetGenerator(PinArrayTargetGenerator):
    def __init__(self, 
                 simulation_object: Object, 
                 manipulator_config: Optional[PinArrayManipulatorConfig] = None,
                 distance_threshold: float = 0.01,
                 targets_to_generate: int = 5,
                 margin_factor: float = 0.8):
        super().__init__(simulation_object, manipulator_config)
        self.distance_threshold = distance_threshold
        self.targets_to_generate = targets_to_generate
        self.margin_factor = margin_factor
        self.current_target_pose: Optional[Pose] = None
        self.targets_generated = 0

    def reset(self) -> None:
        self.current_target_pose = self._pick_random_pose()

    def _pick_random_pose(self) -> Optional[Pose]:
        self.targets_generated += 1
        if self.targets_generated >= self.targets_to_generate:
            return None
        obj_size = self.simulation_object.get_size()
        safe_half_size = (self.manipulator_config.manipulator_size * self.margin_factor) / 2
        limit = max(0, safe_half_size - (obj_size.x / 2))
        return Pose(
            x=random.uniform(-limit, limit),
            y=random.uniform(-limit, limit),
            z=obj_size.z / 2,
            roll=0.0, pitch=0.0, yaw=0.0
        )

    def _generate_target(self, observation: PinArrayEnvObservation) -> Optional[Pose]:
        if self.current_target_pose is not None:
            dist = np.linalg.norm(
                observation.object_pose.array()[:3] - self.current_target_pose.array()[:3]
            )
            if dist < self.distance_threshold:
                self.current_target_pose = self._pick_random_pose()
        elif self.targets_generated < self.targets_to_generate:
            self.current_target_pose = self._pick_random_pose()
        return self.current_target_pose