import random
import numpy as np
from typing import Optional
from pin_array_manipulator_object_control.routines.target_generator import PinArrayTargetGenerator
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Pose, Object

class RobustTargetGenerator(PinArrayTargetGenerator):
    def __init__(self, 
                 simulation_object: Object, 
                 manipulator_config: Optional[PinArrayManipulatorConfig] = None,
                 margin_factor: float = 0.9,
                 min_distance: float = 0.15,  # Minimum distance from start
                 distance_threshold: Optional[float] = None):
        super().__init__(simulation_object, manipulator_config)
        self.margin_factor = margin_factor
        self.min_distance = min_distance
        self.distance_threshold = distance_threshold
        self.current_target_pose: Optional[Pose] = None
        
    def reset(self, seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        self.current_target_pose = self._create_valid_pose(rng)

    def _create_valid_pose(self, rng: np.random.Generator) -> Pose:
        obj_size = self.simulation_object.get_size()
        start_pose = self.simulation_object.get_pose() 
        
        safe_half_size = (self.manipulator_config.manipulator_size * self.margin_factor) / 2
        limit_x = max(0, safe_half_size - (obj_size.x / 2))
        limit_y = max(0, safe_half_size - (obj_size.y / 2))

        for _ in range(100): 
            target = Pose(
                x=rng.uniform(-limit_x, limit_x),
                y=rng.uniform(-limit_y, limit_y),
                z=obj_size.z / 2,
                roll=0.0, pitch=0.0, yaw=0.0
            )
            
            # Use 2D distance to ensure the target is a meaningful distance away
            dist = np.sqrt((target.x - start_pose.x)**2 + (target.y - start_pose.y)**2)
            if dist >= self.min_distance:
                return target
                
        return Pose(
                x=random.uniform(-limit_x, limit_x),
                y=random.uniform(-limit_y, limit_y),
                z=obj_size.z / 2,
                roll=0.0, pitch=0.0, yaw=0.0
            )

    def _generate_target(self, observation: PinArrayEnvObservation) -> Optional[Pose]:
        """Logic for maintaining or terminating the target."""
        dist = observation.object_pose.translation_to(self.current_target_pose).length()
        if dist > self.manipulator_config.manipulator_size * 5: # Fallen off
            return None
        if self.distance_threshold is not None and dist < self.distance_threshold:
            return None
        return self.current_target_pose