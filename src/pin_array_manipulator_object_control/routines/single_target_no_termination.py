import numpy as np
import random
from typing import Optional

from pin_array_manipulator_object_control.routines.target_generator import PinArrayTargetGenerator
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Pose, Object


class SingleTargetNoTerminationGenerator(PinArrayTargetGenerator):
    def __init__(self, 
                 simulation_object: Object, 
                 manipulator_config: Optional[PinArrayManipulatorConfig] = None,
                 margin_factor: float = 0.9):
        super().__init__(simulation_object, manipulator_config)
        self.margin_factor = margin_factor

    def reset(self) -> None:
        pass

    def _generate_target(self, observation: PinArrayEnvObservation) -> Optional[Pose]:
        obj_size = self.simulation_object.get_size()
        safe_half_size = (self.manipulator_config.manipulator_size * self.margin_factor) / 2
        limit_x = safe_half_size - (obj_size.x / 2)
        limit_y = safe_half_size - (obj_size.y / 2)
        limit_x = max(0, limit_x)
        limit_y = max(0, limit_y)
        target_x = random.uniform(-limit_x, limit_x)
        target_y = random.uniform(-limit_y, limit_y)
        target_z = obj_size.z / 2
        return Pose(
            x=target_x, 
            y=target_y, 
            z=target_z, 
            roll=0.0, 
            pitch=0.0, 
            yaw=0.0
        )