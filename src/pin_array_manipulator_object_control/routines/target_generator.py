from abc import ABC, abstractmethod
from typing import Optional

from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Object, Pose


OUT_OF_BOUNDS_TEMPLATE = "Generated target out of bounds. Bounds: (x: {:.2f}-{:.2f}, y: {:.2f}-{:.2f}). Generated: {}"



class PinArrayTargetGenerator(ABC):

    def __init__(self, simulation_object: Object, manipulator_config: Optional[PinArrayManipulatorConfig] = None):
        self.simulation_object = simulation_object
        if manipulator_config is None:
            manipulator_config = PinArrayManipulatorConfig()
        self.manipulator_config = manipulator_config
        self.min_x = - self.manipulator_config.manipulator_size/2
        self.max_x = self.manipulator_config.manipulator_size/2
        self.min_y = - self.manipulator_config.manipulator_size/2
        self.max_y = self.manipulator_config.manipulator_size/2
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    def get_current_target(self, observation: PinArrayEnvObservation) -> Optional[Pose]:
        proposed_target = self._generate_target(observation)
        if proposed_target is None:
            return None
        p_x = proposed_target.x
        p_y = proposed_target.y
        if not (self.min_x < p_x < self.max_x and self.min_y < p_y < self.max_y):
            raise ValueError(OUT_OF_BOUNDS_TEMPLATE.format(
                self.min_x,
                self.max_x,
                self.min_y,
                self.max_y,
                str(proposed_target)
            ))
        return proposed_target
    
    @abstractmethod
    def _generate_target(self, observation: PinArrayEnvObservation) -> Optional[Pose]:
        raise NotImplementedError