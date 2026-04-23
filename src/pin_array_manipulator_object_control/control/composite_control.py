import numpy as np
from scipy.ndimage import binary_dilation

from pin_array_manipulator_object_control.control.control_policy import ControlPolicy
from pin_array_manipulator_object_control.control.pose_shift_control import PoseShiftControlPolicy
from pin_array_manipulator_object_control.control.contact_seeking import ContactSeekingPolicy
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig



class CompositeControlPolicy(ControlPolicy):
    def __init__(self, manipulator_config: PinArrayManipulatorConfig, 
                 base_seek_speed=0.0002, 
                 min_seek_speed=0.0001):
        self.pins_per_side = manipulator_config.pins_per_side
        self.num_pins = self.pins_per_side ** 2
        self.contact_seeking_policy = ContactSeekingPolicy(manipulator_config, base_seek_speed, min_seek_speed)
        self.pose_shift_control_policy = PoseShiftControlPolicy(manipulator_config)

    def update_contact_seeking_speeds(self, base_seek_speed, min_seek_speed):
        self.contact_seeking_policy.base_seek_speed = base_seek_speed
        self.contact_seeking_policy.min_seek_speed = min_seek_speed

    def sample(self, target: np.ndarray, observation: np.ndarray) -> np.ndarray:
        contact_seeking_heights = self.contact_seeking_policy.sample(target, observation)
        pose_shift_heights = self.pose_shift_control_policy.sample(target, observation)
        average_heights = (contact_seeking_heights + pose_shift_heights) / 2
        pin_heights = self._choose_height_to_use_per_pin(observation,
                                                         contact_seeking_heights,
                                                         pose_shift_heights,
                                                         average_heights)
        self.contact_seeking_policy.sync_state(pin_heights)
        return pin_heights
    
    def _choose_height_to_use_per_pin(self,
                                      observation: np.ndarray,
                                      contact_seeking_heights: np.ndarray,
                                      pose_shift_heights: np.ndarray,
                                      average_heights: np.ndarray) -> np.ndarray:
        pin_array_observation = PinArrayEnvObservation.from_array(observation, self.pins_per_side)
        contact_mask = np.abs(pin_array_observation.pin_forces) > 0
        neighbor_only_contact_mask = binary_dilation(contact_mask).astype(bool) & ~contact_mask
        return np.where(
            contact_mask, 
            pose_shift_heights,
            np.where(
                neighbor_only_contact_mask,
                average_heights,
                contact_seeking_heights
            )
        )

