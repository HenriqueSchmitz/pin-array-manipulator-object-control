import numpy as np
from scipy.ndimage import convolve

from pin_array_manipulator_object_control.control.control_policy import ControlPolicy
from pin_array_manipulator_object_control.control.pose_shift_control import PoseShiftControlPolicy
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig



class ContactSeekingPolicy(ControlPolicy):
    def __init__(self, manipulator_config: PinArrayManipulatorConfig, 
                 base_seek_speed=0.0002, 
                 min_seek_speed=0.0001):
        self.pins_per_side = manipulator_config.pins_per_side
        self.num_pins = self.pins_per_side ** 2
        self.max_height = manipulator_config.actuation_length
        self.min_height = -manipulator_config.actuation_length
        self.config = manipulator_config
        self.base_seek_speed = base_seek_speed
        self.min_seek_speed = min_seek_speed
        self.target_heights = None
        self.kernel = np.array([[0.5, 0.75, 0.5],
                                [0.75, 1.0, 0.75],
                                [0.5, 0.75, 0.5]])

    def sample(self, target: np.ndarray, observation: np.ndarray) -> np.ndarray:
        pin_array_observation = PinArrayEnvObservation.from_array(observation, self.pins_per_side)
        if self.target_heights is None:
            self.target_heights = pin_array_observation.pin_positions.copy()
        current_contact = (np.abs(pin_array_observation.pin_forces) > 0).astype(float)
        neighbor_activity = convolve(current_contact, self.kernel, mode='constant', cval=0.0)
        damping_factor = np.clip(neighbor_activity / self.kernel.sum(), 0, 1)
        for i in range(self.pins_per_side):
            for j in range(self.pins_per_side):
                if not current_contact[i, j] > 0:
                    speed = self.base_seek_speed * (1.0 - damping_factor[i, j])
                    speed = max(speed, self.min_seek_speed)
                    self.target_heights[i, j] += speed
                self.target_heights[i, j] = np.clip(
                    self.target_heights[i, j], 
                    self.min_height, 
                    self.max_height
                )
        return self.target_heights
    
    def sync_state(self, true_target_heights: np.ndarray):
        self.target_heights = true_target_heights.copy()