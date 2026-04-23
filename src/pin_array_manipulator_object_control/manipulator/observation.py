import numpy as np

from pin_array_manipulator_object_control.objects.object import Pose, Velocity



class PinArrayEnvObservation():
    def __init__(self,
                 target_pose: Pose,
                 object_pose: Pose,
                 object_velocity: Velocity,
                 pin_positions: np.ndarray,
                 pin_forces: np.ndarray):
        self.target_pose = target_pose
        self.object_pose = object_pose
        self.object_velocity = object_velocity
        self.pin_positions = pin_positions
        self.pin_forces = pin_forces

    def array(self):
        return np.concatenate([
            self.target_pose.array(),
            self.object_pose.array(),
            self.object_velocity.array(),
            self.pin_positions.flatten(),
            self.pin_forces.flatten()
        ]).astype(np.float32)
    
    @staticmethod
    def from_array(array: np.ndarray, pins_per_side: int) -> 'PinArrayEnvObservation':
        num_pins = pins_per_side ** 2
        target_pose = Pose.from_array(array[0:6])
        object_pose = Pose.from_array(array[6:12])
        object_velocity = Velocity.from_array(array[12:18])
        pin_positions = array[18 : 18 + num_pins].reshape(pins_per_side, pins_per_side)
        pin_forces = array[18 + num_pins : 18 + 2 * num_pins].reshape(pins_per_side, pins_per_side)
        return PinArrayEnvObservation(
            target_pose=target_pose,
            object_pose=object_pose,
            object_velocity=object_velocity,
            pin_positions=pin_positions,
            pin_forces=pin_forces
        )
