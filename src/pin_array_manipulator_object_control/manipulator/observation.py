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