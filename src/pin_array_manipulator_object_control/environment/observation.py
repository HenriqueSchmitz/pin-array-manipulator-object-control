import numpy as np

from objects.object import Pose, Velocity



class PinArrayEnvObservation():
    def __init__(self, object_pose: Pose, object_velocity: Velocity, pin_positions: np.ndarray, pin_forces: np.ndarray):
        self.object_pose = object_pose
        self.object_velocity = object_velocity
        self.pin_positions = pin_positions
        self.pin_forces = pin_forces

    def array(self):
        return np.array([
            self.object_pose.array(),
            self.object_velocity.array(),
            self.pin_positions,
            self.pin_forces
        ])