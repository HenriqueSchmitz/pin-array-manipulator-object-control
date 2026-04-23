import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import LinearNDInterpolator

from pin_array_manipulator_object_control.control.control_policy import ControlPolicy
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig



class PoseShiftControlPolicy(ControlPolicy):
    def __init__(self, manipulator_config: PinArrayManipulatorConfig):
        self.pins_per_side = manipulator_config.pins_per_side
        self.num_pins = self.pins_per_side ** 2
        self.max_height = manipulator_config.actuation_length
        self.min_height = -manipulator_config.actuation_length
        self.pin_size = self._find_pin_size(manipulator_config)
        self.pin_radius = self.pin_size / 2
        self.grid_points_fixed = self._precompute_pin_grid_coordinates(manipulator_config)

    def _find_pin_size(self, manipulator_config: PinArrayManipulatorConfig) -> float:
        spaces_per_side = manipulator_config.pins_per_side - 1
        pin_spacing = manipulator_config.pin_spacing
        manipulator_size_no_spaces = manipulator_config.manipulator_size - pin_spacing * spaces_per_side
        pin_size = manipulator_size_no_spaces / manipulator_config.pins_per_side
        return pin_size

    def _precompute_pin_grid_coordinates(self, manipulator_config: PinArrayManipulatorConfig) -> np.ndarray:
        pin_spacing = manipulator_config.pin_spacing
        pin_size_spaced = self.pin_size + pin_spacing
        pin_coordinates = np.linspace(
            -(self.pins_per_side - 1) / 2 * pin_size_spaced,
            (self.pins_per_side - 1) / 2 * pin_size_spaced,
            self.pins_per_side
        )
        self.pin_x, self.pin_y =  np.meshgrid(pin_coordinates, pin_coordinates, indexing='ij')
        self.grid_points_fixed = np.stack([self.pin_x.flatten(), self.pin_y.flatten()], axis=1)
        return self.grid_points_fixed

    def sample(self, target: np.ndarray, observation: np.ndarray) -> np.ndarray:
        pin_array_observation = PinArrayEnvObservation.from_array(observation, self.pins_per_side)
        desired_sphere_centers = self._apply_pose_to_target_movement_on_pin_sphere_centers(pin_array_observation,
                                                                                           target)
        target_heights = self._place_pin_spheres_in_plane_formed_by_desired_spheres(desired_sphere_centers,
                                                                                    pin_array_observation)
        pin_heights =  np.clip(
            target_heights, 
            self.min_height, 
            self.max_height
        )
        return pin_heights
    
    def _apply_pose_to_target_movement_on_pin_sphere_centers(self,
                                                             pin_array_observation: PinArrayEnvObservation,
                                                             target: np.ndarray) -> np.ndarray:
        relative_movement_transform = self._get_transform_from_current_pose_to_target(pin_array_observation, target)
        centers = self._get_pin_sphere_centers(pin_array_observation)
        transformed_centers = (relative_movement_transform @ centers)[:3, :].T 
        return transformed_centers
    
    def _get_transform_from_current_pose_to_target(self,
                                                   pin_array_observation: PinArrayEnvObservation,
                                                   target: np.ndarray) -> np.ndarray:
        transform_matrix_current_pose = self._get_transformation_matrix_from_pose(
            pin_array_observation.object_pose.array()
        )
        transform_matrix_target_pose = self._get_transformation_matrix_from_pose(target)
        relative_movement_transform = transform_matrix_target_pose @ np.linalg.inv(transform_matrix_current_pose)
        return relative_movement_transform

    def _get_transformation_matrix_from_pose(self, pose_array: np.ndarray):
        """Convert a 6D pose into 4x4 transform matrix from origin to pose."""
        translation = pose_array[:3]
        rotation = R.from_euler('xyz', pose_array[3:], degrees=True).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return matrix
    
    def _get_pin_sphere_centers(self, pin_array_observation: PinArrayEnvObservation):
        sphere_centers = pin_array_observation.pin_positions - self.pin_radius
        centers = np.stack([
            self.pin_x.flatten(), 
            self.pin_y.flatten(), 
            sphere_centers.flatten(),
            np.ones(self.num_pins) # One enables rotation on the matrix operation
        ], axis=0)
        return centers
    
    def _place_pin_spheres_in_plane_formed_by_desired_spheres(self,
                                                              desired_sphere_centers: np.ndarray,
                                                              pin_array_observation: PinArrayEnvObservation
                                                             ) -> np.ndarray:
        interpolator = LinearNDInterpolator(
            desired_sphere_centers[:, :2],
            desired_sphere_centers[:, 2]
        )
        new_pin_shpere_centers = interpolator(self.grid_points_fixed)
        target_heights = (new_pin_shpere_centers + self.pin_radius).reshape(self.pins_per_side, self.pins_per_side)
        invalid_heights = np.isnan(target_heights)
        valid_target_heights = np.where(invalid_heights, pin_array_observation.pin_positions, target_heights)
        return valid_target_heights