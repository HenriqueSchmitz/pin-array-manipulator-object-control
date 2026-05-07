import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import binary_dilation

from pin_array_manipulator_object_control.control.control_policy import ControlPolicy
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig



class PoseShiftControlPolicy(ControlPolicy):
    def __init__(self, manipulator_config: PinArrayManipulatorConfig, ramp_intensity: float = 0.0):
        self.pins_per_side = manipulator_config.pins_per_side
        self.num_pins = self.pins_per_side ** 2
        self.max_height = manipulator_config.actuation_length
        self.min_height = -manipulator_config.actuation_length
        self.pin_size = self._find_pin_size(manipulator_config)
        self.pin_radius = self.pin_size / 2
        self.grid_points_fixed = self._precompute_pin_grid_coordinates(manipulator_config)
        self.expected_contact = np.zeros((self.pins_per_side, self.pins_per_side)).astype(bool)
        self.ramp_intensity = ramp_intensity

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
        desired_sphere_centers, expected_contact_heights = self._apply_pose_to_target_movement_on_pin_sphere_centers(
            pin_array_observation,
            target)
        target_heights, expected_contact = self._place_pin_spheres_in_plane_formed_by_desired_spheres(
            desired_sphere_centers,
            expected_contact_heights,
            pin_array_observation)
        target_heights += self._generate_offsets_to_create_ramp(target, pin_array_observation, expected_contact)
        self.expected_contact = expected_contact
        pin_heights =  np.clip(
            target_heights, 
            self.min_height, 
            self.max_height
        )
        return pin_heights
    
    def _apply_pose_to_target_movement_on_pin_sphere_centers(self,
                                                             pin_array_observation: PinArrayEnvObservation,
                                                             target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        relative_movement_transform = self._get_transform_from_current_pose_to_target(pin_array_observation, target)
        centers, contact_adjacent_centers = self._get_pin_sphere_centers(pin_array_observation)
        transformed_centers = (relative_movement_transform @ centers)[:3, :].T 
        transformed_contact = (relative_movement_transform @ contact_adjacent_centers)[:3, :].T
        return transformed_centers, transformed_contact
    
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
        contact_mask = np.abs(pin_array_observation.pin_forces) > 0
        contact_pin_or_neighbor_mask = contact_mask.astype(bool).flatten()
        contact_adjacent_centers = centers[:, contact_pin_or_neighbor_mask]
        return centers, contact_adjacent_centers
    
    def _place_pin_spheres_in_plane_formed_by_desired_spheres(self,
                                                              desired_sphere_centers: np.ndarray,
                                                              expected_contact_heights: np.ndarray,
                                                              pin_array_observation: PinArrayEnvObservation
                                                             ) -> tuple[np.ndarray, np.ndarray]:
        interpolator = LinearNDInterpolator(
            desired_sphere_centers[:, :2],
            desired_sphere_centers[:, 2]
        )
        new_pin_shpere_centers = interpolator(self.grid_points_fixed)
        target_heights = (new_pin_shpere_centers + self.pin_radius).reshape(self.pins_per_side, self.pins_per_side)
        invalid_heights = np.isnan(target_heights)
        valid_target_heights = np.where(invalid_heights, pin_array_observation.pin_positions, target_heights)
        expected_contact = np.zeros((self.pins_per_side, self.pins_per_side))
        if len(expected_contact_heights) < 3:
            return valid_target_heights, expected_contact.astype(bool)
        has_x_difference = False
        has_y_difference = False
        for expected_height in expected_contact_heights:
            if not has_x_difference:
                if expected_contact_heights[0][0] != expected_height[0]:
                    has_x_difference = True
            if not has_y_difference:
                if expected_contact_heights[0][1] != expected_height[1]:
                    has_x_difference = True
            if has_x_difference and has_y_difference:
                break
        if not has_x_difference or not has_y_difference:
            return valid_target_heights, expected_contact.astype(bool)
        contact_interpolator = LinearNDInterpolator(
            expected_contact_heights[:, :2],
            expected_contact_heights[:, 2]
        )
        new_contact_heights = contact_interpolator(self.grid_points_fixed)
        no_contact_pins = np.isnan(new_contact_heights).reshape(self.pins_per_side, self.pins_per_side)
        expected_contact[~no_contact_pins] = 1
        return valid_target_heights, expected_contact.astype(bool)
    
    def _generate_offsets_to_create_ramp(self,
                                         target: np.ndarray,
                                         pin_array_observation: PinArrayEnvObservation,
                                         expected_contact: np.ndarray) -> np.ndarray:
        object_pos_2d = pin_array_observation.object_pose.array()[:2]
        target_pos_2d = target[:2]
        direction = target_pos_2d - object_pos_2d
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return np.zeros((self.pins_per_side, self.pins_per_side))
        unit_dir = direction / distance
        relative_grid = self.grid_points_fixed - object_pos_2d
        projections = relative_grid @ unit_dir
        ramp_offsets = -self.ramp_intensity * projections
        gridded_offsets = ramp_offsets.reshape(self.pins_per_side, self.pins_per_side)
        no_contact_mask = (np.abs(pin_array_observation.pin_forces) > 0).reshape(self.pins_per_side, self.pins_per_side)
        gridded_offsets[no_contact_mask] = 0
        gridded_offsets[~expected_contact] = 0
        return gridded_offsets