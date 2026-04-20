from typing import Optional

from mujoco import MjData # type: ignore
from torch import Tensor
import numpy as np

from pin_array_manipulator_object_control.manipulator.manipulator import Manipulator
from pin_array_manipulator_object_control.objects.object import Size3D



class PinArrayManipulatorConfig():
    def __init__(self,
                 manipulator_size: float = 1.0,
                 pins_per_side: int = 10,
                 pin_height: float = 0.1,
                 actuation_length: float = 0.1,
                 pin_spacing: float = 0.0,
                 has_wall: bool = False):
        self.manipulator_size = manipulator_size
        self.pins_per_side = pins_per_side
        self.pin_height = pin_height
        self.actuation_length = actuation_length
        self.pin_spacing = pin_spacing
        self.has_wall = has_wall
        

class PinArrayManipulator(Manipulator):
    def __init__(self,
                 name="pin_array_manipulator",
                 config: Optional[PinArrayManipulatorConfig] = None):
        super().__init__(name)
        if config is None:
            config = PinArrayManipulatorConfig()
        self.manipulator_size = config.manipulator_size
        self.pins_per_side = config.pins_per_side
        self.pin_height = config.pin_height
        self.actuation_length = config.actuation_length
        self.pin_spacing = config.pin_spacing
        self.has_wall = config.has_wall

        self.spaces_per_side = config.pins_per_side - 1
        manipulator_size_no_spaces = config.manipulator_size - config.pin_spacing * self.spaces_per_side
        self.pin_size = manipulator_size_no_spaces / self.pins_per_side
        self.pin_size_spaced = self.pin_size + config.pin_spacing
        self.actuator_indices = []
        index = 0
        for i in range(config.pins_per_side):
            self.actuator_indices.append([])
            for j in range(config.pins_per_side):
                self.actuator_indices[i].append(index)
                index += 1
        self.data = None
    
    def generate_bodies(self):
        pins_xml = ""
        for i in range(self.pins_per_side):
            for j in range(self.pins_per_side):
                name = f"pin_{i}_{j}"
                x = (i - (self.pins_per_side - 1)/2) * self.pin_size_spaced
                y = (j - (self.pins_per_side - 1)/2) * self.pin_size_spaced
                pins_xml += f"""
                <body name="{name}" pos="{x} {y} 0">
                    <joint name="{name}_joint" type="slide" axis="0 0 1" range="-{self.actuation_length} {self.actuation_length}" damping="10"/>
                    <geom type="box" pos="0 0 {-self.pin_height/2}" size="{self.pin_size/2} {self.pin_size/2} {self.pin_height/2}" rgba="0.8 0.8 0.8 1" contype="2" conaffinity="1"/>
                </body>"""
        if self.has_wall:
            wall_thickness = 0.02
            wall_height = self.pin_height * 5
            half_size = self.manipulator_size / 2
            min_pos = -half_size -  wall_thickness
            max_pos = half_size + wall_thickness
            rgba = "0.5 0.5 0.8 0.1"

            walls_xml = f"""
            <body name="walls" pos="0 0 {wall_height/2 - self.pin_height}">
                <geom name="wall_n" type="box" pos="0 {max_pos} 0" size="{half_size} {wall_thickness} {wall_height/2}" rgba="{rgba}"/>
                <geom name="wall_s" type="box" pos="0 {min_pos} 0" size="{half_size} {wall_thickness} {wall_height/2}" rgba="{rgba}"/>
                <geom name="wall_e" type="box" pos="{max_pos} 0 0" size="{wall_thickness} {half_size} {wall_height/2}" rgba="{rgba}"/>
                <geom name="wall_w" type="box" pos="{min_pos} 0 0" size="{wall_thickness} {half_size} {wall_height/2}" rgba="{rgba}"/>
            </body>"""
            pins_xml += walls_xml
        return pins_xml
    
    def generate_visual_body(self, name: str) -> str:
        return ""
    
    def generate_actuators(self):
        ctrl_min = -self.actuation_length
        ctrl_max = self.actuation_length
        actuators_xml = ""
        for i in range(self.pins_per_side):
            for j in range(self.pins_per_side):
                name = f"pin_{i}_{j}"
                actuators_xml += f'<position name="{name}_act" joint="{name}_joint" ctrlrange="{ctrl_min} {ctrl_max}" kp="2000"/>'
        return actuators_xml
    
    def actuate_from_matrix(self, matrix: np.matrix):
        if matrix.shape != (self.pins_per_side, self.pins_per_side):
            raise Exception("Matrix shape does not match manipulator size")
        if not self.data:
            raise Exception("Data not set")
        flattened = matrix.flatten()
        scaled = flattened * self.actuation_length
        self.data.ctrl[:] = scaled

    def actuate_from_tensor_percentage(self, tensor: Tensor):
        if tensor.shape != (self.pins_per_side, self.pins_per_side):
            raise Exception("Tensor shape does not match manipulator size")
        for i in range(self.pins_per_side):
            for j in range(self.pins_per_side):
                self.actuate_pin_percentage(i, j, tensor[i, j])
    
    def actuate_pin_absolute(self, i, j, height):
        if not self.data:
            raise Exception("Data not set")
        pin_index = self.get_pin_index(i, j)
        self.data.ctrl[pin_index] = height
    
    def actuate_pin_percentage(self, i, j, percentage):
        height = percentage * self.actuation_length
        self.actuate_pin_absolute(i, j, height)
    
    def get_pin_index(self, i, j):
        return self.actuator_indices[i][j]
    
    def get_size(self) -> Size3D:
        return Size3D(self.manipulator_size/2, self.manipulator_size/2, self.pin_height)