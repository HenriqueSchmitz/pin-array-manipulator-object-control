from typing import Optional

import numpy as np
import torch

from control.control_policy import ControlPolicy, EnvironmentState
from manipulator.pin_array_manipulator import PinArrayManipulator
from objects.object import Translation


INPUT_DIM = 4
HIDDEN_DIM = 20
OUTPUT_DIM = 1



class SimpleTranslationController(torch.nn.Module):
    def __init__(self,
                 device: Optional[torch.device] = "cpu", # type: ignore
                 weights_path: Optional[str] = None):
        super(SimpleTranslationController, self).__init__()
        self.device = device
        self.input_layer = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM).to(self.device)
        self.nonlinearity = torch.nn.LeakyReLU().to(self.device)
        self.output_layer = torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM).to(self.device)
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location=self.device))

    def forward(self, input):
        hidden_1 = self.input_layer(input)
        hidden_2 = self.nonlinearity(hidden_1)
        hidden_3 = self.output_layer(hidden_2)
        output = torch.tanh(hidden_3)
        return output


class GeometricRampController(torch.nn.Module):
    
    def __init__(self, ramp_gain: float = 0.5, cup_gain: float = 1.5, cup_depth: float = 0.4):
        """
        ramp_gain: How much the whole floor tilts toward the target.
        cup_gain: How steep the "walls" of the valley are.
        cup_depth: How deep the center of the target 'dip' is.
        """
        super().__init__()
        self.ramp_gain = ramp_gain
        self.cup_gain = cup_gain
        self.cup_depth = cup_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x contains: [move_dir_x, move_dir_y, pin_to_obj_x, pin_to_obj_y]
        # Note: We need the pin's position relative to the TARGET for a stable cup.
        
        move_vec = x[:, 0:2] # Ball -> Target
        pin_to_ball = x[:, 2:4] # Pin -> Ball
        
        # 1. Calculate Pin's position relative to the Target
        # Pin_to_Target = (Pin -> Ball) + (Ball -> Target)
        pin_to_target = pin_to_ball + move_vec
        dist_to_target = torch.norm(move_vec, dim=1, keepdim=True) + 1e-8
        unit_move_dir = move_vec / dist_to_target

        # 2. THE RAMP: Projection of the pin along the movement axis
        # This creates a steady slope that is high behind the ball and low at target
        ramp_proj = torch.sum(pin_to_target * unit_move_dir, dim=1, keepdim=True)
        ramp_height = ramp_proj * self.ramp_gain

        # 3. THE CUP: Quadratic distance from the target
        # This creates the "valley" walls. Pins further from target go UP.
        radial_dist = torch.norm(pin_to_target, dim=1, keepdim=True)
        cup_height = (radial_dist ** 2) * self.cup_gain - self.cup_depth

        # 4. COMBINE: The sum of the tilt and the bowl
        # We use tanh to keep the values smooth and within [-1, 1]
        combined_height = torch.tanh(ramp_height + cup_height)

        return combined_height
    

class SimpleTranslationControlPolicy(ControlPolicy):
    def __init__(self, manipulator: PinArrayManipulator):
        self.manipulator = manipulator
        self.pins_per_side = self.manipulator.pins_per_side
        self.control_tensor = torch.zeros((self.pins_per_side, self.pins_per_side))
        self.controller = SimpleTranslationController()

    def update(self, state: EnvironmentState) -> None:
        move_direction = self.object_move_direction(state)
        object_pose = state.object_pose
        for i in range(self.pins_per_side):
            for j in range(self.pins_per_side):
                pin_pose = self.manipulator.get_pin_pose(i, j)
                pin_to_object = pin_pose.translation_to(object_pose)
                control_input_tensor = torch.tensor([
                    move_direction.x,
                    move_direction.y,
                    move_direction.length(),
                    pin_to_object.x,
                    pin_to_object.y,
                    pin_to_object.length()
                ])
                pin_output = self.controller(control_input_tensor).item()
                self.control_tensor[i, j] = pin_output

    def object_move_direction(self, state: EnvironmentState) -> Translation:
        object_pose = state.object_pose
        target_pose = state.target_pose
        return object_pose.translation_to(target_pose)

import torch
import numpy as np
from control.control_policy import ControlPolicy, EnvironmentState

class FastTranslationControlPolicy(ControlPolicy):
    def __init__(self, manipulator):
        super().__init__()
        self.manipulator = manipulator
        self.pins_per_side = manipulator.pins_per_side
        self.controller = GeometricRampController()
        indices = np.arange(self.pins_per_side)
        grid_x, grid_y = np.meshgrid(indices, indices, indexing='ij')
        self.pin_coords = torch.stack([
            torch.tensor([manipulator.get_pin_pose(i, j).x for i, j in zip(grid_x.flatten(), grid_y.flatten())]),
            torch.tensor([manipulator.get_pin_pose(i, j).y for i, j in zip(grid_x.flatten(), grid_y.flatten())])
        ], dim=1).float()

    def update(self, state: EnvironmentState) -> None:
        obj_pos = torch.tensor([state.object_pose.x, state.object_pose.y]).float()
        tar_pos = torch.tensor([state.target_pose.x, state.target_pose.y]).float()
        move_dir = tar_pos - obj_pos
        pin_to_obj = obj_pos - self.pin_coords
        batch_move = move_dir.repeat(self.pins_per_side**2, 1)
        input_tensor = torch.cat([batch_move, pin_to_obj], dim=1)
        with torch.no_grad():
            outputs = self.controller(input_tensor)
        self.control_tensor = outputs.view(self.pins_per_side, self.pins_per_side)