from typing import Optional

import numpy as np
import torch

from control.control_policy import ControlPolicy, EnvironmentState


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

        self.model = torch.nn.Sequential(
                    torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
                    torch.nn.ReLU(),
                    torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
                ).to(self.device)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location=self.device))

    def forward(self, input):
        hidden_1 = self.input_layer(input)
        hidden_2 = self.nonlinearity(hidden_1)
        hidden_3 = self.output_layer(hidden_2)
        output = torch.tanh(hidden_3)
        return output
    
    def to_device(self, device: torch.device):
        self.model.to(device)
        self.device = device
        return self



class GeometricRampController(torch.nn.Module):
    
    def __init__(self, array_size: float = 1.0, transition_dist: float = 0.2):
        """
        array_size: The physical width/length of the pin array (for scaling).
        transition_dist: Distance to target (normalized) where the ramp fades out.
        """
        super().__init__()
        self.array_size = array_size
        self.transition_dist = transition_dist
        
        # Tuning Parameters
        self.ramp_gain = 2.0
        self.cup_gain = 5.0
        self.cup_depth = 0.5
        self.track_width = 0.15 # Width of the lateral mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        move_vec = x[:, 0:2] / self.array_size
        p2obj = x[:, 2:4] / self.array_size
        p2tar = x[:, 4:6] / self.array_size
        dist_bt = torch.norm(move_vec, dim=1, keepdim=True) + 1e-8
        u = move_vec / dist_bt
        ramp_weight = torch.tanh(dist_bt / self.transition_dist)
        obj_to_pin = -p2obj
        d_parallel = torch.sum(obj_to_pin * u, dim=1, keepdim=True)
        d_perp = torch.norm(obj_to_pin - (d_parallel * u), dim=1, keepdim=True)
        h_ramp = torch.clamp(-d_parallel * self.ramp_gain, min=0.0)
        dist_p2tar = torch.norm(p2tar, dim=1, keepdim=True)
        h_cup = (dist_p2tar**2) * self.cup_gain - self.cup_depth
        line_mask = torch.exp(-(d_perp**2) / (2 * self.track_width**2))
        total_mask = torch.max(line_mask, torch.exp(-(dist_p2tar**2) / 0.1))
        final_h = (ramp_weight * h_ramp) + h_cup
        
        return torch.clamp(final_h * total_mask, -1.0, 1.0)
    

class SimpleTranslationControlPolicy(ControlPolicy):
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

    def parameters(self):
        return self.controller.parameters()

    def update(self, state: EnvironmentState) -> None:
        obj_pos = torch.tensor([state.object_pose.x, state.object_pose.y]).float()
        tar_pos = torch.tensor([state.target_pose.x, state.target_pose.y]).float()
        move_dir = tar_pos - obj_pos
        pin_to_obj = obj_pos - self.pin_coords
        pin_to_tar = tar_pos - self.pin_coords
        batch_move = move_dir.repeat(self.pins_per_side**2, 1)
        input_tensor = torch.cat([batch_move, pin_to_obj, pin_to_tar], dim=1)
        with torch.no_grad():
            outputs = self.controller(input_tensor)
        self.control_tensor = outputs.view(self.pins_per_side, self.pins_per_side)