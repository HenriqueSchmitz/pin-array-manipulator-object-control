import torch
import numpy as np
from control.control_policy import ControlPolicy, EnvironmentState

class SphericalCradlePolicy(ControlPolicy):
    def __init__(self, manipulator, ball_diameter=0.1):
        super().__init__()
        self.manipulator = manipulator
        self.pins_per_side = manipulator.pins_per_side
        self.R = ball_diameter
        
        # 1. Define vertical ball positioning
        # Bottom of ball at -0.5 actuation percentage.
        self.z_min = - self.manipulator.actuation_length * 0.5
        # Center of ball = bottom + radius.
        self.z_center = self.z_min + self.R 
        
        # 2. Account for square pins
        # We add half the pin size to the calculation radius to ensure that 
        # pin centers correctly capture the 'edge' of the ball.
        self.pin_size = manipulator.pin_size
        self.R_eff = self.R + (self.pin_size * 0.5)

        # Precompute pin grid coordinates for efficiency
        indices = np.arange(self.pins_per_side)
        grid_x, grid_y = np.meshgrid(indices, indices, indexing='ij')
        coords = []
        for i, j in zip(grid_x.flatten(), grid_y.flatten()):
            pose = manipulator.get_pin_pose(i, j)
            coords.append([pose.x, pose.y])
        self.pin_coords = torch.tensor(coords).float()
        self.threshold = 1.5 * self.R

    def update(self, state: EnvironmentState) -> None:
        # Get current ball (x, y)
        obj_pos = torch.tensor([state.object_pose.x, state.object_pose.y]).float()
        
        # Calculate distance from each pin center to the ball center
        dist = torch.norm(self.pin_coords - obj_pos, dim=1)
        
        # 3. Spherical curve calculation
        # Using Pythagorean theorem (h = z_center - sqrt(R^2 - d^2)) 
        # which is equivalent to the sine-based vertical component R * cos(arcsin(d/R)).
        clamped_dist = torch.clamp(dist, 0, self.R)
        sphere_h = self.z_center - torch.sqrt(self.R**2 - clamped_dist**2)
        
        # 4. Map the heights to the array
        # Default state for the rest of the board is 1.0.
        h = torch.full_like(dist, 1.0)
        
        # Apply spherical heights to pins 'underneath' the ball.
        mask_under = dist < self.R_eff
        h[mask_under] = sphere_h[mask_under]
        
        # 5. Smooth transition back to 1.0 (the 'rim')
        # This blends from the ball's side height (z_center) back to the 1.0 baseline.
        rim_width = self.R * 0.4
        mask_rim = (dist >= self.R_eff) & (dist < self.R_eff + rim_width)
        
        if mask_rim.any():
            t = (dist[mask_rim] - self.R_eff) / rim_width
            h[mask_rim] = self.z_center + t * (1.0 - self.z_center)

        # Final reshape and clamp to valid range [-1, 1]
        self.control_tensor = torch.clamp(h, -1.0, 1.0).view(self.pins_per_side, self.pins_per_side)


        # distances = torch.norm(self.pin_coords - obj_pos, dim=1)
        
        # # 3. Apply binary logic: -1 if distance <= 1.5R, else 1
        # # We start with a tensor of 1.0s
        # h = torch.ones_like(distances)
        
        # # Set pins within the 1.5R threshold to -1.0
        # h[distances <= self.threshold] = -1.0
        
        # # 4. Reshape for the manipulator
        # self.control_tensor = h.view(self.pins_per_side, self.pins_per_side)

    def parameters(self):
        return []