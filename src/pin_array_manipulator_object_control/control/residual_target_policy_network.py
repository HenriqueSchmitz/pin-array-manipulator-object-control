import torch
import torch.nn as nn
import numpy as np

FINAL_CONVOLUTION_LAYERS = 16

class ResidualTargetNetwork(nn.Module):
    def __init__(self, pins_per_side, vector_dim=18, output_dim=6, device=None):
        super(ResidualTargetNetwork, self).__init__()
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.pins_per_side = pins_per_side
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, FINAL_CONVOLUTION_LAYERS, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)
        convolved_pins_per_side = pins_per_side - 6
        cnn_out_size = FINAL_CONVOLUTION_LAYERS * convolved_pins_per_side * convolved_pins_per_side
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size + vector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(self.device)

    def forward(self, obs_dict):
        cnn_out = self.cnn(obs_dict['grid'].to(self.device))
        combined = torch.cat([cnn_out, obs_dict['vector'].to(self.device)], dim=1)
        incremental_target_residual = self.fc(combined)
        object_pose = obs_dict['vector'][:, 6:12].to(self.device)
        return object_pose + incremental_target_residual

    def get_weights(self):
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_weights(self, weights):
        start = 0
        for p in self.parameters():
            length = p.numel()
            p.data.copy_(weights[start:start+length].reshape(p.shape))
            start += length

def parse_observation(obs, pins_per_side):
    num_pins = pins_per_side ** 2
    vector_data = obs[:18]
    pin_pos = obs[18 : 18 + num_pins].reshape(pins_per_side, pins_per_side)
    pin_force = obs[18 + num_pins :].reshape(pins_per_side, pins_per_side)
    grid = np.stack([pin_pos, pin_force], axis=0)
    return {
        'vector': torch.tensor(vector_data, dtype=torch.float32).unsqueeze(0),
        'grid': torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
    }