import numpy as np
import torch

from control.control_policy import ControlPolicy, EnvironmentState



class SineWaveControlPolicy(ControlPolicy):
    def __init__(self, pins_per_side: int, wave_phase: float = 0.0, wave_speed: float = 0.003, direction: int = 1):
        self.pins_per_side = pins_per_side
        self.wave_phase = wave_phase
        self.wave_speed = wave_speed
        self.direction = direction
        self.control_tensor = torch.zeros((pins_per_side, pins_per_side))

    def parameters(self):
        return []

    def update(self, state: EnvironmentState) -> None:
        self.wave_phase += self.wave_speed * self.direction
        change_direction = False
        for i in range(self.pins_per_side):
            for j in range(self.pins_per_side):
                pin_output = np.sin(self.wave_phase + (i + j) * 0.5)
                self.control_tensor[i, j] = pin_output
                if self.has_reached_end(i, j, pin_output):
                    change_direction = True
        if change_direction:
            self.direction *= -1

    def has_reached_end(self, i, j, pin_output) -> bool:
        is_first_corner_pin = (i == self.pins_per_side - 1 and j == 0)
        is_second_corner_pin = (i == 0 and j == self.pins_per_side - 1)
        if (is_first_corner_pin and self.direction == 1) or (is_second_corner_pin and self.direction == -1):
            return pin_output >= 0.95
        return False