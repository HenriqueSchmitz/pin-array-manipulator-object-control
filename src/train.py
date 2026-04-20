import numpy as np

from control.control_policy import EnvironmentState
from control.simple_translation import SimpleTranslationControlPolicy
from environment import SimulationEnvironment
from objects.ball import Ball
from objects.object import Pose
from manipulator.pin_array_manipulator import PinArrayManipulator

MANIPULATOR_SIZE = 1
PINS_PER_SIDE = 15
PIN_HEIGHT = 0.15
ACTUATION_LENGTH = 0.1
PIN_SPACING = 0.001

target_pose = Pose(0, 0, 0)

manipulator = PinArrayManipulator(manipulator_size=MANIPULATOR_SIZE,
                                  pins_per_side=PINS_PER_SIDE,
                                  pin_height=PIN_HEIGHT,
                                  actuation_length=ACTUATION_LENGTH,
                                  pin_spacing=PIN_SPACING)
object = Ball(diameter=0.1, starting_z=0.2)
control_policy = SimpleTranslationControlPolicy(manipulator)
environment = SimulationEnvironment(manipulator, [object])
parameters = control_policy.parameters()

def control_logic():
    state = EnvironmentState(object.get_pose(), target_pose)
    control_policy.update(state)
    control_tensor = control_policy.get_control_tensor()
    manipulator.actuate_from_tensor_percentage(control_tensor)

environment.run(control_logic)

def generate_targets(num_targets: int, manipulator: PinArrayManipulator, area_limit: float) -> list[Pose]:
    max_value = manipulator.manipulator_size * area_limit / 2
    targets = []
    for _ in range(num_targets):
        x, y = np.random.uniform(-max_value, max_value, size=2)
        targets.append(Pose(x, y, 0))
    return targets

def reset():
    environment = SimulationEnvironment(manipulator, [object])
    targets = generate_targets(4, manipulator, 0.8)
    return environment, targets

def step(environment, targets, current_target_idx, control_logic):
    environment.run(control_logic)
    ball_pose = object.get_pose()
    target = targets[current_target_idx]
    dist = ball_pose.translation_to(target).length()
    reward = -dist
    if dist < 0.06 and current_target_idx < 3:
        current_target_idx += 1
        reward = 10
        print(f"Target reached! Moving to Target {current_target_idx + 1}: {target}")
    return current_target_idx, reward

def rollout(max_negative_steps: int):
    environment, targets = reset()
    current_target_idx = 0
    current_negative_steps = 0
    total_reward = 0
    while current_target_idx < 4:
        current_target_idx, reward = step(environment, targets, current_target_idx, control_logic)
        total_reward += reward
        if reward < 0:
            current_negative_steps += 1
        else:
            current_negative_steps = 0
        if current_negative_steps > max_negative_steps:
            break
    return total_reward


    