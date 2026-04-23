import numpy as np
from pin_array_manipulator_object_control.control.composite_control import CompositeControlPolicy
from pin_array_manipulator_object_control.environment.composite_control_env import CompositeControlEnv
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.objects.object import Pose
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.multi_target_generator import MultiTargetGenerator
from pin_array_manipulator_object_control.environment.pin_array_env import PinArrayEnv



def calculate_incremental_target(
    observation: np.ndarray,
    info: dict,
    config: PinArrayManipulatorConfig,
    max_step_in_pin_sizes: float
) -> Pose:
    pin_array_observation = PinArrayEnvObservation.from_array(observation, config.pins_per_side)
    final_target_pose = Pose.from_array(info["target"])
    translation = pin_array_observation.object_pose.translation_to(final_target_pose)
    distance = translation.length()
    max_step = define_movement_limit(max_step_in_pin_sizes, config)
    if distance <= max_step:
        return pin_array_observation.object_pose + translation
    clipped_translation = translation.resize(max_step)
    return pin_array_observation.object_pose + clipped_translation

def define_movement_limit(number_of_pin_sizes: float, config: PinArrayManipulatorConfig) -> float:
    spaces_per_side = config.pins_per_side - 1
    pin_spacing = config.pin_spacing
    manipulator_size_no_spaces = config.manipulator_size - pin_spacing * spaces_per_side
    pin_size = manipulator_size_no_spaces / config.pins_per_side
    max_step = pin_size * number_of_pin_sizes
    return max_step



BASE_SEEK_SPEED = 0.0005
MIN_SEEK_SPEED = 0.0001

config = PinArrayManipulatorConfig(
    manipulator_size=1.0,
    pins_per_side=15,
    pin_height=0.15,
    actuation_length=0.1,
    pin_spacing=0.001,
    has_wall=True,
    rounded_pins=True
)
ball = Ball(diameter=0.1, starting_z=0.2)
reward_model = Distance3DRewardModel(manipulator_config=config)
target_generator = MultiTargetGenerator(simulation_object=ball, manipulator_config=config)

env = CompositeControlEnv(
    simulation_object=ball,
    target_generator=target_generator,
    reward_model=reward_model,
    manipulator_config=config,
    render_mode="human"
)

def main():
    obs, info = env.reset()
    done = False

    try:
        while not done:
            incremental_target_pose = calculate_incremental_target(obs, info, config, 0.2).array()
            env.update_debug_visuals(incremental_target_pose)
            action = np.array([BASE_SEEK_SPEED, MIN_SEEK_SPEED, *incremental_target_pose])
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        env.close()
    print("Simulation complete.")

if __name__ == "__main__":
    main()