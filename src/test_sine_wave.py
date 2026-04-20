import numpy as np
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.single_target import SingleTargetGenerator
from pin_array_manipulator_object_control.environment.pin_array_env import PinArrayEnv
from pin_array_manipulator_object_control.control.sine_wave import SineWaveControlPolicy


config = PinArrayManipulatorConfig(
    manipulator_size=1.0,
    pins_per_side=15,
    pin_height=0.15,
    actuation_length=0.1,
    pin_spacing=0.001,
    has_wall=True
)
ball = Ball(diameter=0.1, starting_z=0.2)
reward_model = Distance3DRewardModel(manipulator_config=config)
target_generator = SingleTargetGenerator(simulation_object=ball, manipulator_config=config)

env = PinArrayEnv(
    simulation_object=ball,
    target_generator=target_generator,
    reward_model=reward_model,
    manipulator_config=config,
    render_mode="human"
)

policy = SineWaveControlPolicy(manipulator_config=config)

def main():
    obs, info = env.reset()
    done = False

    try:
        while not done:
            target = info["target"]
            action = policy.sample(target=target, observation=obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        env.close()
    print("Simulation complete.")

if __name__ == "__main__":
    main()