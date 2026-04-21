import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your custom classes
from pin_array_manipulator_object_control.environment.pin_array_env import PinArrayEnv
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.single_target import SingleTargetGenerator

def train():
    config = PinArrayManipulatorConfig(pins_per_side=10, has_wall=False, rounded_pins=True)
    ball = Ball(diameter=0.1, starting_z=0.2)
    reward_model = Distance3DRewardModel(manipulator_config=config)
    target_gen = SingleTargetGenerator(simulation_object=ball, manipulator_config=config, distance_threshold=0.1)
    env = PinArrayEnv(
        simulation_object=ball,
        target_generator=target_gen,
        reward_model=reward_model,
        manipulator_config=config,
        render_mode=None,
        max_episode_steps=1000
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        tensorboard_log="./ppo_pin_array_logs/"
    )

    # 4. Train
    print("Starting training...")
    checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path="./models/", name_prefix="ppo_pins")
    model.learn(total_timesteps=10000000, callback=checkpoint_callback)

    # 5. Save the final model
    model.save("ppo_pin_manipulator_final")
    print("Training complete.")

if __name__ == "__main__":
    train()