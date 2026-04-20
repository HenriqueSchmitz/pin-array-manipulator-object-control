# run_trained_policy.py
import numpy as np
from stable_baselines3 import PPO

# Import your custom environment components
from pin_array_manipulator_object_control.environment.pin_array_env import PinArrayEnv
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.multi_target_generator import MultiTargetGenerator

def main():
    # 1. Setup - Use the exact same config as training
    config = PinArrayManipulatorConfig(
        pins_per_side=10, 
        has_wall=True,
        pin_spacing=0.001
    )
    ball = Ball(diameter=0.1, starting_z=0.2)
    reward_model = Distance3DRewardModel(manipulator_config=config)
    target_gen = MultiTargetGenerator(simulation_object=ball,
                                      manipulator_config=config,
                                      targets_to_generate=5,
                                      distance_threshold=0.1)

    # 2. Initialize Env in Human Mode
    env = PinArrayEnv(
        simulation_object=ball,
        target_generator=target_gen,
        reward_model=reward_model,
        manipulator_config=config,
        render_mode="human"
    )

    # 3. Load the Trained Model
    # Replace 'ppo_pin_manipulator_final' with your actual filename if different
    try:
        model = PPO.load("ppo_pin_manipulator_final", env=env)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Could not find the trained model file. Did you finish training?")
        return

    # 4. Visualization Loop
    obs, info = env.reset()
    print("Running policy... Press Ctrl+C to stop.")
    total_reward = 0.0
    try:
        while True:
            # The model predicts the best action based on current observation
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Reset if the episode ends (though your generator likely keeps it going)
            if terminated or truncated:
                print(f"Terminated: {terminated}, Truncated: {truncated}, Reward: {total_reward}")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        env.close()
    print("Simulation complete.")

if __name__ == "__main__":
    main()