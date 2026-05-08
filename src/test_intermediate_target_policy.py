import torch
import numpy as np
import os

from pin_array_manipulator_object_control.control.intermediate_target_policy_network import IntermediateTargetNetwork, parse_observation
from pin_array_manipulator_object_control.environment.composite_control_env import CompositeControlEnv
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.robust_target_generator import RobustTargetGenerator

def test_model(model_path):
    # 1. Configuration (Match the parameters used in train_intermediate_target_policy.py)
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
    
    # We use the Ball object as in your training script
    ball = Ball(diameter=0.1, starting_z=0.2)
    reward_model = Distance3DRewardModel(manipulator_config=config)
    target_generator = RobustTargetGenerator(simulation_object=ball, manipulator_config=config)

    # 2. Initialize Environment in Human Render Mode
    env = CompositeControlEnv(
        simulation_object=ball,
        target_generator=target_generator,
        reward_model=reward_model,
        manipulator_config=config,
        smoothing=0.4,
        render_mode="human", # Visual output enabled
        max_episode_steps=2000
    )

    # 3. Load the Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Loading the entire model object
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    # 4. Run Inference Loop
    obs, info = env.reset()
    done = False
    total_reward = 0

    try:
        while not done:
            # Prepare observation for the NN
            with torch.no_grad():
                obs_parsed = parse_observation(obs, config.pins_per_side)
                # Ensure tensors are on the correct device
                obs_parsed['grid'] = obs_parsed['grid'].to(device)
                obs_parsed['vector'] = obs_parsed['vector'].to(device)
                
                # Get prediction (intermediate target pose)
                nn_output = model(obs_parsed).cpu().numpy().flatten()
            
            # Update debug visuals to see where the NN is trying to move the object
            env.update_debug_visuals(nn_output)
            
            # Combine speeds with NN output to form the full action
            action = np.concatenate([[BASE_SEEK_SPEED, MIN_SEEK_SPEED], nn_output])
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode Finished. Total Reward: {total_reward:.4f}")

    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    # Change this path to the specific model you want to test
    MODEL_FILE = "./models/final_model.pt"
    # MODEL_FILE = "./models/best_model_gen_3.pt"
    test_model(MODEL_FILE)