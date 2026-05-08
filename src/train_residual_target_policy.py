import torch

from pin_array_manipulator_object_control.control.residual_target_policy_network import ResidualTargetNetwork
from pin_array_manipulator_object_control.environment.composite_control_env import CompositeControlEnv
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.objects.cross_3d import Cross3D
from pin_array_manipulator_object_control.objects.cylinder import Cylinder
from pin_array_manipulator_object_control.objects.slab import Slab
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.robust_target_generator import RobustTargetGenerator
from pin_array_manipulator_object_control.training.residual_genetic_trainer import ResidualGeneticTrainer



def train():
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
    slab = Slab(width=0.2, length=0.2, thickness=0.05, starting_z=0.2)
    cylinder = Cylinder(radius = 0.05, length= 0.2, starting_z=0.2)
    cross_3d = Cross3D(arm_length=0.2, thickness = 0.04, starting_z=0.2)
    reward_model = Distance3DRewardModel(manipulator_config=config)
    target_generator = RobustTargetGenerator(simulation_object=cross_3d, manipulator_config=config)

    env = CompositeControlEnv(
        simulation_object=cross_3d,
        target_generator=target_generator,
        reward_model=reward_model,
        manipulator_config=config,
        smoothing=0.4,
        render_mode=None,
        max_episode_steps=1000
    )

    trainer = ResidualGeneticTrainer(env, cross_3d, population_size=30, elite_size=5)
    
    generations = 1000
    for gen in range(generations):
        best_fitness = trainer.evolve(BASE_SEEK_SPEED, MIN_SEEK_SPEED, generation_idx=gen)
        print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}")
        if gen % 10 == 0:
            best_model = trainer.population[0]
            save_data = {
                'state_dict': best_model.state_dict(),
                'object_max_dim': best_model.object_max_dim,
                'pins_per_side': best_model.pins_per_side
            }
            torch.save(save_data, f"./models/best_residual_model_gen_{gen}.pt")
    best_model = trainer.population[0]
    save_data = {
        'state_dict': best_model.state_dict(),
        'object_max_dim': best_model.object_max_dim,
        'pins_per_side': best_model.pins_per_side
    }
    torch.save(save_data, f"./models/final_residual_model.pt")

if __name__ == "__main__":
    train()