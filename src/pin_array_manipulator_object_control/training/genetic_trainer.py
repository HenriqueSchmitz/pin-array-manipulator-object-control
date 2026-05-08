import random

import numpy as np
import torch
from tqdm import tqdm

from pin_array_manipulator_object_control.control.intermediate_target_policy_network import IntermediateTargetNetwork, parse_observation


class GeneticTrainer:
    def __init__(self, env, population_size=20, mutation_rate=0.05, sigma=0.1, device=None):
        self.env = env
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.sigma = sigma # Magnitude of mutation noise
        self.pins_per_side = env.composite_control_policy.pins_per_side
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Initialize population
        self.model = IntermediateTargetNetwork(self.pins_per_side, device=self.device)
        self.num_params = self.model.get_weights().shape[0]
        self.population = [torch.randn(self.num_params) for _ in range(population_size)]

    def evaluate(self, weights, base_seek, min_seek, seeds: list[int]):
        self.model.set_weights(weights)
        total_generation_reward = 0
        
        # Test the policy on multiple fixed targets for this generation
        for seed in seeds:
            obs, info = self.env.reset(seed=seed)
            done = False
            while not done:
                with torch.no_grad():
                    obs_parsed = parse_observation(obs, self.pins_per_side)
                    nn_output = self.model(obs_parsed).numpy().flatten()
                
                action = np.concatenate([[base_seek, min_seek], nn_output]) #
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_generation_reward += reward
                done = terminated or truncated
        return total_generation_reward / len(seeds)

    def evolve(self, base_seek, min_seek, generation=None):
        # 1. Selection with Progress Bar
        scores = []
        num_trials = 3
        gen_seeds = [random.randint(0, 999999) for _ in range(num_trials)]
        
        # Create a descriptive label for the progress bar
        pbar_desc = f"Gen {generation}" if generation is not None else "Evaluating Population"
        
        # Wrap the population loop with tqdm
        for i in tqdm(range(self.pop_size), desc=pbar_desc, unit="indiv"):
            score = self.evaluate(self.population[i], base_seek, min_seek, gen_seeds)
            scores.append(score)
        
        # Sort population by score (highest first)
        sorted_indices = np.argsort(scores)[::-1]
        self.population = [self.population[i] for i in sorted_indices]
        best_score = scores[sorted_indices[0]]
        
        # 2. Elitism: Keep top 2
        new_population = [self.population[0].clone(), self.population[1].clone()]
        
        # 3. Fill remaining with Crossover and Mutation
        while len(new_population) < self.pop_size:
            p1, p2 = random.sample(self.population[:max(2, self.pop_size//2)], 2)
            
            mask = torch.rand(self.num_params) > 0.5
            child = torch.where(mask, p1, p2).clone()
            
            if random.random() < self.mutation_rate:
                child += torch.randn(self.num_params) * self.sigma
                
            new_population.append(child)
            
        self.population = new_population
        return best_score