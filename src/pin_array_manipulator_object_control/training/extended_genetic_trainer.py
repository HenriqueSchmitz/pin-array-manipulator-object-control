import random

import numpy as np
import torch
from tqdm import tqdm

from pin_array_manipulator_object_control.control.intermediate_target_policy_network import IntermediateTargetNetwork, parse_observation

class ExtendedGeneticTrainer:
    def __init__(
        self, 
        env, 
        model_class, 
        population_size=10, 
        elite_size=3, 
        mutation_power=0.05, 
        sparse_rate=0.05, 
        aggressive_ratio=0.15, 
        crossover_rate=0.4,
        tournament_size=3,
        num_trials=3,
        device="cpu"
    ):
        self.env = env
        self.model_class = model_class
        self.pop_size = population_size
        self.elite_size = elite_size
        self.mutation_power = mutation_power
        self.sparse_rate = sparse_rate
        self.aggressive_ratio = aggressive_ratio
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.device = device
        self.num_trials = num_trials
        self.pins_per_side = env.composite_control_policy.pins_per_side
        
        # Initialize population with random models
        self.population = [
            IntermediateTargetNetwork(self.pins_per_side, device=self.device)
            for _ in range(population_size)
        ]

    def _get_weights(self, model):
        """Extracts all weights of the model into a single flat numpy array."""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

    def _set_weights(self, model, weights):
        """Sets the model weights from a flat numpy array."""
        start = 0
        for p in model.parameters():
            shape = p.data.shape
            size = np.prod(shape)
            new_data = weights[start:start + size].reshape(shape)
            p.data.copy_(torch.from_numpy(new_data))
            start += size

    def evaluate_individual(self, model, base_seek, min_seek, seeds: list[int]):
        """Runs a single episode and returns the total reward (fitness)."""
        model.eval()
        obs, info = self.env.reset()
        done = False
        total_reward = 0

        for seed in seeds:
            obs, info = self.env.reset(seed=seed)
            done = False
            while not done:
                with torch.no_grad():
                    obs_parsed = parse_observation(obs, self.pins_per_side)
                    nn_output = model(obs_parsed).numpy().flatten()
                
                action = np.concatenate([[base_seek, min_seek], nn_output]) #
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
        return total_reward / len(seeds)

    def tournament_selection(self, fitness_scores):
        """Randomly picks 'k' individuals and returns the index of the best one."""
        selection_indices = np.random.choice(len(self.population), self.tournament_size, replace=False)
        best_index = selection_indices[np.argmax([fitness_scores[i] for i in selection_indices])]
        return best_index

    def crossover(self, p1_weights, p2_weights):
        """Performs uniform crossover by swapping weight values between two parents."""
        mask = np.random.rand(*p1_weights.shape) > 0.5
        return np.where(mask, p1_weights, p2_weights)

    def evolve(self, base_seek, min_seek, generation_idx=None):
        """Performs one full generation of evaluation, selection, and reproduction."""
        # 1. Evaluation
        fitness_scores = []
        pbar_desc = f"Gen {generation_idx}" if generation_idx is not None else "Evaluating"
        gen_seeds = [random.randint(0, 999999) for _ in range(self.num_trials)]
        
        # Create a descriptive label for the progress bar
        pbar_desc = f"Gen {generation_idx}" if generation_idx is not None else "Evaluating Population"
        for model in tqdm(self.population, desc=pbar_desc, unit="indiv"):
            score = self.evaluate_individual(model, base_seek, min_seek, gen_seeds)
            fitness_scores.append(score)

        # 2. Identify Elites
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        new_population = [self.population[i] for i in elite_indices]
        
        best_score = max(fitness_scores)
        
        # 3. Fill remaining population
        total_to_fill = self.pop_size - self.elite_size
        
        while len(new_population) < self.pop_size:
            # Selection
            p1_idx = self.tournament_selection(fitness_scores)
            p2_idx = self.tournament_selection(fitness_scores)
            
            p1_weights = self._get_weights(self.population[p1_idx])
            p2_weights = self._get_weights(self.population[p2_idx])
            
            # Crossover
            if np.random.rand() < self.crossover_rate:
                child_weights = self.crossover(p1_weights, p2_weights)
            else:
                child_weights = p1_weights.copy()
            
            # Hybrid Mutation
            current_fill_count = len(new_population) - self.elite_size
            if (current_fill_count / total_to_fill) < self.aggressive_ratio:
                # Aggressive: Mutate everything
                noise = np.random.normal(0, self.mutation_power, size=child_weights.shape)
                child_weights += noise
            else:
                # Sparse: Mutate only a subset
                mask = np.random.rand(*child_weights.shape) < self.sparse_rate
                noise = np.random.normal(0, self.mutation_power, size=np.sum(mask))
                child_weights[mask] += noise
            
            # Create new model and apply weights
            child_model = IntermediateTargetNetwork(self.pins_per_side, device=self.device)
            self._set_weights(child_model, child_weights)
            new_population.append(child_model)
            
        self.population = new_population
        return best_score