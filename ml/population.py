import math
import random

import numpy as np

from ml.betting_network import BettingNetwork

class Population:
    def __init__(self, population_size, elite_count, mutation_rate, mutation_strength, seed=None):
        if isinstance(population_size, bool) or not isinstance(
            population_size,
            int,
        ):
            raise TypeError(
                "population_size must be an integer"
            )

        if population_size <= 0:
            raise ValueError(
                "population_size must be greater than zero"
            )

        if population_size % 7 != 0:
            raise ValueError(
                "population_size must be a multiple of seven"
            )

        if isinstance(elite_count, bool) or not isinstance(
            elite_count,
            int,
        ):
            raise TypeError(
                "elite_count must be an integer"
            )

        if elite_count <= 0:
            raise ValueError(
                "elite_count must be greater than zero"
            )

        if elite_count > population_size:
            raise ValueError(
                "elite_count cannot exceed population_size"
            )

        if isinstance(mutation_rate, bool) or not isinstance(
            mutation_rate,
            (int, float),
        ):
            raise TypeError(
                "mutation_rate must be a number"
            )

        if not math.isfinite(mutation_rate):
            raise ValueError(
                "mutation_rate must be finite"
            )

        if not 0 <= mutation_rate <= 1:
            raise ValueError(
                "mutation_rate must be between zero and one"
            )

        if isinstance(
            mutation_strength,
            bool,
        ) or not isinstance(
            mutation_strength,
            (int, float),
        ):
            raise TypeError(
                "mutation_strength must be a number"
            )

        if not math.isfinite(mutation_strength):
            raise ValueError(
                "mutation_strength must be finite"
            )

        if mutation_strength < 0:
            raise ValueError(
                "mutation_strength cannot be negative"
            )

        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise TypeError(
                "seed must be an integer or None"
            )
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        self.random_generator = (
            np.random.default_rng(seed)
        )

        self.networks = []

        for _ in range(population_size):
            network_seed = int(
                self.random_generator.integers(
                    0,
                    2**32
                )
            )

            network = BettingNetwork(seed=network_seed)

            self.networks.append(network)

        self.fitness_scores = np.zeros(
            population_size,
            dtype=float
        )

        self.generation_number = 0

    def reset_fitness(self):
        self.fitness_scores = np.zeros(self.population_size,dtype=float)

    def add_fitness(self, network_index, amount):
        if not isinstance(network_index, int):
            raise TypeError("network index must be integer")

        if network_index >= self.population_size or network_index < 0:
            raise IndexError("network index outside of bounds")

        if not isinstance(amount, (float, int)):
            raise TypeError("amount must be a number")

        if not math.isfinite(amount):
            raise ValueError("amount must not be infinite or NaN")

        self.fitness_scores[network_index] += amount

    def get_ranked_indices(self):

        ranked_indices = np.argsort(self.fitness_scores)[::-1]

        return [int(index) for index in ranked_indices]

    def create_next_generation(self):
        sorted_indices = self.get_ranked_indices()

        elite_indices = sorted_indices[:self.elite_count]
        next_networks = []

        for elite_index in elite_indices:
            cloned_network = self.networks[elite_index].clone()
            next_networks.append(cloned_network)

        while len(next_networks) < self.population_size:
            parent_index = int(self.random_generator.choice(elite_indices))

            child = self.networks[parent_index].clone()

            child_seed = int(self.random_generator.integers(0, 2**32))
            child.random_generator = (
                np.random.default_rng(child_seed)
            )

            child.mutate(
                self.mutation_rate,
                self.mutation_strength,
            )

            next_networks.append(child)


        self.networks = next_networks

        self.generation_number += 1

        self.reset_fitness()

