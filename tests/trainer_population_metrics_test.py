import numpy as np
import pytest

from ml.population import Population
from ml.trainer import Trainer


def test_train_generation_returns_population_fitness_distribution():
    population = Population(
        population_size=14,
        elite_count=2,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=123,
    )
    trainer = Trainer(
        population=population,
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
    )
    fitness_scores = np.array(
        [
            2.0,
            5.0,
            1.0,
            9.0,
            4.0,
            7.0,
            3.0,
            12.0,
            8.0,
            6.0,
            11.0,
            10.0,
            0.0,
            13.0,
        ]
    )

    def fake_evaluate_generation(tournaments_per_network):
        population.fitness_scores = fitness_scores.copy()
        return fitness_scores.copy()

    trainer.evaluate_generation = fake_evaluate_generation

    result = trainer.train_generation(
        tournaments_per_network=1
    )

    assert result["population_best_fitness"] == 13.0
    assert result["average_fitness"] == pytest.approx(
        float(np.mean(fitness_scores))
    )
    assert result["population_worst_fitness"] == 0.0
    assert result["population_fitness_std"] == pytest.approx(
        float(np.std(fitness_scores))
    )
    assert type(result["population_best_fitness"]) is float
    assert type(result["population_worst_fitness"]) is float
    assert type(result["population_fitness_std"]) is float
