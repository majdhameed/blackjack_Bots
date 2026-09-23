import numpy as np

import ml.trainer as trainer_module
from ml.population import Population
from ml.trainer import Trainer


def test_train_generation_measures_evaluated_population_before_reproduction(
    monkeypatch,
):
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

    evaluated_networks = tuple(population.networks)
    expected_diversity = {
        "total_decisions": 14 * 24,
        "action_counts": {"minimum": 14 * 24},
        "action_percentages": {"minimum": 1.0},
        "action_entropy": 0.0,
        "unique_policy_count": 1,
        "unique_policy_rate": 1 / 14,
    }
    measured_networks = []

    def fake_evaluate_generation(tournaments_per_network):
        scores = np.arange(14, dtype=float)
        population.fitness_scores = scores.copy()
        return scores

    def fake_measure_strategy_diversity(networks):
        measured_networks.extend(networks)
        return expected_diversity

    trainer.evaluate_generation = fake_evaluate_generation
    monkeypatch.setattr(
        trainer_module,
        "measure_strategy_diversity",
        fake_measure_strategy_diversity,
        raising=False,
    )

    result = trainer.train_generation(
        tournaments_per_network=1
    )

    assert tuple(measured_networks) == evaluated_networks
    assert result["strategy_diversity"] is expected_diversity
    assert tuple(population.networks) != evaluated_networks
