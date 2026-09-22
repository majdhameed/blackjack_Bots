import numpy as np
import pytest

from ml.population import Population
from ml.trainer import Trainer


def make_trainer(
    population_size=14,
    seed=123,
):
    population = Population(
        population_size=population_size,
        elite_count=2,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=seed,
    )

    return Trainer(
        population=population,
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
    )


def test_evaluate_generation_rejects_invalid_count():
    trainer = make_trainer()

    for invalid_count in (
        0,
        -1,
        1.5,
        True,
    ):
        expected_error = (
            TypeError
            if isinstance(invalid_count, bool)
            or not isinstance(invalid_count, int)
            else ValueError
        )

        with pytest.raises(expected_error):
            trainer.evaluate_generation(
                invalid_count
            )


def test_evaluate_generation_uses_every_network():
    trainer = make_trainer()
    evaluated_groups = []

    def fake_evaluate_group(network_indices):
        evaluated_groups.append(
            list(network_indices)
        )

        for network_index in network_indices:
            trainer.population.add_fitness(
                network_index,
                1,
            )

    trainer.evaluate_group = fake_evaluate_group

    scores = trainer.evaluate_generation(
        tournaments_per_network=3
    )

    assert len(evaluated_groups) == 6

    for network_index in range(14):
        assert scores[network_index] == 3


def test_evaluate_generation_makes_groups_of_seven():
    trainer = make_trainer()
    evaluated_groups = []

    def fake_evaluate_group(network_indices):
        evaluated_groups.append(
            list(network_indices)
        )

    trainer.evaluate_group = fake_evaluate_group

    trainer.evaluate_generation(
        tournaments_per_network=2
    )

    assert len(evaluated_groups) == 4

    for group in evaluated_groups:
        assert len(group) == 7
        assert len(set(group)) == 7


def test_evaluate_generation_resets_old_fitness():
    trainer = make_trainer()

    trainer.population.fitness_scores[:] = 100

    def fake_evaluate_group(network_indices):
        for network_index in network_indices:
            trainer.population.add_fitness(
                network_index,
                2,
            )

    trainer.evaluate_group = fake_evaluate_group

    scores = trainer.evaluate_generation(
        tournaments_per_network=1
    )

    np.testing.assert_array_equal(
        scores,
        np.full(14, 2.0),
    )


def test_returned_fitness_is_a_copy():
    trainer = make_trainer()

    def fake_evaluate_group(network_indices):
        for network_index in network_indices:
            trainer.population.add_fitness(
                network_index,
                1,
            )

    trainer.evaluate_group = fake_evaluate_group

    scores = trainer.evaluate_generation(1)

    scores[0] = 999

    assert (
        trainer.population.fitness_scores[0]
        == 1
    )


def test_train_generation_returns_summary():
    trainer = make_trainer(
        population_size=14
    )

    expected_scores = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
        ],
        dtype=float,
    )

    def fake_evaluate_generation(
        tournaments_per_network,
    ):
        trainer.population.fitness_scores = (
            expected_scores.copy()
        )

        return expected_scores.copy()

    trainer.evaluate_generation = (
        fake_evaluate_generation
    )

    result = trainer.train_generation(
        tournaments_per_network=1
    )

    assert result["generation"] == 0
    assert result["best_network_index"] == 13
    assert result["best_fitness"] == 14
    assert result["average_fitness"] == pytest.approx(
        7.5
    )

    assert result["best_network"] is not None

    assert (
        trainer.population.generation_number
        == 1
    )

    assert np.all(
        trainer.population.fitness_scores == 0
    )


def test_train_generation_best_network_is_snapshot():
    trainer = make_trainer()

    def fake_evaluate_generation(
        tournaments_per_network,
    ):
        scores = np.arange(
            14,
            dtype=float,
        )

        trainer.population.fitness_scores = (
            scores.copy()
        )

        return scores.copy()

    trainer.evaluate_generation = (
        fake_evaluate_generation
    )

    winning_network = (
        trainer.population.networks[13]
    )

    original_weights = (
        winning_network.weights1.copy()
    )

    result = trainer.train_generation(1)

    np.testing.assert_array_equal(
        result["best_network"].weights1,
        original_weights,
    )

    assert (
        result["best_network"]
        is not winning_network
    )