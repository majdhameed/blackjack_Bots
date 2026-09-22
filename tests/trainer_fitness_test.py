import numpy as np
import pytest

from ml.population import Population
from ml.trainer import Trainer


def make_trainer():
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

    return trainer


def test_awards_points_by_finishing_position():
    trainer = make_trainer()

    network_indices = [
        7,
        2,
        10,
        4,
        13,
        0,
        8,
    ]

    # Each tuple is:
    # (local tournament seat, ending bankroll)
    rankings = [
        (3, 16_000),
        (0, 15_000),
        (5, 14_000),
        (1, 13_000),
        (6, 12_000),
        (2, 11_000),
        (4, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    expected_scores = np.zeros(14)

    expected_scores[4] = 6
    expected_scores[7] = 5
    expected_scores[0] = 4
    expected_scores[2] = 3
    expected_scores[8] = 2
    expected_scores[10] = 1
    expected_scores[13] = 0

    np.testing.assert_array_equal(
        trainer.population.fitness_scores,
        expected_scores,
    )


def test_two_way_first_place_tie_shares_points():
    trainer = make_trainer()

    network_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    rankings = [
        (2, 15_000),
        (5, 15_000),
        (0, 14_000),
        (1, 13_000),
        (3, 12_000),
        (4, 11_000),
        (6, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    assert (
        trainer.population.fitness_scores[2]
        == pytest.approx(5.5)
    )

    assert (
        trainer.population.fitness_scores[5]
        == pytest.approx(5.5)
    )

    assert trainer.population.fitness_scores[0] == 4
    assert trainer.population.fitness_scores[1] == 3
    assert trainer.population.fitness_scores[3] == 2
    assert trainer.population.fitness_scores[4] == 1
    assert trainer.population.fitness_scores[6] == 0


def test_three_way_middle_tie_shares_points():
    trainer = make_trainer()

    network_indices = list(range(7))

    rankings = [
        (0, 17_000),
        (1, 16_000),
        (2, 14_000),
        (3, 14_000),
        (4, 14_000),
        (5, 12_000),
        (6, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    # The tied players occupy positions 2, 3, and 4.
    # Their available points are 4, 3, and 2.
    # Each therefore receives 3 points.
    assert trainer.population.fitness_scores[2] == 3
    assert trainer.population.fitness_scores[3] == 3
    assert trainer.population.fitness_scores[4] == 3

    assert trainer.population.fitness_scores[0] == 6
    assert trainer.population.fitness_scores[1] == 5
    assert trainer.population.fitness_scores[5] == 1
    assert trainer.population.fitness_scores[6] == 0


def test_all_players_tied_receive_average_points():
    trainer = make_trainer()

    network_indices = list(range(7))

    rankings = [
        (0, 10_000),
        (1, 10_000),
        (2, 10_000),
        (3, 10_000),
        (4, 10_000),
        (5, 10_000),
        (6, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    # Average of 6, 5, 4, 3, 2, 1, and 0.
    expected_fitness = 3.0

    for network_index in network_indices:
        assert (
            trainer.population.fitness_scores[
                network_index
            ]
            == expected_fitness
        )


def test_fitness_accumulates_across_tournaments():
    trainer = make_trainer()

    network_indices = list(range(7))

    rankings = [
        (0, 16_000),
        (1, 15_000),
        (2, 14_000),
        (3, 13_000),
        (4, 12_000),
        (5, 11_000),
        (6, 10_000),
    ]

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    trainer.award_fitness(
        network_indices,
        rankings,
    )

    assert trainer.population.fitness_scores[0] == 12
    assert trainer.population.fitness_scores[1] == 10
    assert trainer.population.fitness_scores[2] == 8
    assert trainer.population.fitness_scores[3] == 6
    assert trainer.population.fitness_scores[4] == 4
    assert trainer.population.fitness_scores[5] == 2
    assert trainer.population.fitness_scores[6] == 0