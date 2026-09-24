import copy
import importlib
from pathlib import Path

import pytest


def test_create_fitness_plot_writes_population_fitness_png(
    tmp_path,
):
    try:
        training_plots = importlib.import_module(
            "ml.training_plots"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Create ml/training_plots.py for training graphs."
        )

    create_fitness_plot = getattr(
        training_plots,
        "create_fitness_plot",
        None,
    )
    assert callable(create_fitness_plot), (
        "ml.training_plots must expose a callable "
        "create_fitness_plot operation"
    )

    records = [
        {
            "generation": 0,
            "training_best": 20.0,
            "training_mean": 12.0,
            "training_worst": 5.0,
            "training_std": 3.0,
        },
        {
            "generation": 1,
            "training_best": 24.0,
            "training_mean": 15.0,
            "training_worst": 7.0,
            "training_std": 2.5,
        },
        {
            "generation": 2,
            "training_best": 27.0,
            "training_mean": 18.0,
            "training_worst": 9.0,
            "training_std": 2.0,
        },
    ]
    original_records = copy.deepcopy(records)
    output_path = tmp_path / "nested" / "fitness.png"

    returned_path = create_fitness_plot(
        records,
        output_path,
    )

    assert returned_path == Path(output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )
    assert records == original_records


def test_create_evaluation_plot_writes_holdout_scores_png(
    tmp_path,
):
    training_plots = importlib.import_module(
        "ml.training_plots"
    )
    create_evaluation_plot = getattr(
        training_plots,
        "create_evaluation_plot",
        None,
    )
    assert callable(create_evaluation_plot), (
        "ml.training_plots must expose a callable "
        "create_evaluation_plot operation"
    )

    records = [
        {
            "generation": 0,
            "benchmark_score": 0.31,
            "normal_score": 0.42,
            "mid_score": 0.28,
            "late_score": 0.12,
            "full_tournament_score": 0.38,
            "minimum_control_score": 0.45,
            "robust_score": 0.29,
            "verified_score": None,
            "checkpoint_saved": False,
        },
        {
            "generation": 1,
            "benchmark_score": 0.39,
            "normal_score": 0.49,
            "mid_score": 0.39,
            "late_score": 0.23,
            "full_tournament_score": 0.62,
            "minimum_control_score": 0.54,
            "robust_score": 0.50,
            "verified_score": 0.433,
            "checkpoint_saved": True,
        },
        {
            "generation": 2,
            "benchmark_score": 0.37,
            "normal_score": 0.47,
            "mid_score": 0.32,
            "late_score": 0.26,
            "full_tournament_score": 0.62,
            "minimum_control_score": 0.54,
            "robust_score": 0.50,
            "verified_score": None,
            "checkpoint_saved": False,
        },
    ]
    original_records = copy.deepcopy(records)
    output_path = tmp_path / "nested" / "evaluation.png"

    returned_path = create_evaluation_plot(
        records,
        output_path,
    )

    assert returned_path == Path(output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )
    assert records == original_records


def test_create_diversity_plot_writes_policy_diversity_png(
    tmp_path,
):
    training_plots = importlib.import_module(
        "ml.training_plots"
    )
    create_diversity_plot = getattr(
        training_plots,
        "create_diversity_plot",
        None,
    )
    assert callable(create_diversity_plot), (
        "ml.training_plots must expose a callable "
        "create_diversity_plot operation"
    )

    records = [
        {
            "generation": 0,
            "action_entropy": 0.91,
            "unique_policy_rate": 0.84,
        },
        {
            "generation": 1,
            "action_entropy": 0.78,
            "unique_policy_rate": 0.63,
        },
        {
            "generation": 2,
            "action_entropy": 0.69,
            "unique_policy_rate": 0.51,
        },
    ]
    original_records = copy.deepcopy(records)
    output_path = tmp_path / "nested" / "diversity.png"

    returned_path = create_diversity_plot(
        records,
        output_path,
    )

    assert returned_path == Path(output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )
    assert records == original_records
