import importlib
from pathlib import Path

import pytest


def test_plot_training_metrics_main_generates_requested_run(
    tmp_path,
    monkeypatch,
    capsys,
):
    try:
        plot_training_metrics = importlib.import_module(
            "plot_training_metrics"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Create plot_training_metrics.py as the plotting CLI."
        )

    metrics_path = tmp_path / "run" / "metrics.csv"
    expected_paths = {
        "fitness": metrics_path.parent / "fitness.png",
        "evaluation": metrics_path.parent / "evaluation.png",
        "diversity": metrics_path.parent / "diversity.png",
        "actions": metrics_path.parent / "actions.png",
    }
    received_paths = []

    def fake_generate_training_plots(received_path):
        received_paths.append(Path(received_path))
        return expected_paths

    monkeypatch.setattr(
        plot_training_metrics,
        "generate_training_plots",
        fake_generate_training_plots,
    )

    returned_paths = plot_training_metrics.main(
        [str(metrics_path)]
    )

    assert received_paths == [metrics_path]
    assert returned_paths == expected_paths

    output = capsys.readouterr().out
    for plot_name, output_path in expected_paths.items():
        assert plot_name in output
        assert str(output_path) in output
