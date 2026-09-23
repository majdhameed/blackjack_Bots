import importlib

import pytest

from tournament.observation import BettingObservation


def test_shared_betting_probes_cover_inspection_states():
    try:
        betting_probes = importlib.import_module(
            "ml.betting_probes"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Create ml/betting_probes.py for the shared "
            "inspection scenarios."
        )

    create_betting_probes = getattr(
        betting_probes,
        "create_betting_probes",
        None,
    )
    assert callable(create_betting_probes), (
        "ml.betting_probes must expose a callable "
        "create_betting_probes operation"
    )

    probes = create_betting_probes()

    assert len(probes) == 24

    names = [probe["name"] for probe in probes]
    observations = [
        probe["observation"] for probe in probes
    ]

    assert len(set(names)) == len(names), (
        "Every betting probe must have a unique name"
    )
    assert all(
        isinstance(observation, BettingObservation)
        for observation in observations
    )

    round_numbers = {
        observation.round_number
        for observation in observations
    }
    assert 1 in round_numbers
    assert 6 in round_numbers
    assert 11 in round_numbers
    assert 12 in round_numbers

    assert any(
        observation.has_previous_round
        and observation.previous_result < 0
        for observation in observations
    ), "Include at least one previous-loss probe"

    assert any(
        "gap to second" in name.lower()
        for name in names
    ), "Include probes that vary the gap to second place"

    inspector = importlib.import_module(
        "inspect_betting_strategy"
    )
    assert getattr(
        inspector,
        "create_betting_probes",
        None,
    ) is create_betting_probes, (
        "inspect_betting_strategy must import and use the shared "
        "create_betting_probes operation"
    )
    assert not callable(
        getattr(inspector, "create_scenarios", None)
    ), (
        "Remove the old create_scenarios operation from the inspector "
        "instead of duplicating the probes"
    )
