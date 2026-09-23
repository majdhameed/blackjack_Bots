import importlib

import pytest

from ml.betting_network import BETTING_ACTION_NAMES


class FixedActionNetwork:
    def __init__(self, action_index):
        self.action_index = action_index
        self.probe_count = 0

    def preferred_action(self, features):
        assert len(features) == 57
        self.probe_count += 1
        return self.action_index


def test_strategy_diversity_distinguishes_collapse_from_variety():
    try:
        strategy_diversity = importlib.import_module(
            "ml.strategy_diversity"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Create ml/strategy_diversity.py for population "
            "behavior analysis."
        )

    measure_strategy_diversity = getattr(
        strategy_diversity,
        "measure_strategy_diversity",
        None,
    )
    assert callable(measure_strategy_diversity), (
        "ml.strategy_diversity must expose a callable "
        "measure_strategy_diversity operation"
    )

    collapsed_networks = [
        FixedActionNetwork(3)
        for _ in range(4)
    ]
    collapsed = measure_strategy_diversity(
        collapsed_networks
    )

    assert collapsed["total_decisions"] == 4 * 24
    assert collapsed["action_counts"][
        "controlled_recovery"
    ] == 4 * 24
    assert collapsed["action_percentages"][
        "controlled_recovery"
    ] == pytest.approx(1.0)
    assert collapsed["action_entropy"] == pytest.approx(0.0)
    assert collapsed["unique_policy_count"] == 1
    assert collapsed["unique_policy_rate"] == pytest.approx(0.25)
    assert all(
        network.probe_count == 24
        for network in collapsed_networks
    )

    diverse_networks = [
        FixedActionNetwork(action_index)
        for action_index in range(len(BETTING_ACTION_NAMES))
    ]
    diverse = measure_strategy_diversity(diverse_networks)

    assert diverse["total_decisions"] == 10 * 24
    assert set(diverse["action_counts"]) == set(
        BETTING_ACTION_NAMES
    )
    assert all(
        count == 24
        for count in diverse["action_counts"].values()
    )
    assert sum(
        diverse["action_percentages"].values()
    ) == pytest.approx(1.0)
    assert diverse["action_entropy"] == pytest.approx(1.0)
    assert diverse["unique_policy_count"] == 10
    assert diverse["unique_policy_rate"] == pytest.approx(1.0)
