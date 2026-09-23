from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import (
    active_opponent_bankrolls,
    legal_bet,
)


class ControlledLeadMartingaleAgent(BasicStrategyAgent):
    """Build a small lead with a capped 5%-10%-20% progression."""

    def __init__(
        self,
        base_fraction=0.05,
        maximum_fraction=0.20,
        loss_multiplier=2.0,
        target_lead=100,
    ):
        if not 0 < base_fraction <= 1:
            raise ValueError(
                "base_fraction must be greater than zero and at most one"
            )
        if not base_fraction <= maximum_fraction <= 1:
            raise ValueError(
                "maximum_fraction must be at least base_fraction and at most one"
            )
        if loss_multiplier < 1:
            raise ValueError(
                "loss_multiplier must be at least one"
            )
        if target_lead < 0:
            raise ValueError(
                "target_lead cannot be negative"
            )

        self.base_fraction = base_fraction
        self.maximum_fraction = maximum_fraction
        self.loss_multiplier = loss_multiplier
        self.target_lead = target_lead

    def choose_bet(self, observation):
        opponents = active_opponent_bankrolls(
            observation
        )
        has_target_lead = (
            not opponents
            or observation.bankroll
            >= max(opponents) + self.target_lead
        )

        if has_target_lead:
            requested_bet = observation.minimum_bet
        elif (
            observation.has_previous_round
            and observation.previous_result < 0
        ):
            requested_bet = min(
                observation.previous_bet
                * self.loss_multiplier,
                observation.bankroll
                * self.maximum_fraction,
            )
        else:
            requested_bet = (
                observation.bankroll
                * self.base_fraction
            )

        return legal_bet(
            observation,
            requested_bet,
        )
