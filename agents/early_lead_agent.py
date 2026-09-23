from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import (
    active_opponent_bankrolls,
    legal_bet,
)


class EarlyLeadAgent(BasicStrategyAgent):
    """Take measured early risks, then defend an established lead."""

    def __init__(
        self,
        aggressive_round_fraction=0.5,
        bankroll_fraction=0.25,
        safe_lead_fraction=0.10,
    ):
        if not 0 < aggressive_round_fraction <= 1:
            raise ValueError(
                "aggressive_round_fraction must be greater than zero and at most one"
            )

        if not 0 < bankroll_fraction <= 1:
            raise ValueError(
                "bankroll_fraction must be greater than zero and at most one"
            )

        if not 0 <= safe_lead_fraction <= 1:
            raise ValueError(
                "safe_lead_fraction must be between zero and one"
            )

        self.aggressive_round_fraction = (
            aggressive_round_fraction
        )
        self.bankroll_fraction = bankroll_fraction
        self.safe_lead_fraction = safe_lead_fraction

    def choose_bet(self, observation):
        opponents = active_opponent_bankrolls(
            observation
        )

        highest_opponent = max(opponents, default=0)
        lead_margin = (
            observation.bankroll - highest_opponent
        )
        safe_lead = (
            not opponents
            or lead_margin
            >= highest_opponent * self.safe_lead_fraction
        )

        aggressive_rounds = max(
            1,
            round(
                observation.total_rounds
                * self.aggressive_round_fraction
            ),
        )

        if (
            observation.round_number <= aggressive_rounds
            and not safe_lead
        ):
            requested_bet = (
                observation.bankroll
                * self.bankroll_fraction
            )
        else:
            requested_bet = observation.minimum_bet

        return legal_bet(
            observation,
            requested_bet,
        )
