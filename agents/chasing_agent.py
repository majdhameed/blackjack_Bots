from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import (
    legal_bet,
    top_two_cutoff,
)


class ChasingAgent(BasicStrategyAgent):
    """Preserve bankroll early, then increase risk when outside the top two."""

    def __init__(
        self,
        chase_after_fraction=0.5,
        maximum_bankroll_fraction=0.5,
    ):
        if not 0 <= chase_after_fraction <= 1:
            raise ValueError(
                "chase_after_fraction must be between zero and one"
            )

        if not 0 < maximum_bankroll_fraction <= 1:
            raise ValueError(
                "maximum_bankroll_fraction must be greater than zero and at most one"
            )

        self.chase_after_fraction = chase_after_fraction
        self.maximum_bankroll_fraction = (
            maximum_bankroll_fraction
        )

    def choose_bet(self, observation):
        minimum_bet = legal_bet(
            observation,
            observation.minimum_bet,
        )

        progress = (
            observation.round_number
            / observation.total_rounds
        )

        cutoff = top_two_cutoff(observation)

        if (
            progress <= self.chase_after_fraction
            or observation.bankroll >= cutoff
        ):
            return minimum_bet

        rounds_left = max(1, observation.rounds_remaining + 1)
        deficit = cutoff - observation.bankroll
        catch_up_bet = (
            observation.minimum_bet
            + deficit / rounds_left
        )

        maximum_bet = (
            observation.bankroll
            * self.maximum_bankroll_fraction
        )

        if observation.rounds_remaining == 0:
            maximum_bet = observation.bankroll

        return legal_bet(
            observation,
            min(catch_up_bet, maximum_bet),
        )
