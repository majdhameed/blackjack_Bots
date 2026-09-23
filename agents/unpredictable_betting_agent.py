import random

from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import (
    legal_bet,
    top_two_cutoff,
)


class UnpredictableBettingAgent(BasicStrategyAgent):
    """A noisy bettor that changes tactics instead of following one policy."""

    def __init__(
        self,
        seed=None,
        minimum_bet_probability=0.45,
        maximum_bankroll_fraction=0.35,
    ):
        if (
            isinstance(seed, bool)
            or seed is not None
            and not isinstance(seed, int)
        ):
            raise TypeError("seed must be an integer or None")

        if not 0 <= minimum_bet_probability <= 1:
            raise ValueError(
                "minimum_bet_probability must be between zero and one"
            )

        if not 0 < maximum_bankroll_fraction <= 1:
            raise ValueError(
                "maximum_bankroll_fraction must be greater than zero and at most one"
            )

        self.random_generator = random.Random(seed)
        self.minimum_bet_probability = (
            minimum_bet_probability
        )
        self.maximum_bankroll_fraction = (
            maximum_bankroll_fraction
        )

    def choose_bet(self, observation):
        decision = self.random_generator.random()

        if decision < self.minimum_bet_probability:
            requested_bet = observation.minimum_bet
        elif decision < 0.7:
            # Pick a fresh risk level this round, like a player betting by feel.
            requested_bet = observation.bankroll * (
                self.random_generator.uniform(
                    0.01,
                    self.maximum_bankroll_fraction,
                )
            )
        elif decision < 0.85:
            # Occasionally react to being outside an advancement position.
            deficit = max(
                0,
                top_two_cutoff(observation)
                - observation.bankroll,
            )

            requested_bet = min(
                observation.bankroll
                * self.maximum_bankroll_fraction,
                observation.minimum_bet + deficit,
            )
        elif decision < 0.95:
            # Copy the largest wager already visible at the table.
            requested_bet = max(
                (bet for bet in observation.current_bets),
                default=observation.minimum_bet,
            )
        else:
            # Rare impulsive wager.
            requested_bet = observation.bankroll * (
                self.random_generator.uniform(
                    self.maximum_bankroll_fraction / 2,
                    self.maximum_bankroll_fraction,
                )
            )

        return legal_bet(observation, requested_bet)
