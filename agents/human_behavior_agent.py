import random

from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import (
    active_opponent_bankrolls,
    legal_bet,
)


class HumanBehaviorAgent(BasicStrategyAgent):
    """A configurable bettor with tilt, win-pressing, copying, and impulses."""

    def __init__(
        self,
        seed=None,
        base_fraction=0.05,
        maximum_fraction=0.25,
        loss_multiplier=1.5,
        win_multiplier=1.25,
        impulse_probability=0.08,
        copy_probability=0.12,
        protect_probability=0.80,
    ):
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if not 0 < base_fraction <= maximum_fraction <= 1:
            raise ValueError(
                "fractions must satisfy 0 < base <= maximum <= 1"
            )
        if loss_multiplier < 1 or win_multiplier < 1:
            raise ValueError("multipliers must be at least one")
        for name, probability in (
            ("impulse_probability", impulse_probability),
            ("copy_probability", copy_probability),
            ("protect_probability", protect_probability),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(
                    f"{name} must be between zero and one"
                )

        self.random_generator = random.Random(seed)
        self.base_fraction = base_fraction
        self.maximum_fraction = maximum_fraction
        self.loss_multiplier = loss_multiplier
        self.win_multiplier = win_multiplier
        self.impulse_probability = impulse_probability
        self.copy_probability = copy_probability
        self.protect_probability = protect_probability

    def choose_bet(self, observation):
        opponents = active_opponent_bankrolls(
            observation
        )
        is_leading = (
            not opponents
            or observation.bankroll > max(opponents)
        )
        random_value = self.random_generator.random()

        if (
            is_leading
            and random_value < self.protect_probability
        ):
            requested_bet = observation.minimum_bet
        else:
            requested_bet = (
                observation.bankroll * self.base_fraction
            )
            if observation.has_previous_round:
                if observation.previous_result < 0:
                    requested_bet = max(
                        requested_bet,
                        observation.previous_bet
                        * self.loss_multiplier,
                    )
                elif observation.previous_result > 0:
                    requested_bet = max(
                        requested_bet,
                        observation.previous_bet
                        * self.win_multiplier,
                    )

            if (
                max(observation.current_bets, default=0) > 0
                and self.random_generator.random()
                < self.copy_probability
            ):
                requested_bet = max(
                    requested_bet,
                    max(observation.current_bets),
                )

            if (
                self.random_generator.random()
                < self.impulse_probability
            ):
                requested_bet = observation.bankroll * (
                    self.random_generator.uniform(
                        self.base_fraction,
                        self.maximum_fraction,
                    )
                )

            requested_bet *= self.random_generator.uniform(
                0.85,
                1.15,
            )
            requested_bet = min(
                requested_bet,
                observation.bankroll
                * self.maximum_fraction,
            )

        return legal_bet(observation, requested_bet)
