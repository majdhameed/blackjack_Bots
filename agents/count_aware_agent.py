from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import legal_bet


class CountAwareAgent(BasicStrategyAgent):
    """Spread bets only when the Hi-Lo true count indicates an advantage."""

    def __init__(
        self,
        fraction_per_true_count=0.025,
        maximum_bankroll_fraction=0.2,
    ):
        if fraction_per_true_count <= 0:
            raise ValueError(
                "fraction_per_true_count must be positive"
            )

        if not 0 < maximum_bankroll_fraction <= 1:
            raise ValueError(
                "maximum_bankroll_fraction must be greater than zero and at most one"
            )

        self.fraction_per_true_count = (
            fraction_per_true_count
        )
        self.maximum_bankroll_fraction = (
            maximum_bankroll_fraction
        )

    def choose_bet(self, observation):
        if observation.true_count < 2:
            requested_bet = observation.minimum_bet
        else:
            bankroll_fraction = min(
                self.maximum_bankroll_fraction,
                observation.true_count
                * self.fraction_per_true_count,
            )

            requested_bet = (
                observation.bankroll
                * bankroll_fraction
            )

        return legal_bet(
            observation,
            requested_bet,
        )
