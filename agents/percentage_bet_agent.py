# Betting variant that wagers a configured fraction of the current bankroll
# while using BasicStrategyAgent for card-play decisions.
from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import legal_bet
from tournament.observation import BettingObservation


class PercentageBetAgent(BasicStrategyAgent):
    def __init__(self, percent):
        super().__init__()

        if percent <= 0 or percent > 1:
            raise ValueError("percent must be greater than 0 and at most 1")

        self.percent = percent

    def choose_bet(self, betting_observation: BettingObservation):
        return legal_bet(
            betting_observation,
            betting_observation.bankroll * self.percent,
        )

