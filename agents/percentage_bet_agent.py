from agents.basic_strategy_agent import BasicStrategyAgent
from tournament.observation import BettingObservation

class PercentageBetAgent(BasicStrategyAgent):
    def __init__(self, percent):
        super().__init__()

        if percent <= 0 or percent > 1:
            raise ValueError("percent must be greater than 0 and at most 1")

        self.percent = percent

    def choose_bet(self, betting_observation: BettingObservation):
        if betting_observation.bankroll <= betting_observation.minimum_bet:
            return betting_observation.bankroll

        percent_bet = int(round(betting_observation.bankroll * self.percent))

        if percent_bet < betting_observation.minimum_bet:
            percent_bet = betting_observation.minimum_bet

        if percent_bet > betting_observation.bankroll:
            raise ValueError("bankroll can't support bet")

        return percent_bet

