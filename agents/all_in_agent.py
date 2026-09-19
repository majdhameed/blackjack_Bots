# Betting variant that keeps the inherited basic-strategy playing decisions
# but risks the entire available bankroll each round.
from agents.basic_strategy_agent import BasicStrategyAgent
from tournament.observation import BettingObservation


class AllInAgent(BasicStrategyAgent):
    def choose_bet(self, betting_observation: BettingObservation):
        return betting_observation.bankroll
