from agents.basic_strategy_agent import BasicStrategyAgent
from ml.betting_encoder import encode_betting_observation
from ml.betting_network import BettingNetwork
from tournament.observation import BettingObservation


class NeuralBettingAgent(BasicStrategyAgent):
    def __init__(self, network):
        if not callable(getattr(network, "forward", None)):
            raise TypeError("network must have a forward method")

        self.network = network

    def choose_bet(self, betting_observation):
        if not isinstance(
            betting_observation,
            BettingObservation,
        ):
            raise TypeError(
                "observation must be a BettingObservation"
            )

        # Convert the observation into 46 numbers.
        features = encode_betting_observation(
            betting_observation
        )

        # The network returns a value from 0 to 1.
        bankroll_fraction = self.network.forward(features)

        bankroll = betting_observation.bankroll
        minimum_bet = betting_observation.minimum_bet

        # A player below the minimum must go all-in.
        if bankroll <= minimum_bet:
            return bankroll
        available_extra = (
            bankroll - minimum_bet
        )

        additional_bet = round(
            available_extra * bankroll_fraction
        )

        final_bet = (
            minimum_bet + additional_bet
        )

        final_bet = min(
            bankroll,
            final_bet,
        )

        return int(final_bet)