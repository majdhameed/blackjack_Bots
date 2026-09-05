from blackjack.player import Player
from blackjack.round import Round

from blackjack.actions import Action
from blackjack.basic_strategy import (
    choose_action as choose_basic_strategy_action,
)



class BasicStrategyAgent:
    def choose_bet(self, player_: Player, minimum_bet):
        if player_.bankroll <= 0:
            raise ValueError("No money")
        if player_.bankroll >= minimum_bet:
            return minimum_bet
        else:
            return player_.bankroll

    def choose_action(self, round_: Round, hand_index):
        player_hand = round_.player.get_hand(hand_index)

        legal_actions = round_.get_legal_actions(hand_index)

        if not legal_actions:
            raise ValueError("This hand has no legal actions")

        dealer_upcard = round_.dealer.hand.cards[0]

        action = choose_basic_strategy_action(
            hand=player_hand.hand,
            dealer_upcard=dealer_upcard,
            can_double=Action.DOUBLE in legal_actions,
            can_split=Action.SPLIT in legal_actions,
            can_surrender=Action.SURRENDER in legal_actions,
            hit_soft_17=round_.dealer.hit_soft_17,
        )

        if action not in legal_actions:
            raise ValueError(
                f"Basic strategy returned an illegal action: {action}"
            )

        return action


    

    
