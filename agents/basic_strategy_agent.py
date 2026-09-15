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

    def choose_action(self, table_round, round_player_index, hand_index):
        player = table_round.get_player(round_player_index)
        player_hand = player.get_hand(hand_index)
        legal_actions = table_round.get_legal_actions(round_player_index, hand_index)   

        if not legal_actions:
            raise ValueError("No legal actions available")

        dealer_upcard = table_round.dealer.hand.cards[0]

        action = choose_basic_strategy_action(player_hand.hand, dealer_upcard, can_double=Action.DOUBLE in legal_actions, can_split=Action.SPLIT in legal_actions, can_surrender=Action.SURRENDER in legal_actions, hit_soft_17=table_round.hit_soft_17 )    

        if action not in legal_actions:
            raise ValueError("Chosen action is not legal")

        return action

    
