import random

from blackjack.actions import Action
from blackjack.cards import Shoe
from blackjack.dealer import Dealer
from blackjack.player import Player

class TableRound:
    def __init__(self, players, shoe, minimum_bet, starting_player, hit_soft_17, max_hands):

        if not isinstance(players, list):
            raise TypeError("Players must be provided as a list.") 
        if len(players) < 2:
            raise ValueError("At least two players are required to start a round.")
        for player in players:
            if not isinstance(player, Player):
                raise TypeError("Each player must be an instance of the Player class.")
        if not isinstance(shoe, Shoe):
            raise TypeError("Shoe must be an instance of the Shoe class.")
        if minimum_bet <= 0:
            raise ValueError("Minimum bet must be a positive value.")
        if max_hands <= 0:
            raise ValueError("Maximum hands must be a positive value.")
        if not isinstance(hit_soft_17, bool):
            raise TypeError("hit_soft_17 must be a boolean value.")
        if not isinstance(starting_player, int):
            raise TypeError("Starting player must be an integer index.")
        if starting_player < 0 or starting_player > len(players):
            raise ValueError("Starting player must be a valid index for one of the players in the round.")

        if starting_player == 0:
            starting_player_index = random.randint(0, len(players) - 1)
        else:
            starting_player_index = starting_player - 1


        self.players = players
        self.shoe = shoe
        self.minimum_bet = minimum_bet
        self.starting_player_index = starting_player_index
        self.hit_soft_17 = hit_soft_17
        self.max_hands = max_hands

        self.dealer = Dealer(hit_soft_17)

        betting_order = []

        for i in range(len(players)):
            player_index = (starting_player_index + i) % len(players)
            betting_order.append(player_index)

        self.betting_order = betting_order

        self.action_order = list(range(len(players)))
        self.current_action_position = 0

        self.next_bettor_index = 0

        self.naturals_checked = False
        self.dealer_turn_complete = False
        self.is_over = False

        self.outcomes = [[] for _ in range(len(players))]

    def get_player(self, player_index):
        if not isinstance(player_index, int):
            raise TypeError("Player index must be an integer.")
        if player_index < 0 or player_index >= len(self.players):
            raise IndexError("Player index out of range.")
        return self.players[player_index]
    
    def is_player_finished(self, player_index):
        player = self.get_player(player_index)

        for player_hand in player.hands:
            if (
                not player_hand.is_settled
                and not player_hand.is_finished()
            ):
                return False

        return True
    
    def get_current_player_index(self):
        if self.current_action_position >= len(
            self.action_order
        ):
            return None

        return self.action_order[
            self.current_action_position
        ]

    def advance_action_turn(self):
        while self.current_action_position < len(
            self.action_order
        ):
            player_index = self.get_current_player_index()

            if not self.is_player_finished(player_index):
                return

            self.current_action_position += 1

    def _finish_unplayable_split_aces(self, player_index):
            player = self.get_player(player_index)
            player_hands = player.hands
            for player_hand in player_hands:
                if not player_hand.split_aces:
                    continue
    
                if len(player_hand.hand.cards) != 2:
                    continue
    
                if player_hand.has_stood:
                    continue
    
                cards_are_aces = (
                    player_hand.hand.cards[0].rank == "A"
                    and player_hand.hand.cards[1].rank == "A"
                )
    
                has_room_to_split = (
                    len(player.hands) < self.max_hands
                )
    
                can_afford_split = (
                    player.bankroll >= player_hand.bet
                )
    
                can_resplit = (
                    cards_are_aces
                    and has_room_to_split
                    and can_afford_split
                )
    
                if not can_resplit:
                    player_hand.finish_split_aces()

    def place_bet(self, player_index, bet_amount):
        if self.is_over:
            raise Exception("Cannot place a bet after the round is over.")
        if self.next_bettor_index >= len(self.betting_order):
            raise Exception("All players have already placed their bets.")

        expected_player_index = self.betting_order[self.next_bettor_index]

        if player_index != expected_player_index:
            raise Exception(f"It's not player {player_index}'s turn to bet. Expected player {expected_player_index}.")

        selected_player = self.players[player_index]

        selected_player.place_bet(bet_amount, self.minimum_bet)

        self.next_bettor_index += 1

    def betting_complete(self):
        return self.next_bettor_index >= len(self.betting_order)

    def deal_initial_cards(self):
        if self.is_over:
            raise ValueError(
                "Cannot deal cards after the round is over"
            )

        if not self.betting_complete():
            raise ValueError(
                "All players must bet before cards are dealt"
            )

        if self.dealer.hand.cards:
            raise ValueError(
                "Initial cards have already been dealt"
            )

        for player in self.players:
            if len(player.hands) != 1:
                raise ValueError(
                    "Each player must have exactly one initial hand"
                )

            player_hand = player.get_hand(0)

            if player_hand.hand.cards:
                raise ValueError(
                    "Players cannot already have cards"
                )

        # First card to every player.
        for player in self.players:
            card = self.shoe.deal_card()
            player.add_card(0, card)

        # Dealer's visible upcard.
        self.dealer.add_card(
            self.shoe.deal_card()
        )

        # Second card to every player.
        for player in self.players:
            card = self.shoe.deal_card()
            player.add_card(0, card)

        # Dealer's hidden card.
        self.dealer.add_card(
            self.shoe.deal_card()
        )

        
    def resolve_naturals(self):
        if self.is_over:
            raise ValueError(
                "Cannot resolve naturals after the round is over"
            )

        if self.naturals_checked:
            raise ValueError(
                "Naturals have already been resolved"
            )

        if len(self.dealer.hand.cards) != 2:
            raise ValueError(
                "Dealer must have exactly two cards"
            )

        for player in self.players:
            if len(player.hands) != 1:
                raise ValueError(
                    "Each player must have exactly one hand"
                )

            player_hand = player.get_hand(0)

            if len(player_hand.hand.cards) != 2:
                raise ValueError(
                    "Each player must have exactly two cards"
                )

        self.naturals_checked = True

        dealer_blackjack = (
            self.dealer.hand.is_blackjack()
        )

        if dealer_blackjack:
            for player_index, player in enumerate(
                self.players
            ):
                player_hand = player.get_hand(0)

                if player_hand.is_natural_blackjack():
                    player.push(0)
                    player_hand.mark_settled("push")
                    self.outcomes[player_index].append(
                        "push"
                    )
                else:
                    player_hand.mark_settled(
                        "dealer_blackjack"
                    )
                    self.outcomes[player_index].append(
                        "dealer_blackjack"
                    )

            self.dealer_turn_complete = True
            self.is_over = True
            return True

        for player_index, player in enumerate(
            self.players
        ):
            player_hand = player.get_hand(0)

            if player_hand.is_natural_blackjack():
                player.win_blackjack(0)
                player_hand.mark_settled("blackjack")
                self.outcomes[player_index].append(
                    "blackjack"
                )

        all_hands_settled = all(
            player.get_hand(0).is_settled
            for player in self.players
        )

        if all_hands_settled:
            self.dealer_turn_complete = True
            self.is_over = True
            return True

        self.advance_action_turn()

        return False


    def get_legal_actions(
        self,
        player_index,
        hand_index,
    ):
        if self.is_over:
            return set()

        if not self.naturals_checked:
            return set()

        if self.dealer_turn_complete:
            return set()

        player = self.get_player(player_index)
        player_hand = player.get_hand(hand_index)

        if player_hand.is_settled:
            return set()

        if player_hand.is_finished():
            return set()

        current_player_index = (
            self.get_current_player_index()
        )

        if player_index != current_player_index:
            return set()

        cards = player_hand.hand.cards

        # These restrictions apply only to hands
        # created by splitting Aces.
        if player_hand.split_aces:
            legal_actions = {
                Action.STAND,
            }

            cards_are_aces = (
                len(cards) == 2
                and cards[0].rank == "A"
                and cards[1].rank == "A"
            )

            can_resplit = (
                cards_are_aces
                and len(player.hands)
                < self.max_hands
                and player.bankroll
                >= player_hand.bet
            )

            if can_resplit:
                legal_actions.add(
                    Action.SPLIT
                )

            return legal_actions

        legal_actions = {
            Action.HIT,
            Action.STAND,
        }

        has_two_cards = len(cards) == 2

        # Your current rules allow doubling for less.
        if (
            has_two_cards
            and player.bankroll > 0
        ):
            legal_actions.add(
                Action.DOUBLE
            )

        if (
            player_hand.hand.can_split()
            and len(player.hands)
            < self.max_hands
            and player.bankroll
            >= player_hand.bet
        ):
            legal_actions.add(
                Action.SPLIT
            )

        # Late surrender is unavailable after splitting.
        if (
            has_two_cards
            and not player_hand.came_from_split
        ):
            legal_actions.add(
                Action.SURRENDER
            )

        return legal_actions

    def player_hit(self, player_index, hand_index):
        legal_actions = self.get_legal_actions(
            player_index, hand_index
        )

        if Action.HIT not in legal_actions:
            raise ValueError(
                "Hit is not a legal action for this hand"
            )

        player = self.get_player(player_index)

        card = self.shoe.deal_card()
        player.add_card(hand_index, card)

        self.advance_action_turn()

        return card


    def player_stand(self, player_index, hand_index):
        legal_actions = self.get_legal_actions(
            player_index, hand_index
        )

        if Action.STAND not in legal_actions:
            raise ValueError(
                "Stand is not a legal action for this hand"
            )

        player = self.get_player(player_index)

        player.stand(hand_index)
        self.advance_action_turn()



    def player_double(self, player_index, hand_index, additional_bet):
        legal_actions = self.get_legal_actions(
            player_index, hand_index
        )

        if Action.DOUBLE not in legal_actions:
            raise ValueError(
                "Double is not a legal action for this hand"
            )

        player = self.get_player(player_index)

        player.double_down(hand_index, additional_bet)

        card = self.shoe.deal_card()
        player.add_card(hand_index, card)
        player.finish_double(hand_index)
        self.advance_action_turn()

        return card


    def player_split(self, player_index, hand_index):
        legal_actions = self.get_legal_actions(
            player_index, hand_index
        )

        if Action.SPLIT not in legal_actions:
            raise ValueError(
                "Split is not a legal action for this hand"
            )

        player = self.get_player(player_index)

        player.split_hand(hand_index, max_hands=self.max_hands)

        # Deal one card to each of the two new hands.
        first_card = self.shoe.deal_card()
        second_card = self.shoe.deal_card()
        player.add_card(hand_index, first_card)
        player.add_card(hand_index + 1, second_card)

        self._finish_unplayable_split_aces(player_index)
        self.advance_action_turn()

        return  first_card, second_card

    def player_surrender(self, player_index, hand_index):
        legal_actions = self.get_legal_actions(
            player_index, hand_index
        )

        if Action.SURRENDER not in legal_actions:
            raise ValueError(
                "Surrender is not a legal action for this hand"
            )

        player = self.get_player(player_index)

        player.surrender(hand_index)
        self.advance_action_turn()

    def has_playable_hand(self):
        for player in self.players:
            for player_hand in player.hands:
                if (
                    not player_hand.has_surrendered
                    and not player_hand.hand.is_bust()
                    and not player_hand.is_settled
                ):
                    return True

        return False

    def play_dealer(self):
        if self.is_over:
            raise ValueError("Round is already over")
        if not self.naturals_checked:
            raise ValueError("Naturals have not been checked")

        if self.get_current_player_index() is not None:
            raise ValueError("Players have not finished")

        if self.dealer_turn_complete:
            raise ValueError("Dealer has already played")

        if not self.has_playable_hand():
            self.dealer_turn_complete = True
            return False

        self.dealer.play(self.shoe)
        self.dealer_turn_complete = True
        return True

    def settle_hand(self, player_index, hand_index):
        if self.is_over:
            raise ValueError("The round is already over")
        
        if not self.dealer_turn_complete:
            raise ValueError(
                "Dealer turn is not complete"
            )

        player = self.get_player(player_index)

        player_hand = player.get_hand(hand_index)

        if player_hand.is_settled:
            raise ValueError(
                "This hand has already been settled"
            )

        if not player_hand.is_finished():
            raise ValueError(
                "This hand is still active"
            )

        if player_hand.has_surrendered:
            player_hand.mark_settled(
                "surrender"
            )
            return "surrender"

        if player_hand.hand.is_bust():
            player_hand.mark_settled("loss")
            return "loss"

        player_total = (
            player_hand.hand.get_total()
        )
        dealer_total = (
            self.dealer.hand.get_total()
        )

        if self.dealer.hand.is_bust():
            player.win(hand_index)
            player_hand.mark_settled("win")
            return "win"

        if player_total > dealer_total:
            player.win(hand_index)
            player_hand.mark_settled("win")
            return "win"

        if player_total < dealer_total:
            player_hand.mark_settled("loss")
            return "loss"

        self.player.push(hand_index)
        player_hand.mark_settled("push")

        return "push"

    def settle_round(self):
        if self.is_over:
            raise ValueError("Round is already over")

        if not self.dealer_turn_complete:
            raise ValueError("Dealer turn is not complete")

        for player_index, player in enumerate(self.players):
            for hand_index in range(len(player.hands)):
                player_hand = player.get_hand(hand_index)

                if not player_hand.is_settled:
                    self.settle_hand(
                        player_index,
                        hand_index,
                    )

        self.outcomes = [
            [
                player_hand.outcome
                for player_hand in player.hands
            ]
            for player in self.players
        ]

        self.is_over = True
        return self.outcomes



