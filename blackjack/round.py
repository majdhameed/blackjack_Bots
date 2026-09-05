from blackjack.actions import Action
from blackjack.cards import Shoe
from blackjack.dealer import Dealer
from blackjack.player import Player


class Round:
    def __init__(
        self,
        player,
        shoe,
        minimum_bet,
        hit_soft_17=False,
        max_hands=4,
    ):
        if not isinstance(player, Player):
            raise TypeError("player must be a Player")

        if not isinstance(shoe, Shoe):
            raise TypeError("shoe must be a Shoe")

        if minimum_bet <= 0:
            raise ValueError(
                "minimum_bet must be greater than 0"
            )

        if max_hands < 1:
            raise ValueError(
                "max_hands must be at least 1"
            )

        self.player = player
        self.shoe = shoe
        self.minimum_bet = minimum_bet
        self.max_hands = max_hands
        self.dealer = Dealer(hit_soft_17)

        self.is_over = False
        self.naturals_checked = False
        self.dealer_turn_complete = False
        self.outcomes = []

    def _finish_unplayable_split_aces(self):
        for player_hand in self.player.hands:
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
                len(self.player.hands) < self.max_hands
            )

            can_afford_split = (
                self.player.bankroll >= player_hand.bet
            )

            can_resplit = (
                cards_are_aces
                and has_room_to_split
                and can_afford_split
            )

            if not can_resplit:
                player_hand.finish_split_aces()

    def deal_initial_cards(self):
        if self.is_over:
            raise ValueError("Round is already over")

        if not self.player.hands:
            raise ValueError(
                "Player must place a bet before cards are dealt"
            )

        if len(self.player.hands) != 1:
            raise ValueError(
                "Initial cards require exactly one player hand"
            )

        player_hand = self.player.get_hand(0)

        if player_hand.hand.cards:
            raise ValueError("Player already has cards")

        if self.dealer.hand.cards:
            raise ValueError("Dealer already has cards")

        self.player.add_card(
            0,
            self.shoe.deal_card(),
        )
        self.dealer.add_card(
            self.shoe.deal_card()
        )

        self.player.add_card(
            0,
            self.shoe.deal_card(),
        )
        self.dealer.add_card(
            self.shoe.deal_card()
        )

    def resolve_naturals(self):
        if self.is_over:
            raise ValueError("Round is already over")

        if self.naturals_checked:
            raise ValueError(
                "Naturals have already been checked"
            )

        if not self.player.hands:
            raise ValueError("Player has no hand")

        player_hand = self.player.get_hand(0)

        if len(player_hand.hand.cards) != 2:
            raise ValueError(
                "Player must have two cards"
            )

        if len(self.dealer.hand.cards) != 2:
            raise ValueError(
                "Dealer must have two cards"
            )

        self.naturals_checked = True

        player_blackjack = (
            player_hand.is_natural_blackjack()
        )

        dealer_blackjack = (
            self.dealer.hand.is_blackjack()
        )

        if player_blackjack and dealer_blackjack:
            self.player.push(0)
            player_hand.mark_settled("push")

            self.outcomes.append("push")
            self.dealer_turn_complete = True
            self.is_over = True

            return True

        if player_blackjack:
            self.player.win_blackjack(0)
            player_hand.mark_settled("blackjack")

            self.outcomes.append("blackjack")
            self.dealer_turn_complete = True
            self.is_over = True

            return True

        if dealer_blackjack:
            player_hand.mark_settled(
                "dealer_blackjack"
            )

            self.outcomes.append(
                "dealer_blackjack"
            )

            self.dealer_turn_complete = True
            self.is_over = True

            return True

        return False

    def get_legal_actions(self, hand_index):
        if self.is_over:
            return set()

        if not self.naturals_checked:
            return set()

        if self.dealer_turn_complete:
            return set()

        player_hand = self.player.get_hand(
            hand_index
        )

        if player_hand.is_finished():
            return set()

        cards = player_hand.hand.cards

        if player_hand.split_aces:
            legal_actions = {Action.STAND}

            cards_are_aces = (
                len(cards) == 2
                and cards[0].rank == "A"
                and cards[1].rank == "A"
            )

            can_resplit = (
                cards_are_aces
                and len(self.player.hands)
                < self.max_hands
                and self.player.bankroll
                >= player_hand.bet
            )

            if can_resplit:
                legal_actions.add(Action.SPLIT)

            return legal_actions

        legal_actions = {
            Action.HIT,
            Action.STAND,
        }

        has_two_cards = len(cards) == 2

        if (
            has_two_cards
            and self.player.bankroll > 0
        ):
            legal_actions.add(Action.DOUBLE)

        if (
            player_hand.hand.can_split()
            and len(self.player.hands)
            < self.max_hands
            and self.player.bankroll
            >= player_hand.bet
        ):
            legal_actions.add(Action.SPLIT)

        if (
            has_two_cards
            and not player_hand.came_from_split
        ):
            legal_actions.add(Action.SURRENDER)

        return legal_actions

    def player_hit(self, hand_index):
        legal_actions = self.get_legal_actions(
            hand_index
        )

        if Action.HIT not in legal_actions:
            raise ValueError(
                "Hit is not legal for this hand"
            )

        card = self.shoe.deal_card()
        self.player.add_card(hand_index, card)

        return card

    def player_stand(self, hand_index):
        legal_actions = self.get_legal_actions(
            hand_index
        )

        if Action.STAND not in legal_actions:
            raise ValueError(
                "Stand is not legal for this hand"
            )

        self.player.stand(hand_index)

    def player_double(
        self,
        hand_index,
        additional_bet,
    ):
        legal_actions = self.get_legal_actions(
            hand_index
        )

        if Action.DOUBLE not in legal_actions:
            raise ValueError(
                "Double is not legal for this hand"
            )

        self.player.double_down(
            hand_index,
            additional_bet,
        )

        card = self.shoe.deal_card()
        self.player.add_card(hand_index, card)
        self.player.finish_double(hand_index)

        return card

    def player_surrender(self, hand_index):
        legal_actions = self.get_legal_actions(
            hand_index
        )

        if Action.SURRENDER not in legal_actions:
            raise ValueError(
                "Surrender is not legal for this hand"
            )

        self.player.surrender(hand_index)

    def player_split(self, hand_index):
        legal_actions = self.get_legal_actions(
            hand_index
        )

        if Action.SPLIT not in legal_actions:
            raise ValueError(
                "Split is not legal for this hand"
            )

        self.player.split_hand(
            hand_index,
            max_hands=self.max_hands,
        )

        first_card = self.shoe.deal_card()
        self.player.add_card(
            hand_index,
            first_card,
        )

        second_card = self.shoe.deal_card()
        self.player.add_card(
            hand_index + 1,
            second_card,
        )

        self._finish_unplayable_split_aces()

        return first_card, second_card

    def play_dealer(self):
        if self.is_over:
            raise ValueError("Round is already over")

        if not self.naturals_checked:
            raise ValueError(
                "Naturals must be checked first"
            )

        if self.dealer_turn_complete:
            raise ValueError(
                "Dealer turn is already complete"
            )

        if not self.player.hands:
            raise ValueError("Player has no hands")

        for player_hand in self.player.hands:
            if not player_hand.is_finished():
                raise ValueError(
                    "A player hand is still active"
                )

        playable_hands = [
            player_hand
            for player_hand in self.player.hands
            if not player_hand.has_surrendered
            and not player_hand.hand.is_bust()
        ]

        if  len(playable_hands) == 0:
            self.dealer_turn_complete = True
            return False

        self.dealer.play(self.shoe)
        self.dealer_turn_complete = True

        return True

    def settle_hand(self, hand_index):
        if self.is_over:
            raise ValueError("Round is already over")

        if not self.dealer_turn_complete:
            raise ValueError(
                "Dealer turn is not complete"
            )

        player_hand = self.player.get_hand(
            hand_index
        )

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
            self.player.win(hand_index)
            player_hand.mark_settled("win")
            return "win"

        if player_total > dealer_total:
            self.player.win(hand_index)
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
            raise ValueError(
                "Dealer turn is not complete"
            )

        results = []

        for hand_index in range(
            len(self.player.hands)
        ):
            player_hand = self.player.get_hand(
                hand_index
            )

            if player_hand.is_settled:
                result = player_hand.outcome
            else:
                result = self.settle_hand(
                    hand_index
                )

            results.append(result)

        self.outcomes = results
        self.is_over = True

        return results