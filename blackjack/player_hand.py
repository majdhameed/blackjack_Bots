# Per-hand state for one player. Splits create multiple PlayerHand instances,
# each with its own wager, cards, completion flags, and settlement outcome.
from blackjack.cards import Card
from blackjack.hand import Hand


class PlayerHand:
    def __init__(
        self,
        bet,
        came_from_split=False,
        split_aces=False,
    ):
        if bet <= 0:
            raise ValueError("Bet must be greater than 0")

        self.hand = Hand()
        self.bet = bet

        self.has_stood = False
        self.has_doubled = False
        self.has_surrendered = False

        self.came_from_split = came_from_split
        self.split_aces = split_aces

        self.is_settled = False
        self.outcome = None

    def add_card(self, card):
        if not isinstance(card, Card):
            raise TypeError("Not a card")

        if self.is_finished():
            raise ValueError("This hand is already finished")

        self.hand.add_card(card)

    def stand(self):
        if self.is_finished():
            raise ValueError("This hand is already finished")

        self.has_stood = True

    def mark_doubled(self):
        if self.is_finished():
            raise ValueError("This hand is already finished")

        if len(self.hand.cards) != 2:
            raise ValueError(
                "Player can only double with two cards"
            )

        self.has_doubled = True

    def finish_double(self):
        if not self.has_doubled:
            raise ValueError("This hand has not doubled")

        self.has_stood = True

    def mark_surrendered(self):
        if self.is_finished():
            raise ValueError("This hand is already finished")

        if len(self.hand.cards) != 2:
            raise ValueError(
                "Player can only surrender with two cards"
            )

        self.has_surrendered = True
        self.has_stood = True

    def is_finished(self):
        # Reaching 21 ends player decisions even without an explicit stand.
        return (
            self.has_stood
            or self.has_surrendered
            or self.hand.is_bust()
            or self.hand.get_total() >= 21
        )

    def finish_split_aces(self):
        if not self.split_aces:
            raise ValueError("hand is not aces")
        if len(self.hand.cards) != 2:
            raise ValueError("Hand does not have 2 cards")

        self.has_stood = True

    def is_natural_blackjack(self):
        # A two-card 21 created by a split is paid as a normal win.
        return self.hand.is_blackjack() and not self.came_from_split 


    def mark_settled(self, outcome):
        # Settlement is intentionally one-way so a hand cannot pay out twice.
        valid_outcomes = {
            "win",
            "loss",
            "push",
            "surrender",
            "blackjack",
            "dealer_blackjack",
        }

        if outcome not in valid_outcomes:
            raise ValueError("Invalid hand outcome")

        if self.is_settled:
            raise ValueError("This hand has already been settled")

        self.outcome = outcome
        self.is_settled = True
        
