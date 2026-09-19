# Hi-Lo card counter used to track the distribution of cards already seen.
from blackjack.cards import Card


class CardCounter:
    def __init__(self):
        self.value_counts = {
            "Ace": 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0
        }
        self.cards_seen = 0
        self.running_count = 0

    def record_card(self, card: Card):
        if not isinstance(card, Card):
            raise TypeError("Card is not of type card")

        card_value = card.get_value()

        if card_value == 11:
            bucket = "Ace"
        elif card_value == 10:
            bucket = 10
        else:
            bucket = card_value

        self.value_counts[bucket] += 1
        self.cards_seen += 1

        # Hi-Lo treats low cards as +1 and tens/aces as -1.
        if 2 <= card_value <= 6:
            self.running_count += 1
        elif card_value in [10, 11]:
            self.running_count -= 1

    def record_cards(self, cards):
        for card in cards:
            self.record_card(card)

    def get_true_count(self, decks_remaining):
        if not isinstance(decks_remaining, (int, float)):
            raise TypeError("decks remaining must be of type float or int")

        if decks_remaining <= 0:
            raise ValueError("Decks remaining must be greater than 0")

        # Dividing by decks remaining makes counts comparable across shoe sizes.
        return self.running_count / decks_remaining

    def get_counts(self):
        return tuple(self.value_counts.values())

    def reset(self):
        for key in self.value_counts:
            self.value_counts[key] = 0
        self.cards_seen = 0
        self.running_count = 0
