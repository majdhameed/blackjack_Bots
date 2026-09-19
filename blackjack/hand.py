# A blackjack hand with helpers for totals, soft aces, busts, and pairs.
from blackjack.cards import Card


class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        if not isinstance(card, Card) :
            raise TypeError("Invalid card!")

        self.cards.append(card)

    def get_total(self):
        total = 0
        ace_count = 0
        for card in self.cards:
            total += card.get_value()

            if card.rank == 'A':
                ace_count += 1

        # Convert aces from 11 to 1 only until the hand no longer busts.
        while total > 21 and ace_count > 0:
            total -= 10
            ace_count -= 1

        return total

    def is_bust(self):
        return self.get_total() > 21

    def is_blackjack(self):
        if len(self.cards) == 2 and self.get_total() == 21:
            return True
        return False

    def is_soft(self):
        total = 0
        ace_count = 0

        for card in self.cards:
            total += card.get_value()
            if card.rank == 'A':
                ace_count += 1

        while total > 21 and ace_count > 0:
            total -= 10
            ace_count -= 1

        # A hand is soft when at least one ace still counts as 11.
        return ace_count > 0

    def can_split(self):
        # Face cards share a value, so any two ten-valued cards may split.
        if len(self.cards) == 2 and self.cards[0].get_value() == self.cards[1].get_value():
            return True
        return False
    
    





