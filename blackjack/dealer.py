# Dealer behavior, including the configurable hit-on-soft-17 table rule.
from blackjack.cards import Card, Shoe
from blackjack.hand import Hand


class Dealer:
    def __init__(self, hit_soft_17):
        self.hand = Hand()
        self.hit_soft_17 = hit_soft_17

    def add_card(self, card):
        if not isinstance(card, Card):
            raise TypeError("Not a card")

        self.hand.add_card(card)

    def play(self, shoe):
        # The dealer has no strategic choices: draw to 17, optionally hitting a
        # soft 17, then stop.
        while (
            self.hand.get_total() < 17
            or (
                self.hand.get_total() == 17
                and self.hand.is_soft()
                and self.hit_soft_17
            )
        ):
            card = shoe.deal_card()
            self.hand.add_card(card)




