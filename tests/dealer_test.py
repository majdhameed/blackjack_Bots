import pytest

from blackjack.cards import Card
from blackjack.dealer import Dealer


class FakeShoe:
    """A predictable shoe used only for testing."""

    def __init__(self, cards):
        self.cards = cards

    def deal_card(self):
        if len(self.cards) == 0:
            raise ValueError("There are no cards to deal!")

        return self.cards.pop(0)


def test_dealer_adds_card():
    dealer = Dealer(False)
    card = Card(10, "Hearts")

    dealer.add_card(card)

    assert len(dealer.hand.cards) == 1
    assert dealer.hand.cards[0] is card


def test_dealer_rejects_invalid_card():
    dealer = Dealer(False)

    with pytest.raises(TypeError):
        dealer.add_card("King of Hearts")


def test_dealer_hits_below_17():
    dealer = Dealer(False)
    dealer.add_card(Card(10, "Hearts"))
    dealer.add_card(Card(6, "Spades"))

    shoe = FakeShoe([
        Card(4, "Clubs"),
    ])

    dealer.play(shoe)

    assert dealer.hand.get_total() == 20
    assert len(dealer.hand.cards) == 3
    assert len(shoe.cards) == 0


def test_dealer_continues_hitting_below_17():
    dealer = Dealer(False)
    dealer.add_card(Card(5, "Hearts"))
    dealer.add_card(Card(6, "Spades"))

    shoe = FakeShoe([
        Card(2, "Clubs"),
        Card(7, "Diamonds"),
    ])

    dealer.play(shoe)

    assert dealer.hand.get_total() == 20
    assert len(dealer.hand.cards) == 4
    assert len(shoe.cards) == 0


def test_dealer_stands_on_hard_17():
    dealer = Dealer(False)
    dealer.add_card(Card(10, "Hearts"))
    dealer.add_card(Card(7, "Spades"))

    shoe = FakeShoe([
        Card(4, "Clubs"),
    ])

    dealer.play(shoe)

    assert dealer.hand.get_total() == 17
    assert len(dealer.hand.cards) == 2
    assert len(shoe.cards) == 1


def test_dealer_stands_above_17():
    dealer = Dealer(False)
    dealer.add_card(Card(10, "Hearts"))
    dealer.add_card(Card(8, "Spades"))

    shoe = FakeShoe([
        Card(4, "Clubs"),
    ])

    dealer.play(shoe)

    assert dealer.hand.get_total() == 18
    assert len(dealer.hand.cards) == 2
    assert len(shoe.cards) == 1


def test_dealer_stands_on_soft_17_when_rule_is_disabled():
    dealer = Dealer(False)
    dealer.add_card(Card("A", "Hearts"))
    dealer.add_card(Card(6, "Spades"))

    shoe = FakeShoe([
        Card(2, "Clubs"),
    ])

    dealer.play(shoe)

    assert dealer.hand.get_total() == 17
    assert dealer.hand.is_soft() is True
    assert len(dealer.hand.cards) == 2
    assert len(shoe.cards) == 1


def test_dealer_hits_soft_17_when_rule_is_enabled():
    dealer = Dealer(True)
    dealer.add_card(Card("A", "Hearts"))
    dealer.add_card(Card(6, "Spades"))

    shoe = FakeShoe([
        Card(2, "Clubs"),
    ])

    dealer.play(shoe)

    assert dealer.hand.get_total() == 19
    assert dealer.hand.is_soft() is True
    assert len(dealer.hand.cards) == 3
    assert len(shoe.cards) == 0


def test_dealer_stops_after_busting():
    dealer = Dealer(False)
    dealer.add_card(Card(10, "Hearts"))
    dealer.add_card(Card(6, "Spades"))

    shoe = FakeShoe([
        Card("K", "Clubs"),
        Card(2, "Diamonds"),
    ])

    dealer.play(shoe)

    assert dealer.hand.get_total() == 26
    assert dealer.hand.is_bust() is True
    assert len(dealer.hand.cards) == 3

    # The dealer should stop after busting, leaving the next card untouched.
    assert len(shoe.cards) == 1