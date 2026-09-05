import pytest

from blackjack.cards import Card
from blackjack.hand import Hand


def test_new_hand_is_empty():
    hand = Hand()

    assert hand.cards == []
    assert hand.get_total() == 0


def test_add_card():
    hand = Hand()
    card = Card(7, "Hearts")

    hand.add_card(card)

    assert len(hand.cards) == 1
    assert hand.cards[0] is card


def test_add_invalid_card():
    hand = Hand()

    with pytest.raises(TypeError):
        hand.add_card("Ace of Spades")


def test_hard_total():
    hand = Hand()
    hand.add_card(Card(10, "Hearts"))
    hand.add_card(Card(7, "Clubs"))

    assert hand.get_total() == 17
    assert hand.is_soft() is False


def test_soft_total():
    hand = Hand()
    hand.add_card(Card("A", "Spades"))
    hand.add_card(Card(6, "Hearts"))

    assert hand.get_total() == 17
    assert hand.is_soft() is True


def test_ace_changes_from_eleven_to_one():
    hand = Hand()
    hand.add_card(Card("A", "Spades"))
    hand.add_card(Card(6, "Hearts"))
    hand.add_card(Card("K", "Clubs"))

    assert hand.get_total() == 17
    assert hand.is_soft() is False


def test_multiple_aces():
    hand = Hand()
    hand.add_card(Card("A", "Spades"))
    hand.add_card(Card("A", "Hearts"))
    hand.add_card(Card(9, "Clubs"))

    assert hand.get_total() == 21
    assert hand.is_soft() is True


def test_multiple_aces_become_hard():
    hand = Hand()
    hand.add_card(Card("A", "Spades"))
    hand.add_card(Card("A", "Hearts"))
    hand.add_card(Card("K", "Clubs"))

    assert hand.get_total() == 12
    assert hand.is_soft() is False


def test_bust():
    hand = Hand()
    hand.add_card(Card("K", "Spades"))
    hand.add_card(Card("Q", "Hearts"))
    hand.add_card(Card(5, "Clubs"))

    assert hand.get_total() == 25
    assert hand.is_bust() is True


def test_not_bust():
    hand = Hand()
    hand.add_card(Card("K", "Spades"))
    hand.add_card(Card("Q", "Hearts"))

    assert hand.is_bust() is False


def test_natural_blackjack():
    hand = Hand()
    hand.add_card(Card("A", "Spades"))
    hand.add_card(Card("K", "Hearts"))

    assert hand.is_blackjack() is True


def test_three_card_twenty_one_is_not_blackjack():
    hand = Hand()
    hand.add_card(Card("A", "Spades"))
    hand.add_card(Card(5, "Hearts"))
    hand.add_card(Card(5, "Clubs"))

    assert hand.get_total() == 21
    assert hand.is_blackjack() is False


def test_pair_can_split():
    hand = Hand()
    hand.add_card(Card(8, "Spades"))
    hand.add_card(Card(8, "Hearts"))

    assert hand.can_split() is True


def test_equal_value_face_cards_can_split():
    hand = Hand()
    hand.add_card(Card("K", "Spades"))
    hand.add_card(Card("Q", "Hearts"))

    assert hand.can_split() is True


def test_different_values_cannot_split():
    hand = Hand()
    hand.add_card(Card(8, "Spades"))
    hand.add_card(Card(9, "Hearts"))

    assert hand.can_split() is False


def test_three_cards_cannot_split():
    hand = Hand()
    hand.add_card(Card(8, "Spades"))
    hand.add_card(Card(8, "Hearts"))
    hand.add_card(Card(2, "Clubs"))

    assert hand.can_split() is False