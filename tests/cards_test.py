import pytest

from blackjack.cards import Card, Shoe


def test_number_card_value():
    card = Card(7, "Hearts")
    assert card.get_value() == 7


def test_face_card_values():
    assert Card("J", "Clubs").get_value() == 10
    assert Card("Q", "Diamonds").get_value() == 10
    assert Card("K", "Spades").get_value() == 10


def test_ace_value():
    card = Card("A", "Hearts")
    assert card.get_value() == 11


def test_card_string():
    card = Card("A", "Spades")
    assert str(card) == "A of Spades"


def test_invalid_rank():
    with pytest.raises(ValueError):
        Card(15, "Hearts")


def test_invalid_suit():
    with pytest.raises(ValueError):
        Card(10, "Stars")


def test_one_deck_has_52_cards():
    shoe = Shoe(1)
    assert shoe.cards_remaining() == 52


def test_six_decks_have_312_cards():
    shoe = Shoe(6)
    assert shoe.cards_remaining() == 312


def test_dealing_removes_one_card():
    shoe = Shoe(6)

    card = shoe.deal_card()

    assert isinstance(card, Card)
    assert shoe.cards_remaining() == 311


def test_invalid_deck_count():
    with pytest.raises(ValueError):
        Shoe(0)


def test_dealing_from_empty_shoe():
    shoe = Shoe(1)

    for _ in range(52):
        shoe.deal_card()

    with pytest.raises(ValueError):
        shoe.deal_card()