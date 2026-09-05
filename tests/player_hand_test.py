import pytest

from blackjack.cards import Card
from blackjack.player_hand import PlayerHand


def test_invalid_bet():
    with pytest.raises(ValueError):
        PlayerHand(0)


def test_new_player_hand():
    player_hand = PlayerHand(1_000)

    assert player_hand.bet == 1_000
    assert player_hand.hand.cards == []
    assert player_hand.has_stood is False
    assert player_hand.has_doubled is False
    assert player_hand.has_surrendered is False
    assert player_hand.came_from_split is False
    assert player_hand.split_aces is False


def test_add_card():
    player_hand = PlayerHand(1_000)
    card = Card(10, "Hearts")

    player_hand.add_card(card)

    assert player_hand.hand.cards[0] is card


def test_reject_invalid_card():
    player_hand = PlayerHand(1_000)

    with pytest.raises(TypeError):
        player_hand.add_card("10 of Hearts")


def test_stand_finishes_hand():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(10, "Hearts"))
    player_hand.add_card(Card(7, "Spades"))

    player_hand.stand()

    assert player_hand.has_stood is True
    assert player_hand.is_finished() is True


def test_cannot_add_card_after_standing():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(10, "Hearts"))
    player_hand.add_card(Card(7, "Spades"))
    player_hand.stand()

    with pytest.raises(ValueError):
        player_hand.add_card(Card(2, "Clubs"))


def test_bust_finishes_hand():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(10, "Hearts"))
    player_hand.add_card(Card(8, "Spades"))
    player_hand.add_card(Card(5, "Clubs"))

    assert player_hand.hand.is_bust() is True
    assert player_hand.is_finished() is True


def test_twenty_one_finishes_hand():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(10, "Hearts"))
    player_hand.add_card(Card(5, "Spades"))
    player_hand.add_card(Card(6, "Clubs"))

    assert player_hand.hand.get_total() == 21
    assert player_hand.is_finished() is True


def test_mark_doubled():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(5, "Hearts"))
    player_hand.add_card(Card(6, "Spades"))

    player_hand.mark_doubled()

    assert player_hand.has_doubled is True
    assert player_hand.has_stood is False


def test_cannot_double_with_more_than_two_cards():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(2, "Hearts"))
    player_hand.add_card(Card(3, "Spades"))
    player_hand.add_card(Card(4, "Clubs"))

    with pytest.raises(ValueError):
        player_hand.mark_doubled()


def test_finish_double():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(5, "Hearts"))
    player_hand.add_card(Card(6, "Spades"))
    player_hand.mark_doubled()

    player_hand.add_card(Card(10, "Clubs"))
    player_hand.finish_double()

    assert player_hand.has_doubled is True
    assert player_hand.has_stood is True
    assert player_hand.is_finished() is True


def test_cannot_finish_double_before_doubling():
    player_hand = PlayerHand(1_000)

    with pytest.raises(ValueError):
        player_hand.finish_double()


def test_mark_surrendered():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(10, "Hearts"))
    player_hand.add_card(Card(6, "Spades"))

    player_hand.mark_surrendered()

    assert player_hand.has_surrendered is True
    assert player_hand.has_stood is True
    assert player_hand.is_finished() is True


def test_cannot_surrender_after_hitting():
    player_hand = PlayerHand(1_000)
    player_hand.add_card(Card(5, "Hearts"))
    player_hand.add_card(Card(6, "Spades"))
    player_hand.add_card(Card(2, "Clubs"))

    with pytest.raises(ValueError):
        player_hand.mark_surrendered()


def test_split_properties():
    player_hand = PlayerHand(
        bet=1_000,
        came_from_split=True,
        split_aces=True,
    )

    assert player_hand.came_from_split is True
    assert player_hand.split_aces is True

def test_original_blackjack_is_natural():
    player_hand = PlayerHand(bet=1_000)
    player_hand.add_card(Card("A", "Hearts"))
    player_hand.add_card(Card("K", "Spades"))

    assert player_hand.hand.is_blackjack() is True
    assert player_hand.is_natural_blackjack() is True


def test_split_blackjack_is_not_natural():
    player_hand = PlayerHand(
        bet=1_000,
        came_from_split=True,
        split_aces=True,
    )

    player_hand.add_card(Card("A", "Hearts"))
    player_hand.add_card(Card("K", "Spades"))

    assert player_hand.hand.is_blackjack() is True
    assert player_hand.is_natural_blackjack() is False


def test_three_card_twenty_one_is_not_natural():
    player_hand = PlayerHand(bet=1_000)
    player_hand.add_card(Card(7, "Hearts"))
    player_hand.add_card(Card(7, "Spades"))
    player_hand.add_card(Card(7, "Clubs"))

    assert player_hand.hand.get_total() == 21
    assert player_hand.is_natural_blackjack() is False


def test_finish_split_aces():
    player_hand = PlayerHand(
        bet=1_000,
        came_from_split=True,
        split_aces=True,
    )

    player_hand.add_card(Card("A", "Hearts"))
    player_hand.add_card(Card(9, "Spades"))

    player_hand.finish_split_aces()

    assert player_hand.has_stood is True
    assert player_hand.is_finished() is True


def test_cannot_finish_non_split_aces():
    player_hand = PlayerHand(bet=1_000)
    player_hand.add_card(Card("A", "Hearts"))
    player_hand.add_card(Card(9, "Spades"))

    with pytest.raises(ValueError):
        player_hand.finish_split_aces()


def test_split_aces_must_have_two_cards_before_finishing():
    player_hand = PlayerHand(
        bet=1_000,
        came_from_split=True,
        split_aces=True,
    )

    player_hand.add_card(Card("A", "Hearts"))

    with pytest.raises(ValueError):
        player_hand.finish_split_aces()