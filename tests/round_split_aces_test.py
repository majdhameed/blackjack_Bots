import pytest

from blackjack.cards import Card, Shoe
from blackjack.player import Player
from blackjack.round import Round


def make_round_with_pair(
    first_card,
    second_card,
    max_hands=4,
):
    player = Player(10_000)
    player.place_bet(1_000, minimum_bet=100)

    shoe = Shoe(1)
    round_ = Round(
        player=player,
        shoe=shoe,
        minimum_bet=100,
        max_hands=max_hands,
    )

    # deal_card() uses pop(). Arrange a non-natural opening deal
    # in player, dealer, player, dealer order.
    shoe.cards = [
        Card(7, "Clubs"),
        second_card,
        Card(10, "Diamonds"),
        first_card,
    ]

    round_.deal_initial_cards()
    assert round_.resolve_naturals() is False

    return round_


def make_round_with_aces(max_hands=4):
    return make_round_with_pair(
        Card("A", "Hearts"),
        Card("A", "Spades"),
        max_hands=max_hands,
    )


def set_next_cards(shoe, first_card, second_card):
    """
    Shoe.deal_card() uses pop(), so the first card to be
    dealt must be last in the list.
    """
    shoe.cards = [
        second_card,
        first_card,
    ]


def test_split_aces_receive_one_card_each_and_finish():
    round_ = make_round_with_aces()

    set_next_cards(
        round_.shoe,
        first_card=Card("K", "Clubs"),
        second_card=Card(9, "Diamonds"),
    )

    round_.player_split(0)

    assert len(round_.player.hands) == 2

    first_hand = round_.player.get_hand(0)
    second_hand = round_.player.get_hand(1)

    assert len(first_hand.hand.cards) == 2
    assert len(second_hand.hand.cards) == 2

    assert first_hand.has_stood is True
    assert second_hand.has_stood is True


def test_split_ace_and_ten_is_not_natural_blackjack():
    round_ = make_round_with_aces()

    set_next_cards(
        round_.shoe,
        first_card=Card("K", "Clubs"),
        second_card=Card(9, "Diamonds"),
    )

    round_.player_split(0)

    first_hand = round_.player.get_hand(0)

    assert first_hand.hand.get_total() == 21
    assert first_hand.hand.is_blackjack() is True
    assert first_hand.is_natural_blackjack() is False


def test_ace_pair_remains_available_for_resplit():
    round_ = make_round_with_aces()

    set_next_cards(
        round_.shoe,
        first_card=Card("A", "Clubs"),
        second_card=Card(9, "Diamonds"),
    )

    round_.player_split(0)

    first_hand = round_.player.get_hand(0)
    second_hand = round_.player.get_hand(1)

    assert first_hand.hand.can_split() is True
    assert first_hand.has_stood is False

    assert second_hand.has_stood is True


def test_aces_can_be_resplit():
    round_ = make_round_with_aces()

    set_next_cards(
        round_.shoe,
        first_card=Card("A", "Clubs"),
        second_card=Card(9, "Diamonds"),
    )

    round_.player_split(0)

    # Resplit the Ace-Ace hand.
    set_next_cards(
        round_.shoe,
        first_card=Card("K", "Hearts"),
        second_card=Card(8, "Clubs"),
    )

    round_.player_split(0)

    assert len(round_.player.hands) == 3

    assert round_.player.get_hand(0).has_stood is True
    assert round_.player.get_hand(1).has_stood is True
    assert round_.player.get_hand(2).has_stood is True


def test_ace_pair_finishes_at_maximum_hands():
    round_ = make_round_with_aces(max_hands=2)

    set_next_cards(
        round_.shoe,
        first_card=Card("A", "Clubs"),
        second_card=Card(9, "Diamonds"),
    )

    round_.player_split(0)

    first_hand = round_.player.get_hand(0)

    assert len(round_.player.hands) == 2
    assert first_hand.hand.can_split() is True
    assert first_hand.has_stood is True


def test_cannot_hit_resplittable_aces():
    round_ = make_round_with_aces()

    set_next_cards(
        round_.shoe,
        first_card=Card("A", "Clubs"),
        second_card=Card(9, "Diamonds"),
    )

    round_.player_split(0)

    with pytest.raises(ValueError):
        round_.player_hit(0)


def test_cannot_double_resplittable_aces():
    round_ = make_round_with_aces()

    set_next_cards(
        round_.shoe,
        first_card=Card("A", "Clubs"),
        second_card=Card(9, "Diamonds"),
    )

    round_.player_split(0)

    with pytest.raises(ValueError):
        round_.player_double(0, 1_000)


def test_cannot_surrender_a_split_hand():
    round_ = make_round_with_pair(
        Card(8, "Hearts"),
        Card(8, "Spades"),
    )

    set_next_cards(
        round_.shoe,
        first_card=Card(2, "Clubs"),
        second_card=Card(3, "Diamonds"),
    )

    round_.player_split(0)

    with pytest.raises(ValueError):
        round_.player_surrender(0)
