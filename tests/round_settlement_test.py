import pytest

from blackjack.cards import Card, Shoe
from blackjack.player import Player
from blackjack.round import Round


def make_round(
    player_cards,
    dealer_cards,
    bankroll=10_000,
    bet=1_000,
):
    player = Player(bankroll)
    player.place_bet(bet, minimum_bet=100)

    shoe = Shoe(1)

    round_ = Round(
        player=player,
        shoe=shoe,
        minimum_bet=100,
    )

    # deal_card() uses pop(). Arrange the shoe so the round deals
    # player 1, dealer 1, player 2, dealer 2.
    shoe.cards = [
        dealer_cards[1],
        player_cards[1],
        dealer_cards[0],
        player_cards[0],
    ]

    round_.deal_initial_cards()
    assert round_.resolve_naturals() is False
    return round_


def finish_hand(player, hand_index=0):
    player_hand = player.get_hand(hand_index)

    if not player_hand.is_finished():
        player.stand(hand_index)


def test_cannot_play_dealer_before_natural_check():
    player = Player(10_000)
    shoe = Shoe(1)

    round_ = Round(
        player=player,
        shoe=shoe,
        minimum_bet=100,
    )

    with pytest.raises(
        ValueError,
        match="Naturals must be checked first",
    ):
        round_.play_dealer()


def test_cannot_play_dealer_during_active_hand():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(6, "Diamonds"),
        ],
    )

    with pytest.raises(ValueError):
        round_.play_dealer()


def test_dealer_does_not_play_if_player_busted():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Diamonds"),
            Card(6, "Hearts"),
        ],
    )


    round_.shoe.cards = [
        Card(5, "Spades"),
    ]

    round_.player_hit(0)

    assert (
        round_.player.get_hand(0).hand.is_bust()
        is True
    )

    dealer_played = round_.play_dealer()

    assert dealer_played is False
    assert round_.dealer_turn_complete is True

def test_dealer_does_not_play_if_player_surrendered():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(6, "Diamonds"),
        ],
    )

    round_.player_surrender(0)

    dealer_played = round_.play_dealer()

    assert dealer_played is False
    assert round_.dealer_turn_complete is True


def test_dealer_draws_when_player_has_live_hand():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(6, "Diamonds"),
        ],
    )

    round_.player_stand(0)

    # deal_card() uses pop(), so this is the next card.
    round_.shoe.cards = [
        Card(4, "Hearts"),
    ]

    dealer_played = round_.play_dealer()

    assert dealer_played is True
    assert round_.dealer.hand.get_total() == 20
    assert round_.dealer_turn_complete is True


def test_cannot_play_dealer_after_round_is_over():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(7, "Diamonds"),
        ],
    )

    round_.is_over = True

    with pytest.raises(ValueError):
        round_.play_dealer()


def test_cannot_settle_before_dealer_turn():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(7, "Diamonds"),
        ],
    )

    round_.player_stand(0)

    with pytest.raises(ValueError):
        round_.settle_hand(0)


def test_cannot_settle_active_hand():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(8, "Diamonds"),
        ],
    )

    # Set directly so this test specifically reaches the
    # active-hand validation inside settle_hand().
    round_.dealer_turn_complete = True

    with pytest.raises(ValueError):
        round_.settle_hand(0)


def test_player_wins_with_higher_total():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card("Q", "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(8, "Diamonds"),
        ],
    )

    round_.player_stand(0)
    round_.play_dealer()

    outcome = round_.settle_hand(0)
    player_hand = round_.player.get_hand(0)

    assert outcome == "win"
    assert round_.player.bankroll == 11_000
    assert player_hand.is_settled is True
    assert player_hand.outcome == "win"


def test_player_loses_with_lower_total():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card("Q", "Diamonds"),
        ],
    )

    round_.player_stand(0)
    round_.play_dealer()

    outcome = round_.settle_hand(0)
    player_hand = round_.player.get_hand(0)

    assert outcome == "loss"
    assert round_.player.bankroll == 9_000
    assert player_hand.is_settled is True
    assert player_hand.outcome == "loss"


def test_equal_totals_push():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(8, "Diamonds"),
        ],
    )

    round_.player_stand(0)
    round_.play_dealer()

    outcome = round_.settle_hand(0)
    player_hand = round_.player.get_hand(0)

    assert outcome == "push"
    assert round_.player.bankroll == 10_000
    assert player_hand.outcome == "push"


def test_player_wins_when_dealer_busts():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(6, "Diamonds"),
        ],
    )

    round_.player_stand(0)

    round_.shoe.cards = [
        Card("K", "Hearts"),
    ]

    round_.play_dealer()
    outcome = round_.settle_hand(0)

    assert round_.dealer.hand.is_bust() is True
    assert outcome == "win"
    assert round_.player.bankroll == 11_000


def test_busted_player_loses():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Diamonds"),
            Card(8, "Hearts"),
        ],
    )

    round_.shoe.cards = [
        Card(5, "Clubs"),
    ]

    round_.player_hit(0)

    assert (
        round_.player.get_hand(0).hand.is_bust()
        is True
    )

    round_.play_dealer()
    outcome = round_.settle_hand(0)

    assert outcome == "loss"
    assert round_.player.bankroll == 9_000
    assert (
        round_.player.get_hand(0).outcome
        == "loss"
    )

def test_surrender_is_not_paid_twice():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(8, "Diamonds"),
        ],
    )

    round_.player_surrender(0)

    assert round_.player.bankroll == 9_500

    round_.play_dealer()
    outcome = round_.settle_hand(0)

    assert outcome == "surrender"
    assert round_.player.bankroll == 9_500
    assert round_.player.get_hand(0).outcome == "surrender"


def test_cannot_settle_same_hand_twice():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Clubs"),
            Card(8, "Diamonds"),
        ],
    )

    round_.player_stand(0)
    round_.play_dealer()
    round_.settle_hand(0)

    with pytest.raises(ValueError):
        round_.settle_hand(0)


def test_split_hands_settle_independently():
    round_ = make_round(
        player_cards=[
            Card(9, "Hearts"),
            Card(9, "Spades"),
        ],
        dealer_cards=[
            Card(10, "Hearts"),
            Card(8, "Clubs"),
        ],
    )

    # Hand 0 becomes 19; Hand 1 becomes 17.
    round_.shoe.cards = [
        Card(8, "Diamonds"),
        Card("K", "Clubs"),
    ]
    round_.player_split(0)
    round_.player_stand(0)
    round_.player_stand(1)

    round_.play_dealer()

    first_outcome = round_.settle_hand(0)
    second_outcome = round_.settle_hand(1)

    assert first_outcome == "win"
    assert second_outcome == "loss"

    assert round_.player.get_hand(0).outcome == "win"
    assert round_.player.get_hand(1).outcome == "loss"

    # 10,000 - 1,000 original bet - 1,000 split bet
    # + 2,000 from the winning hand.
    assert round_.player.bankroll == 10_000


def test_split_ace_and_ten_receives_normal_win():
    round_ = make_round(
        player_cards=[
            Card("A", "Hearts"),
            Card("A", "Spades"),
        ],
        dealer_cards=[
            Card(10, "Hearts"),
            Card("Q", "Spades"),
        ],
    )

    round_.shoe.cards = [
        Card(9, "Diamonds"),
        Card("K", "Clubs"),
    ]
    round_.player_split(0)

    round_.play_dealer()
    outcome = round_.settle_hand(0)

    assert outcome == "win"

    # Bankroll was 8,000 after the original bet and split.
    # A normal win adds 2,000, not the 2,500 blackjack return.
    assert round_.player.bankroll == 10_000
