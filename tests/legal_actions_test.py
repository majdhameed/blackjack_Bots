from blackjack.actions import Action
from blackjack.cards import Card, Shoe
from blackjack.player import Player
from blackjack.round import Round


def make_round(
    player_cards,
    dealer_cards=None,
    bankroll=10_000,
    bet=1_000,
    max_hands=4,
    check_naturals=True,
):
    if dealer_cards is None:
        dealer_cards = [
            Card(10, "Clubs"),
            Card(8, "Diamonds"),
        ]

    player = Player(bankroll)
    player.place_bet(
        bet,
        minimum_bet=100,
    )

    shoe = Shoe(1)

    # Initial deal order:
    # player 1, dealer 1, player 2, dealer 2.
    # Shoe uses pop(), so store them backward.
    shoe.cards = [
        dealer_cards[1],
        player_cards[1],
        dealer_cards[0],
        player_cards[0],
    ]

    round_ = Round(
        player=player,
        shoe=shoe,
        minimum_bet=100,
        max_hands=max_hands,
    )

    round_.deal_initial_cards()

    if check_naturals:
        assert round_.resolve_naturals() is False

    return round_


def test_no_legal_actions_before_natural_check():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ],
        check_naturals=False,
    )

    assert round_.get_legal_actions(0) == set()


def test_normal_two_card_hand_actions():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ]
    )

    legal_actions = round_.get_legal_actions(0)

    assert Action.HIT in legal_actions
    assert Action.STAND in legal_actions
    assert Action.DOUBLE in legal_actions
    assert Action.SURRENDER in legal_actions
    assert Action.SPLIT not in legal_actions


def test_pair_can_split_with_enough_bankroll():
    round_ = make_round(
        player_cards=[
            Card(8, "Hearts"),
            Card(8, "Spades"),
        ]
    )

    legal_actions = round_.get_legal_actions(0)

    assert Action.SPLIT in legal_actions


def test_pair_cannot_split_without_enough_bankroll():
    round_ = make_round(
        player_cards=[
            Card(8, "Hearts"),
            Card(8, "Spades"),
        ],
        bankroll=1_000,
        bet=1_000,
    )

    legal_actions = round_.get_legal_actions(0)

    assert Action.SPLIT not in legal_actions
    assert Action.DOUBLE not in legal_actions


def test_double_and_surrender_unavailable_after_hit():
    round_ = make_round(
        player_cards=[
            Card(5, "Hearts"),
            Card(6, "Spades"),
        ]
    )

    round_.shoe.cards = [
        Card(2, "Clubs"),
    ]

    round_.player_hit(0)

    legal_actions = round_.get_legal_actions(0)

    assert Action.HIT in legal_actions
    assert Action.STAND in legal_actions
    assert Action.DOUBLE not in legal_actions
    assert Action.SURRENDER not in legal_actions


def test_no_legal_actions_after_standing():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ]
    )

    round_.player_stand(0)

    assert round_.get_legal_actions(0) == set()


def test_no_legal_actions_after_busting():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ]
    )

    round_.shoe.cards = [
        Card("K", "Clubs"),
    ]

    round_.player_hit(0)

    assert (
        round_.player.get_hand(0).hand.is_bust()
        is True
    )
    assert round_.get_legal_actions(0) == set()


def test_split_hand_cannot_surrender():
    round_ = make_round(
        player_cards=[
            Card(8, "Hearts"),
            Card(8, "Spades"),
        ]
    )

    # First split hand receives 2.
    # Second split hand receives 3.
    round_.shoe.cards = [
        Card(3, "Diamonds"),
        Card(2, "Clubs"),
    ]

    round_.player_split(0)

    first_actions = round_.get_legal_actions(0)
    second_actions = round_.get_legal_actions(1)

    assert Action.SURRENDER not in first_actions
    assert Action.SURRENDER not in second_actions

    assert Action.HIT in first_actions
    assert Action.STAND in first_actions


def test_resplittable_aces_only_allow_split_or_stand():
    round_ = make_round(
        player_cards=[
            Card("A", "Hearts"),
            Card("A", "Spades"),
        ]
    )

    # First hand becomes A-A and can be resplit.
    # Second hand becomes A-9 and finishes.
    round_.shoe.cards = [
        Card(9, "Diamonds"),
        Card("A", "Clubs"),
    ]

    round_.player_split(0)

    first_actions = round_.get_legal_actions(0)
    second_actions = round_.get_legal_actions(1)

    assert first_actions == {
        Action.SPLIT,
        Action.STAND,
    }

    assert second_actions == set()


def test_ace_pair_cannot_resplit_at_maximum_hands():
    round_ = make_round(
        player_cards=[
            Card("A", "Hearts"),
            Card("A", "Spades"),
        ],
        max_hands=2,
    )

    round_.shoe.cards = [
        Card(9, "Diamonds"),
        Card("A", "Clubs"),
    ]

    round_.player_split(0)

    first_hand = round_.player.get_hand(0)

    assert first_hand.has_stood is True
    assert round_.get_legal_actions(0) == set()


def test_split_ace_hand_cannot_hit_or_double():
    round_ = make_round(
        player_cards=[
            Card("A", "Hearts"),
            Card("A", "Spades"),
        ]
    )

    round_.shoe.cards = [
        Card(9, "Diamonds"),
        Card("A", "Clubs"),
    ]

    round_.player_split(0)

    legal_actions = round_.get_legal_actions(0)

    assert Action.HIT not in legal_actions
    assert Action.DOUBLE not in legal_actions
    assert Action.SURRENDER not in legal_actions


def test_no_legal_actions_after_dealer_turn():
    round_ = make_round(
        player_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ]
    )

    round_.player_stand(0)
    round_.play_dealer()

    assert round_.dealer_turn_complete is True
    assert round_.get_legal_actions(0) == set()


def test_no_legal_actions_after_natural_blackjack():
    player = Player(10_000)
    player.place_bet(
        1_000,
        minimum_bet=100,
    )

    shoe = Shoe(1)
    shoe.cards = [
        Card(8, "Diamonds"),
        Card("K", "Spades"),
        Card(10, "Clubs"),
        Card("A", "Hearts"),
    ]

    round_ = Round(
        player=player,
        shoe=shoe,
        minimum_bet=100,
    )

    round_.deal_initial_cards()

    assert round_.resolve_naturals() is True
    assert round_.is_over is True
    assert round_.get_legal_actions(0) == set()