import pytest

from blackjack.actions import Action
from blackjack.card_counter import CardCounter
from blackjack.cards import Card, Shoe
from blackjack.player import Player
from tournament.table_round import TableRound


def set_next_cards(shoe, cards):
    """
    Set cards in their future deal order.

    Shoe.deal_card() uses pop(), so the list must
    be stored in reverse.
    """
    shoe.cards = list(reversed(cards))


def make_table(
    player_count=3,
    bankroll=1_000,
    minimum_bet=100,
    starting_player=1,
):
    players = [
        Player(bankroll)
        for _ in range(player_count)
    ]

    card_counter = CardCounter()

    table = TableRound(
        players=players,
        shoe=Shoe(1),
        minimum_bet=minimum_bet,
        starting_player=starting_player,
        hit_soft_17=False,
        max_hands=4,
        card_counter=card_counter
    )

    return table


def place_all_bets(table, bet=100):
    for player_index in table.betting_order:
        table.place_bet(player_index, bet)


def prepare_normal_table():
    table = make_table()
    place_all_bets(table)

    set_next_cards(
        table.shoe,
        [
            # First card to each player.
            Card(10, "Hearts"),
            Card(9, "Hearts"),
            Card(8, "Hearts"),

            # Dealer upcard.
            Card(10, "Clubs"),

            # Second card to each player.
            Card(6, "Spades"),
            Card(8, "Spades"),
            Card(9, "Spades"),

            # Dealer hole card.
            Card(6, "Diamonds"),
        ],
    )

    table.deal_initial_cards()
    table.resolve_naturals()

    return table


def test_starting_player_one_creates_normal_betting_order():
    table = make_table(starting_player=1)

    assert table.starting_player_index == 0
    assert table.betting_order == [0, 1, 2]


def test_betting_order_wraps_around_table():
    table = make_table(starting_player=2)

    assert table.starting_player_index == 1
    assert table.betting_order == [1, 2, 0]


def test_zero_starting_player_selects_random_player(
    monkeypatch,
):
    monkeypatch.setattr(
        "tournament.table_round.random.randint",
        lambda minimum, maximum: 2,
    )

    table = make_table(starting_player=0)

    assert table.starting_player_index == 2
    assert table.betting_order == [2, 0, 1]


def test_player_cannot_bet_out_of_order():
    table = make_table(starting_player=2)

    # Player 1, internal index 1, must bet first.
    with pytest.raises(Exception):
        table.place_bet(0, 100)

    table.place_bet(1, 100)

    assert table.next_bettor_index == 1


def test_betting_complete_after_every_player_bets():
    table = make_table()

    assert table.betting_complete() is False

    place_all_bets(table)

    assert table.betting_complete() is True

    for player in table.players:
        assert len(player.hands) == 1
        assert player.bankroll == 900


def test_cannot_deal_before_betting_is_complete():
    table = make_table()

    table.place_bet(0, 100)

    with pytest.raises(ValueError):
        table.deal_initial_cards()


def test_initial_cards_are_dealt_in_expected_order():
    table = make_table()
    place_all_bets(table)

    cards = [
        Card(2, "Hearts"),
        Card(3, "Hearts"),
        Card(4, "Hearts"),
        Card(5, "Clubs"),
        Card(6, "Spades"),
        Card(7, "Spades"),
        Card(8, "Spades"),
        Card(9, "Diamonds"),
    ]

    set_next_cards(table.shoe, cards)
    table.deal_initial_cards()

    assert table.players[0].get_hand(
        0
    ).hand.cards == [cards[0], cards[4]]

    assert table.players[1].get_hand(
        0
    ).hand.cards == [cards[1], cards[5]]

    assert table.players[2].get_hand(
        0
    ).hand.cards == [cards[2], cards[6]]

    assert table.dealer.hand.cards == [
        cards[3],
        cards[7],
    ]


def test_player_blackjack_is_paid_and_skipped():
    table = make_table()
    place_all_bets(table)

    set_next_cards(
        table.shoe,
        [
            Card("A", "Hearts"),
            Card(10, "Hearts"),
            Card(9, "Hearts"),
            Card(10, "Clubs"),
            Card("K", "Spades"),
            Card(7, "Spades"),
            Card(8, "Spades"),
            Card(6, "Diamonds"),
        ],
    )

    table.deal_initial_cards()

    round_ended = table.resolve_naturals()

    first_hand = table.players[0].get_hand(0)

    assert round_ended is False
    assert first_hand.is_settled is True
    assert first_hand.outcome == "blackjack"
    assert table.players[0].bankroll == 1_150

    # Player zero is skipped because their natural
    # has already been settled.
    assert table.get_current_player_index() == 1


def test_dealer_blackjack_settles_every_player():
    table = make_table()
    place_all_bets(table)

    set_next_cards(
        table.shoe,
        [
            Card("A", "Hearts"),
            Card(10, "Hearts"),
            Card(9, "Hearts"),
            Card("A", "Clubs"),
            Card("K", "Spades"),
            Card(7, "Spades"),
            Card(8, "Spades"),
            Card("Q", "Diamonds"),
        ],
    )

    table.deal_initial_cards()

    round_ended = table.resolve_naturals()

    assert round_ended is True
    assert table.is_over is True
    assert table.dealer_turn_complete is True

    # Player zero also had blackjack, so they push.
    assert table.players[0].bankroll == 1_000
    assert table.players[0].get_hand(
        0
    ).outcome == "push"

    assert table.players[1].bankroll == 900
    assert table.players[1].get_hand(
        0
    ).outcome == "dealer_blackjack"

    assert table.players[2].bankroll == 900
    assert table.players[2].get_hand(
        0
    ).outcome == "dealer_blackjack"


def test_player_cannot_act_out_of_turn():
    table = prepare_normal_table()

    assert table.get_current_player_index() == 0

    assert table.get_legal_actions(1, 0) == set()

    with pytest.raises(ValueError):
        table.player_stand(1, 0)


def test_hit_keeps_turn_when_hand_remains_active():
    table = prepare_normal_table()

    set_next_cards(
        table.shoe,
        [Card(2, "Diamonds")],
    )

    card = table.player_hit(0, 0)

    assert card.rank == 2
    assert table.players[0].get_hand(
        0
    ).hand.get_total() == 18

    assert table.get_current_player_index() == 0


def test_stand_advances_to_next_player():
    table = prepare_normal_table()

    table.player_stand(0, 0)

    assert table.get_current_player_index() == 1


def test_bust_advances_to_next_player():
    table = prepare_normal_table()

    set_next_cards(
        table.shoe,
        [Card(10, "Diamonds")],
    )

    table.player_hit(0, 0)

    assert table.players[0].get_hand(
        0
    ).hand.is_bust()

    assert table.get_current_player_index() == 1


def test_dealer_cannot_play_during_player_turn():
    table = prepare_normal_table()

    with pytest.raises(ValueError):
        table.play_dealer()


def test_complete_round_with_win_loss_and_surrender():
    table = prepare_normal_table()

    # Player 0 has 16 and stands.
    table.player_stand(0, 0)

    # Player 1 has 17 and stands.
    table.player_stand(1, 0)

    # Player 2 has 17 and surrenders.
    table.player_surrender(2, 0)

    assert table.get_current_player_index() is None

    # Dealer has 16 and draws a ten, causing a bust.
    set_next_cards(
        table.shoe,
        [Card(10, "Diamonds")],
    )

    dealer_played = table.play_dealer()

    assert dealer_played is True
    assert table.dealer.hand.is_bust() is True

    outcomes = table.settle_round()

    assert outcomes == [
        ["win"],
        ["win"],
        ["surrender"],
    ]

    assert table.players[0].bankroll == 1_100
    assert table.players[1].bankroll == 1_100
    assert table.players[2].bankroll == 950

    assert table.is_over is True


def test_dealer_does_not_draw_when_no_hands_are_playable():
    table = prepare_normal_table()

    # Bust player zero.
    set_next_cards(
        table.shoe,
        [Card(10, "Diamonds")],
    )
    table.player_hit(0, 0)

    # Surrender player one.
    table.player_surrender(1, 0)

    # Surrender player two.
    table.player_surrender(2, 0)

    dealer_cards_before = len(
        table.dealer.hand.cards
    )

    dealer_played = table.play_dealer()

    assert dealer_played is False
    assert table.dealer_turn_complete is True
    assert len(table.dealer.hand.cards) == (
        dealer_cards_before
    )

    outcomes = table.settle_round()

    assert outcomes == [
        ["loss"],
        ["surrender"],
        ["surrender"],
    ]