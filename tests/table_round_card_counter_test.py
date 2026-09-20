from blackjack.card_counter import CardCounter
from blackjack.cards import Card, Shoe
from blackjack.player import Player
from tournament.table_round import TableRound


def set_next_cards(shoe, cards):
    """
    Shoe.deal_card() uses pop(), so cards must
    be stored in reverse dealing order.
    """
    shoe.cards = list(reversed(cards))


def make_table():
    players = [
        Player(1_000),
        Player(1_000),
    ]

    shoe = Shoe(1)
    counter = CardCounter()

    table = TableRound(
        players=players,
        shoe=shoe,
        minimum_bet=100,
        starting_player=1,
        hit_soft_17=False,
        max_hands=4,
        card_counter=counter,
    )

    table.place_bet(0, 100)
    table.place_bet(1, 100)

    return table, counter


def deal_cards(
    table,
    player_zero_cards,
    player_one_cards,
    dealer_cards,
):
    set_next_cards(
        table.shoe,
        [
            player_zero_cards[0],
            player_one_cards[0],
            dealer_cards[0],
            player_zero_cards[1],
            player_one_cards[1],
            dealer_cards[1],
        ],
    )

    table.deal_initial_cards()


def test_initial_deal_counts_only_visible_cards():
    table, counter = make_table()

    deal_cards(
        table,
        player_zero_cards=[
            Card(2, "Hearts"),
            Card(10, "Spades"),
        ],
        player_one_cards=[
            Card(5, "Diamonds"),
            Card(9, "Clubs"),
        ],
        dealer_cards=[
            Card("A", "Spades"),
            Card("K", "Hearts"),
        ],
    )

    # Four player cards plus the dealer upcard.
    assert counter.cards_seen == 5

    # 2: +1
    # 10: -1
    # 5: +1
    # 9:  0
    # Ace: -1
    assert counter.running_count == 0

    # Dealer's hidden King must not be counted yet.
    assert counter.get_counts()[-1] == 1


def test_dealer_blackjack_reveals_hole_card():
    table, counter = make_table()

    deal_cards(
        table,
        player_zero_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        player_one_cards=[
            Card(9, "Diamonds"),
            Card(7, "Clubs"),
        ],
        dealer_cards=[
            Card("A", "Spades"),
            Card("K", "Hearts"),
        ],
    )

    assert counter.cards_seen == 5

    round_ended = table.resolve_naturals()

    assert round_ended is True
    assert table.is_over is True

    # The dealer revealed the hidden King.
    assert counter.cards_seen == 6

    # Player 10, dealer Ace, and dealer King
    # are all high cards.
    assert counter.running_count == -3

    assert counter.get_counts()[-1] == 2


def test_player_hit_card_is_counted():
    table, counter = make_table()

    deal_cards(
        table,
        player_zero_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ],
        player_one_cards=[
            Card(10, "Diamonds"),
            Card(7, "Clubs"),
        ],
        dealer_cards=[
            Card(9, "Spades"),
            Card(7, "Hearts"),
        ],
    )

    table.resolve_naturals()

    cards_seen_before = counter.cards_seen
    count_before = counter.running_count

    set_next_cards(
        table.shoe,
        [Card(2, "Diamonds")],
    )

    table.player_hit(0, 0)

    assert counter.cards_seen == (
        cards_seen_before + 1
    )

    assert counter.running_count == (
        count_before + 1
    )


def test_double_card_is_counted():
    table, counter = make_table()

    deal_cards(
        table,
        player_zero_cards=[
            Card(5, "Hearts"),
            Card(6, "Spades"),
        ],
        player_one_cards=[
            Card(10, "Diamonds"),
            Card(7, "Clubs"),
        ],
        dealer_cards=[
            Card(9, "Spades"),
            Card(7, "Hearts"),
        ],
    )

    table.resolve_naturals()

    cards_seen_before = counter.cards_seen
    count_before = counter.running_count

    set_next_cards(
        table.shoe,
        [Card(3, "Diamonds")],
    )

    table.player_double(
        player_index=0,
        hand_index=0,
        additional_bet=100,
    )

    assert counter.cards_seen == (
        cards_seen_before + 1
    )

    assert counter.running_count == (
        count_before + 1
    )


def test_both_split_cards_are_counted():
    table, counter = make_table()

    deal_cards(
        table,
        player_zero_cards=[
            Card(8, "Hearts"),
            Card(8, "Spades"),
        ],
        player_one_cards=[
            Card(10, "Diamonds"),
            Card(7, "Clubs"),
        ],
        dealer_cards=[
            Card(9, "Spades"),
            Card(7, "Hearts"),
        ],
    )

    table.resolve_naturals()

    cards_seen_before = counter.cards_seen
    count_before = counter.running_count

    # The 2 adds one and the King subtracts one,
    # so the running count should not change.
    set_next_cards(
        table.shoe,
        [
            Card(2, "Diamonds"),
            Card("K", "Clubs"),
        ],
    )

    table.player_split(0, 0)

    assert counter.cards_seen == (
        cards_seen_before + 2
    )

    assert counter.running_count == count_before


def test_dealer_play_counts_hole_card_and_draws():
    table, counter = make_table()

    deal_cards(
        table,
        player_zero_cards=[
            Card(10, "Hearts"),
            Card(8, "Spades"),
        ],
        player_one_cards=[
            Card(10, "Diamonds"),
            Card(7, "Clubs"),
        ],
        dealer_cards=[
            Card(10, "Spades"),
            Card(6, "Hearts"),
        ],
    )

    table.resolve_naturals()

    table.player_stand(0, 0)
    table.player_stand(1, 0)

    assert table.get_current_player_index() is None

    cards_seen_before = counter.cards_seen
    count_before = counter.running_count

    # Dealer has 16 and draws a King.
    set_next_cards(
        table.shoe,
        [Card("K", "Diamonds")],
    )

    dealer_played = table.play_dealer()

    assert dealer_played is True
    assert table.dealer.hand.is_bust() is True

    # Dealer hole card plus one dealer draw.
    assert counter.cards_seen == (
        cards_seen_before + 2
    )

    # Hidden 6 adds one; drawn King subtracts one.
    assert counter.running_count == count_before


def test_hidden_card_remains_unseen_when_dealer_does_not_play():
    table, counter = make_table()

    deal_cards(
        table,
        player_zero_cards=[
            Card(10, "Hearts"),
            Card(6, "Spades"),
        ],
        player_one_cards=[
            Card(10, "Diamonds"),
            Card(7, "Clubs"),
        ],
        dealer_cards=[
            Card(9, "Spades"),
            Card("K", "Hearts"),
        ],
    )

    table.resolve_naturals()

    # Player zero busts.
    set_next_cards(
        table.shoe,
        [Card(10, "Clubs")],
    )
    table.player_hit(0, 0)

    # Player one surrenders.
    table.player_surrender(1, 0)

    cards_seen_before = counter.cards_seen
    count_before = counter.running_count

    dealer_played = table.play_dealer()

    assert dealer_played is False

    # Dealer never reveals the hidden King.
    assert counter.cards_seen == cards_seen_before
    assert counter.running_count == count_before