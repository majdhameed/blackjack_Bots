from blackjack.cards import Card
from blackjack.player import Player
from tournament.tournament import Tournament


class DummyBot:
    def choose_bet(self, observation):
        return observation.minimum_bet

    def choose_action(self, observation):
        return observation.legal_actions[0]


def make_tournament(decks=2):
    players = [
        Player(10_000),
        Player(10_000),
    ]

    bots = [
        DummyBot(),
        DummyBot(),
    ]

    return Tournament(
        players=players,
        bots=bots,
        number_of_rounds=12,
        decks=decks,
        minimum_bet=100,
        hit_soft_17=False,
        max_hands=4,
    )


def test_reshuffle_threshold_is_twenty_percent():
    tournament = make_tournament(decks=2)

    # Two decks contain 104 cards.
    # Twenty percent becomes 20 after int().
    assert tournament.reshuffle_threshold == 20


def test_low_shoe_is_replaced_before_round():
    tournament = make_tournament(decks=2)

    original_shoe = tournament.shoe

    # Leave fewer cards than the threshold.
    tournament.shoe.cards = (
        tournament.shoe.cards[:10]
    )

    table_round = tournament.start_next_round()

    assert tournament.shoe is not original_shoe
    assert tournament.shoe.cards_remaining() == 104

    assert table_round.shoe is tournament.shoe


def test_reshuffle_resets_card_counter():
    tournament = make_tournament(decks=2)

    original_counter = tournament.card_counter

    tournament.card_counter.record_cards(
        [
            Card(2, "Hearts"),
            Card(5, "Diamonds"),
            Card("K", "Clubs"),
        ]
    )

    assert tournament.card_counter.cards_seen == 3
    assert tournament.card_counter.running_count == 1

    tournament.shoe.cards = (
        tournament.shoe.cards[:10]
    )

    table_round = tournament.start_next_round()

    # The same counter object should be reset,
    # not replaced with None or a different counter.
    assert tournament.card_counter is original_counter

    assert tournament.card_counter.cards_seen == 0
    assert tournament.card_counter.running_count == 0

    assert tournament.card_counter.get_counts() == (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    assert (
        table_round.card_counter
        is tournament.card_counter
    )


def test_shoe_is_not_replaced_at_threshold():
    tournament = make_tournament(decks=2)

    original_shoe = tournament.shoe
    original_counter = tournament.card_counter

    tournament.card_counter.record_card(
        Card(2, "Hearts")
    )

    threshold = tournament.reshuffle_threshold

    tournament.shoe.cards = (
        tournament.shoe.cards[:threshold]
    )

    table_round = tournament.start_next_round()

    # Your current condition uses:
    # cards_remaining() < threshold
    assert tournament.shoe is original_shoe
    assert tournament.shoe.cards_remaining() == threshold

    assert tournament.card_counter is original_counter
    assert tournament.card_counter.cards_seen == 1
    assert tournament.card_counter.running_count == 1

    assert table_round.shoe is original_shoe
    assert table_round.card_counter is original_counter


def test_shoe_is_not_replaced_above_threshold():
    tournament = make_tournament(decks=2)

    original_shoe = tournament.shoe

    tournament.card_counter.record_card(
        Card(6, "Spades")
    )

    tournament.shoe.cards = tournament.shoe.cards[
        :tournament.reshuffle_threshold + 5
    ]

    table_round = tournament.start_next_round()

    assert tournament.shoe is original_shoe
    assert tournament.card_counter.cards_seen == 1
    assert tournament.card_counter.running_count == 1

    assert table_round.shoe is original_shoe
    assert (
        table_round.card_counter
        is tournament.card_counter
    )