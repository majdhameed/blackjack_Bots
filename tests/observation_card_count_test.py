from dataclasses import FrozenInstanceError

import pytest

from blackjack.card_counter import CardCounter
from blackjack.cards import Card, Shoe
from blackjack.player import Player
from tournament.table_round import TableRound
from tournament.tournament import Tournament


class DummyBot:
    def choose_bet(self, observation):
        return min(
            observation.minimum_bet,
            observation.bankroll,
        )

    def choose_action(self, observation):
        return observation.legal_actions[0]


def set_next_cards(shoe, cards):
    """
    Replace the next cards without changing the
    total number of cards remaining in the shoe.

    Shoe.deal_card() uses pop(), so the future
    dealing order must be reversed.
    """
    count = len(cards)

    shoe.cards[-count:] = list(
        reversed(cards)
    )


def make_tournament(decks=1):
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


def start_betting_phase(tournament):
    table_round = tournament.start_next_round()

    round_player_index = (
        table_round.betting_order[
            table_round.next_bettor_index
        ]
    )

    return table_round, round_player_index


def prepare_action_phase():
    tournament = make_tournament(decks=1)

    table_round = tournament.start_next_round()

    tournament.place_round_bets()

    set_next_cards(
        tournament.shoe,
        [
            # First card to each player.
            Card(2, "Hearts"),
            Card(5, "Diamonds"),

            # Dealer upcard.
            Card(9, "Clubs"),

            # Second card to each player.
            Card(10, "Spades"),
            Card(9, "Hearts"),

            # Dealer hidden card.
            Card("K", "Diamonds"),
        ],
    )

    naturals_ended = (
        tournament.begin_player_actions()
    )

    assert naturals_ended is False

    return tournament, table_round


def test_first_betting_observation_has_zero_count():
    tournament = make_tournament()

    _, round_player_index = (
        start_betting_phase(tournament)
    )

    observation = (
        tournament.build_betting_observation(
            round_player_index
        )
    )

    assert observation.cards_seen == 0
    assert observation.running_count == 0
    assert observation.true_count == 0.0
    assert observation.shoe_penetration == 0.0

    assert observation.card_value_counts == (
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


def test_betting_observation_contains_prior_count():
    tournament = make_tournament(decks=2)

    visible_cards = [
        Card(2, "Hearts"),
        Card(3, "Diamonds"),
        Card("K", "Clubs"),
    ]

    tournament.card_counter.record_cards(
        visible_cards
    )

    # Simulate three cards having been removed
    # from the physical shoe.
    for _ in visible_cards:
        tournament.shoe.deal_card()

    _, round_player_index = (
        start_betting_phase(tournament)
    )

    observation = (
        tournament.build_betting_observation(
            round_player_index
        )
    )

    assert observation.cards_seen == 3

    # 2 and 3 give +2; King gives -1.
    assert observation.running_count == 1

    assert observation.card_value_counts == (
        0,  # Ace
        1,  # 2
        1,  # 3
        0,  # 4
        0,  # 5
        0,  # 6
        0,  # 7
        0,  # 8
        0,  # 9
        1,  # ten-valued
    )

    expected_cards_remaining = 104 - 3
    expected_decks_remaining = (
        expected_cards_remaining / 52
    )
    expected_true_count = (
        1 / expected_decks_remaining
    )

    assert observation.cards_remaining == (
        expected_cards_remaining
    )

    assert observation.decks_remaining == (
        pytest.approx(
            expected_decks_remaining
        )
    )

    assert observation.true_count == (
        pytest.approx(expected_true_count)
    )

    assert observation.shoe_penetration == (
        pytest.approx(3 / 104)
    )


def test_action_observation_excludes_hidden_card():
    tournament, table_round = (
        prepare_action_phase()
    )

    observation = (
        tournament.build_action_observation(
            round_player_index=0,
            hand_index=0,
        )
    )

    # Four player cards plus dealer upcard.
    assert observation.cards_seen == 5

    # Player zero: 2 (+1), 10 (-1)
    # Player one: 5 (+1), 9 (0)
    # Dealer upcard: 9 (0)
    assert observation.running_count == 1

    # Only the player's 10 has been counted.
    # Dealer's hidden King is still excluded.
    assert observation.card_value_counts[-1] == 1

    assert observation.cards_remaining == (
        table_round.shoe.cards_remaining()
    )


def test_action_observation_updates_after_hit():
    tournament, table_round = (
        prepare_action_phase()
    )

    before = (
        tournament.build_action_observation(
            round_player_index=0,
            hand_index=0,
        )
    )

    # Replace the next shoe card with a visible 2.
    set_next_cards(
        table_round.shoe,
        [Card(2, "Clubs")],
    )

    table_round.player_hit(0, 0)

    after = (
        tournament.build_action_observation(
            round_player_index=0,
            hand_index=0,
        )
    )

    assert after.cards_seen == (
        before.cards_seen + 1
    )

    assert after.running_count == (
        before.running_count + 1
    )

    assert after.card_value_counts[1] == (
        before.card_value_counts[1] + 1
    )

    assert after.cards_remaining == (
        before.cards_remaining - 1
    )


def test_reshuffle_resets_betting_observation():
    tournament = make_tournament(decks=2)

    tournament.card_counter.record_cards(
        [
            Card(2, "Hearts"),
            Card(5, "Diamonds"),
            Card("K", "Clubs"),
        ]
    )

    tournament.shoe.cards = (
        tournament.shoe.cards[:5]
    )

    table_round, round_player_index = (
        start_betting_phase(tournament)
    )

    observation = (
        tournament.build_betting_observation(
            round_player_index
        )
    )

    assert tournament.shoe.cards_remaining() == 104
    assert table_round.shoe is tournament.shoe

    assert observation.cards_seen == 0
    assert observation.running_count == 0
    assert observation.true_count == 0.0
    assert observation.shoe_penetration == 0.0

    assert observation.card_value_counts == (
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


def test_observation_does_not_expose_engine_objects():
    tournament, _ = prepare_action_phase()

    observation = (
        tournament.build_action_observation(
            round_player_index=0,
            hand_index=0,
        )
    )

    exposed_values = vars(observation).values()

    assert not any(
        isinstance(
            value,
            (CardCounter, Shoe, TableRound),
        )
        for value in exposed_values
    )


def test_observation_is_immutable():
    tournament, _ = prepare_action_phase()

    observation = (
        tournament.build_action_observation(
            round_player_index=0,
            hand_index=0,
        )
    )

    with pytest.raises(FrozenInstanceError):
        observation.running_count = 100