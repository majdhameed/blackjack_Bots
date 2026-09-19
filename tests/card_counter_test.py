import pytest

from blackjack.card_counter import CardCounter
from blackjack.cards import Card


def test_low_cards_increase_running_count():
    counter = CardCounter()

    counter.record_cards(
        [
            Card(2, "Hearts"),
            Card(4, "Diamonds"),
            Card(6, "Clubs"),
        ]
    )

    assert counter.running_count == 3
    assert counter.cards_seen == 3


def test_neutral_cards_do_not_change_count():
    counter = CardCounter()

    counter.record_cards(
        [
            Card(7, "Hearts"),
            Card(8, "Diamonds"),
            Card(9, "Clubs"),
        ]
    )

    assert counter.running_count == 0
    assert counter.cards_seen == 3


def test_high_cards_decrease_running_count():
    counter = CardCounter()

    counter.record_cards(
        [
            Card(10, "Hearts"),
            Card("K", "Diamonds"),
            Card("A", "Clubs"),
        ]
    )

    assert counter.running_count == -3
    assert counter.cards_seen == 3


def test_mixed_card_sequence():
    counter = CardCounter()

    counter.record_cards(
        [
            Card(2, "Hearts"),
            Card(5, "Diamonds"),
            Card(8, "Clubs"),
            Card("K", "Spades"),
            Card("A", "Hearts"),
        ]
    )

    assert counter.running_count == 0
    assert counter.cards_seen == 5

    # Order:
    # Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10
    assert counter.get_counts() == (
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
    )


def test_face_cards_share_ten_value_bucket():
    counter = CardCounter()

    counter.record_cards(
        [
            Card(10, "Hearts"),
            Card("J", "Diamonds"),
            Card("Q", "Clubs"),
            Card("K", "Spades"),
        ]
    )

    counts = counter.get_counts()

    assert counts[-1] == 4
    assert counter.running_count == -4
    assert counter.cards_seen == 4


def test_true_count():
    counter = CardCounter()

    # Six low cards produce a running count of +6.
    counter.record_cards(
        [
            Card(2, "Hearts"),
            Card(2, "Diamonds"),
            Card(3, "Clubs"),
            Card(4, "Spades"),
            Card(5, "Hearts"),
            Card(6, "Diamonds"),
        ]
    )

    assert counter.running_count == 6
    assert counter.get_true_count(3) == 2


def test_true_count_can_be_fractional():
    counter = CardCounter()

    counter.record_cards(
        [
            Card(2, "Hearts"),
            Card(3, "Diamonds"),
            Card(10, "Clubs"),
        ]
    )

    # Running count is +1.
    assert counter.get_true_count(2) == 0.5


def test_reset_clears_counter():
    counter = CardCounter()

    counter.record_cards(
        [
            Card(2, "Hearts"),
            Card(5, "Diamonds"),
            Card("A", "Clubs"),
        ]
    )

    counter.reset()

    assert counter.cards_seen == 0
    assert counter.running_count == 0

    assert counter.get_counts() == (
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


def test_record_card_rejects_invalid_object():
    counter = CardCounter()

    with pytest.raises(TypeError):
        counter.record_card("Ace of Hearts")


def test_true_count_rejects_zero_decks():
    counter = CardCounter()

    with pytest.raises(ValueError):
        counter.get_true_count(0)


def test_true_count_rejects_negative_decks():
    counter = CardCounter()

    with pytest.raises(ValueError):
        counter.get_true_count(-1)


def test_true_count_rejects_invalid_type():
    counter = CardCounter()

    with pytest.raises(TypeError):
        counter.get_true_count("three")