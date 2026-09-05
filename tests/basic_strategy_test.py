import pytest

from blackjack.actions import Action
from blackjack.basic_strategy import choose_action
from blackjack.cards import Card
from blackjack.dealer import Dealer
from blackjack.hand import Hand


DEALER_RANKS = [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]
ACTION_CODES = {
    "H": Action.HIT,
    "S": Action.STAND,
    "D": Action.DOUBLE,
    "P": Action.SPLIT,
    "R": Action.SURRENDER,
}


def make_hand(*ranks):
    hand = Hand()
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    for index, rank in enumerate(ranks):
        hand.add_card(Card(rank, suits[index % len(suits)]))
    return hand


def hard_hand(total):
    if total <= 11:
        return make_hand(2, total - 2)
    return make_hand(10, total - 10)


@pytest.mark.parametrize(
    ("hit_soft_17", "total", "actions"),
    [
        (False, 4, "HHHHHHHHHH"),
        (False, 9, "HDDDDHHHHH"),
        (False, 10, "DDDDDDDDHH"),
        (False, 11, "DDDDDDDDDH"),
        (False, 12, "HHSSSHHHHH"),
        (False, 13, "SSSSSHHHHH"),
        (False, 14, "SSSSSHHHHH"),
        (False, 15, "SSSSSHHHRH"),
        (False, 16, "SSSSSHHRRR"),
        (False, 17, "SSSSSSSSSS"),
        (False, 18, "SSSSSSSSSS"),
        (True, 4, "HHHHHHHHHH"),
        (True, 9, "HDDDDHHHHH"),
        (True, 10, "DDDDDDDDHH"),
        (True, 11, "DDDDDDDDDD"),
        (True, 12, "HHSSSHHHHH"),
        (True, 13, "SSSSSHHHHH"),
        (True, 14, "SSSSSHHHHH"),
        (True, 15, "SSSSSHHHRR"),
        (True, 16, "SSSSSHHRRR"),
        (True, 17, "SSSSSSSSSR"),
        (True, 18, "SSSSSSSSSS"),
    ],
)
def test_hard_total_chart(hit_soft_17, total, actions):
    hand = hard_hand(total)

    for dealer_rank, action_code in zip(DEALER_RANKS, actions):
        action = choose_action(
            hand,
            Card(dealer_rank, "Spades"),
            can_double=True,
            can_split=False,
            can_surrender=True,
            hit_soft_17=hit_soft_17,
        )
        assert action is ACTION_CODES[action_code]


@pytest.mark.parametrize(
    ("hit_soft_17", "total", "actions"),
    [
        (False, 13, "HHHDDHHHHH"),
        (False, 14, "HHHDDHHHHH"),
        (False, 15, "HHDDDHHHHH"),
        (False, 16, "HHDDDHHHHH"),
        (False, 17, "HDDDDHHHHH"),
        (False, 18, "SDDDDSSHHH"),
        (False, 19, "SSSSSSSSSS"),
        (True, 13, "HHHDDHHHHH"),
        (True, 14, "HHHDDHHHHH"),
        (True, 15, "HHDDDHHHHH"),
        (True, 16, "HHDDDHHHHH"),
        (True, 17, "HDDDDHHHHH"),
        (True, 18, "DDDDDSSHHH"),
        (True, 19, "SSSSDSSSSS"),
        (True, 20, "SSSSSSSSSS"),
    ],
)
def test_soft_total_chart(hit_soft_17, total, actions):
    hand = make_hand("A", total - 11)

    for dealer_rank, action_code in zip(DEALER_RANKS, actions):
        action = choose_action(
            hand,
            Card(dealer_rank, "Spades"),
            can_double=True,
            can_split=False,
            can_surrender=True,
            hit_soft_17=hit_soft_17,
        )
        assert action is ACTION_CODES[action_code]


@pytest.mark.parametrize(
    ("hit_soft_17", "rank", "actions"),
    [
        (False, 2, "PPPPPPHHHH"),
        (False, 3, "PPPPPPHHHH"),
        (False, 4, "HHHPPHHHHH"),
        (False, 6, "PPPPPHHHHH"),
        (False, 7, "PPPPPPHHHH"),
        (False, 8, "PPPPPPPPPP"),
        (False, 9, "PPPPPSPPSS"),
        (False, "A", "PPPPPPPPPP"),
        (True, 2, "PPPPPPHHHH"),
        (True, 3, "PPPPPPHHHH"),
        (True, 4, "HHHPPHHHHH"),
        (True, 6, "PPPPPHHHHH"),
        (True, 7, "PPPPPPHHHH"),
        (True, 8, "PPPPPPPPPR"),
        (True, 9, "PPPPPSPPSS"),
        (True, "A", "PPPPPPPPPP"),
    ],
)
def test_pair_chart(hit_soft_17, rank, actions):
    hand = make_hand(rank, rank)

    for dealer_rank, action_code in zip(DEALER_RANKS, actions):
        action = choose_action(
            hand,
            Card(dealer_rank, "Spades"),
            can_double=True,
            can_split=True,
            can_surrender=True,
            hit_soft_17=hit_soft_17,
        )
        assert action is ACTION_CODES[action_code]


def test_conditional_actions_use_the_chart_fallbacks():
    dealer_ten = Card(10, "Spades")

    assert choose_action(hard_hand(11), dealer_ten, False, False, False) is Action.HIT
    assert choose_action(make_hand("A", 7), Card(3, "Spades"), False, False, False) is Action.STAND
    assert choose_action(hard_hand(16), dealer_ten, False, False, False) is Action.HIT
    assert choose_action(hard_hand(17), Card("A", "Spades"), False, False, False, True) is Action.STAND
    assert choose_action(make_hand(8, 8), Card("A", "Spades"), False, True, False, True) is Action.SPLIT


def test_dealer_soft_17_rule_can_be_passed_directly():
    dealer = Dealer(True)
    action = choose_action(
        hard_hand(11),
        Card("A", "Spades"),
        can_double=True,
        can_split=False,
        can_surrender=False,
        hit_soft_17=dealer.hit_soft_17,
    )

    assert action is Action.DOUBLE
