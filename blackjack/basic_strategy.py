# Rule-based basic-strategy chart. The caller supplies which optional actions
# are legal so unavailable doubles and surrenders can fall back safely.
from blackjack.actions import Action
from blackjack.cards import Card
from blackjack.hand import Hand


def _double_or(fallback, can_double):
    # Strategy charts use "double, otherwise X" entries.
    if can_double:
        return Action.DOUBLE
    return fallback


def _surrender_or(fallback, can_surrender):
    # Strategy charts likewise provide a fallback when surrender is disabled.
    if can_surrender:
        return Action.SURRENDER
    return fallback


def choose_action(
    hand,
    dealer_upcard,
    can_double,
    can_split,
    can_surrender,
    hit_soft_17=False,
):
    """Return the chart's basic-strategy action for a 4-8 deck game.

    ``hit_soft_17`` should be passed from ``Dealer.hit_soft_17``. A double
    falls back to hit or stand as indicated by the chart, and a surrender
    falls back to hit, stand, or split as indicated by the chart.
    """
    if not isinstance(hand, Hand):
        raise TypeError("Hand is invalid")
    if not isinstance(dealer_upcard, Card):
        raise TypeError("dealer's upcard is not a card")
    if hand.is_bust():
        raise ValueError("Hand is bust")
    if hand.is_blackjack():
        return Action.STAND

    player_total = hand.get_total()
    dealer_value = dealer_upcard.get_value()

    # Pairs are considered before the hard/soft total tables. The chart's
    # "Ph" entries are treated as split when splitting is available.
    if can_split and hand.can_split():
        pair_value = hand.cards[0].get_value()

        if pair_value == 11:
            return Action.SPLIT
        if pair_value == 10:
            return Action.STAND
        if pair_value == 9:
            if dealer_value in (2, 3, 4, 5, 6, 8, 9):
                return Action.SPLIT
            return Action.STAND
        if pair_value == 8:
            if hit_soft_17 and dealer_value == 11:
                return _surrender_or(Action.SPLIT, can_surrender)
            return Action.SPLIT
        if pair_value == 7:
            if dealer_value <= 7:
                return Action.SPLIT
            return Action.HIT
        if pair_value == 6:
            if dealer_value <= 6:
                return Action.SPLIT
            return Action.HIT
        if pair_value == 4:
            if dealer_value in (5, 6):
                return Action.SPLIT
            return Action.HIT
        if pair_value in (2, 3):
            if dealer_value <= 7:
                return Action.SPLIT
            return Action.HIT

    # Soft totals are evaluated separately because an ace still counts as 11.
    if hand.is_soft():
        if player_total >= 20:
            return Action.STAND

        if player_total == 19:
            if hit_soft_17 and dealer_value == 6:
                return _double_or(Action.STAND, can_double)
            return Action.STAND

        if player_total == 18:
            if hit_soft_17 and dealer_value in (2, 3, 4, 5, 6):
                return _double_or(Action.STAND, can_double)
            if not hit_soft_17 and dealer_value in (3, 4, 5, 6):
                return _double_or(Action.STAND, can_double)
            if dealer_value in (2, 7, 8):
                return Action.STAND
            return Action.HIT

        if player_total == 17:
            if dealer_value in (3, 4, 5, 6):
                return _double_or(Action.HIT, can_double)
            return Action.HIT

        if player_total in (15, 16):
            if dealer_value in (4, 5, 6):
                return _double_or(Action.HIT, can_double)
            return Action.HIT

        if player_total in (13, 14):
            if dealer_value in (5, 6):
                return _double_or(Action.HIT, can_double)
            return Action.HIT

        return Action.HIT

    # Hard totals.
    if player_total >= 18:
        return Action.STAND

    if player_total == 17:
        if hit_soft_17 and dealer_value == 11:
            return _surrender_or(Action.STAND, can_surrender)
        return Action.STAND

    if player_total == 16:
        if dealer_value in (9, 10, 11):
            return _surrender_or(Action.HIT, can_surrender)
        if dealer_value <= 6:
            return Action.STAND
        return Action.HIT

    if player_total == 15:
        surrender_values = (10, 11) if hit_soft_17 else (10,)
        if dealer_value in surrender_values:
            return _surrender_or(Action.HIT, can_surrender)
        if dealer_value <= 6:
            return Action.STAND
        return Action.HIT

    if player_total in (13, 14):
        if dealer_value <= 6:
            return Action.STAND
        return Action.HIT

    if player_total == 12:
        if dealer_value in (4, 5, 6):
            return Action.STAND
        return Action.HIT

    if player_total == 11:
        if hit_soft_17 or dealer_value != 11:
            return _double_or(Action.HIT, can_double)
        return Action.HIT

    if player_total == 10:
        if dealer_value <= 9:
            return _double_or(Action.HIT, can_double)
        return Action.HIT

    if player_total == 9:
        if dealer_value in (3, 4, 5, 6):
            return _double_or(Action.HIT, can_double)
        return Action.HIT

    return Action.HIT

