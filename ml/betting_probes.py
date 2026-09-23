from tournament.observation import BettingObservation


TOTAL_ROUNDS = 12
MINIMUM_BET = 100


def make_observation(
    round_number,
    bankrolls,
    player_index,
    betting_position,
    current_bets,
    bets_placed,
    true_count=0.0,
    previous_bet=0.0,
    previous_bankroll_change=0.0,
    previous_result=0.0,
    consecutive_losses=0,
    has_previous_round=False,
):
    return BettingObservation(
        round_number=round_number,
        total_rounds=TOTAL_ROUNDS,
        rounds_remaining=(
            TOTAL_ROUNDS - round_number
        ),
        player_index=player_index,
        round_player_index=player_index,
        betting_position=betting_position,
        minimum_bet=MINIMUM_BET,
        bankroll=bankrolls[player_index],
        bankrolls=tuple(bankrolls),
        active_players=(
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ),
        current_bets=tuple(current_bets),
        bets_placed=tuple(bets_placed),
        betting_order=(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ),
        card_value_counts=(
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
        ),
        cards_seen=0,
        running_count=0,
        true_count=true_count,
        cards_remaining=312,
        decks_remaining=6.0,
        shoe_penetration=0.0,
        previous_bet=previous_bet,
        previous_bankroll_change=(
            previous_bankroll_change
        ),
        previous_result=previous_result,
        consecutive_losses=consecutive_losses,
        has_previous_round=has_previous_round,
    )

def create_betting_probes():
    tied_bankrolls = [
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
    ]

    leading_bankrolls = [
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        12_000,
    ]

    behind_bankrolls = [
        11_000,
        10_500,
        10_000,
        10_000,
        10_000,
        10_000,
        8_000,
    ]

    slightly_behind_bankrolls = [
        10_500,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
        10_000,
    ]

    scenarios = [
        {
            "name": "Round 1, tied, betting first",
            "observation": make_observation(
                round_number=1,
                bankrolls=tied_bankrolls,
                player_index=0,
                betting_position=0,
                current_bets=[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                bets_placed=[
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ),
        },
        {
            "name": "Round 1, tied, betting last",
            "observation": make_observation(
                round_number=1,
                bankrolls=tied_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Round 4, three leaders established",
            "observation": make_observation(
                round_number=4,
                bankrolls=[
                    16_000,
                    14_500,
                    13_000,
                    10_000,
                    9_500,
                    9_200,
                    9_000,
                ],
                player_index=6,
                betting_position=6,
                current_bets=[
                    500,
                    300,
                    200,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Round 6, leading, betting first",
            "observation": make_observation(
                round_number=6,
                bankrolls=[
                    12_000,
                    10_000,
                    10_000,
                    10_000,
                    10_000,
                    10_000,
                    10_000,
                ],
                player_index=0,
                betting_position=0,
                current_bets=[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                bets_placed=[
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ),
        },
        {
            "name": "Round 6, behind, betting last",
            "observation": make_observation(
                round_number=6,
                bankrolls=behind_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Round 11, leading, betting last",
            "observation": make_observation(
                round_number=11,
                bankrolls=leading_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Round 11, far behind, betting last",
            "observation": make_observation(
                round_number=11,
                bankrolls=behind_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Final round, tied, betting first",
            "observation": make_observation(
                round_number=12,
                bankrolls=tied_bankrolls,
                player_index=0,
                betting_position=0,
                current_bets=[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                bets_placed=[
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ),
        },
        {
            "name": "Final round, tied, betting last",
            "observation": make_observation(
                round_number=12,
                bankrolls=tied_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": (
                "Final round, slightly behind, "
                "betting last"
            ),
            "observation": make_observation(
                round_number=12,
                bankrolls=(
                    slightly_behind_bankrolls
                ),
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": (
                "Final round, far behind, "
                "betting last"
            ),
            "observation": make_observation(
                round_number=12,
                bankrolls=behind_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": (
                "Final round, behind, opponents "
                "bet large"
            ),
            "observation": make_observation(
                round_number=12,
                bankrolls=behind_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    2_000,
                    1_500,
                    1_000,
                    500,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
        {
            "name": "Final round, leading, betting last",
            "observation": make_observation(
                round_number=12,
                bankrolls=leading_bankrolls,
                player_index=6,
                betting_position=6,
                current_bets=[
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    0,
                ],
                bets_placed=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            ),
        },
    ]

    scenarios.append(
        {
            "name": "Round 4, narrow 5% lead",
            "observation": make_observation(
                round_number=4,
                bankrolls=[
                    10_500,
                    10_000,
                    9_900,
                    9_800,
                    9_700,
                    9_600,
                    9_500,
                ],
                player_index=0,
                betting_position=0,
                current_bets=[0] * 7,
                bets_placed=[False] * 7,
            ),
        }
    )

    gap_checks = (
        (6, 5),
        (6, 10),
        (6, 20),
        (11, 3),
        (11, 5),
        (11, 8),
        (12, 3),
        (12, 8),
    )

    for round_number, gap_percent in gap_checks:
        focal_bankroll = 10_000
        second_bankroll = round(
            focal_bankroll
            * (1 + gap_percent / 100)
        )
        leader_bankroll = max(
            second_bankroll + 500,
            11_500,
        )
        scenarios.append(
            {
                "name": (
                    f"Round {round_number}, 3rd, "
                    f"{gap_percent}% gap to second"
                ),
                "observation": make_observation(
                    round_number=round_number,
                    bankrolls=[
                        leader_bankroll,
                        second_bankroll,
                        9_800,
                        9_600,
                        9_400,
                        9_200,
                        focal_bankroll,
                    ],
                    player_index=6,
                    betting_position=6,
                    current_bets=[
                        100,
                        100,
                        100,
                        100,
                        100,
                        100,
                        0,
                    ],
                    bets_placed=[
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                    ],
                ),
            }
        )

    for loss_count, previous_bet in (
        (1, 500),
        (2, 1_000),
    ):
        scenarios.append(
            {
                "name": (
                    f"Round 6, after {loss_count} "
                    f"loss{'es' if loss_count > 1 else ''}"
                ),
                "observation": make_observation(
                    round_number=6,
                    bankrolls=[
                        10_600,
                        10_100,
                        9_900,
                        9_800,
                        9_700,
                        9_600,
                        9_500,
                    ],
                    player_index=6,
                    betting_position=6,
                    current_bets=[100] * 6 + [0],
                    bets_placed=[True] * 6 + [False],
                    previous_bet=previous_bet,
                    previous_bankroll_change=-previous_bet,
                    previous_result=-1.0,
                    consecutive_losses=loss_count,
                    has_previous_round=True,
                ),
            }
        )

    return scenarios