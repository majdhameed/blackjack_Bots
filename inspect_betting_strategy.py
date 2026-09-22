from agents.neural_betting_agent import (
    NeuralBettingAgent,
)
from ml.betting_encoder import (
    encode_betting_observation,
)
from ml.betting_network import BettingNetwork
from tournament.observation import (
    BettingObservation,
)


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
    )


def create_scenarios():
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

    return [
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


def inspect_strategy(network_path):
    network = BettingNetwork.load(network_path)
    agent = NeuralBettingAgent(network)

    scenarios = create_scenarios()

    print()
    print("Learned neural betting strategy")
    print("-------------------------------")

    header = (
        f"{'Scenario':<48}"
        f"{'Output':>10}"
        f"{'Bet':>12}"
        f"{'Bankroll %':>14}"
    )

    print(header)
    print("-" * len(header))

    for scenario in scenarios:
        observation = scenario["observation"]

        features = encode_betting_observation(
            observation
        )

        raw_fraction = network.forward(features)

        final_bet = agent.choose_bet(
            observation
        )

        final_fraction = (
            final_bet / observation.bankroll
        )

        print(
            f"{scenario['name']:<48}"
            f"{raw_fraction:>10.4f}"
            f"{final_bet:>12,}"
            f"{final_fraction:>13.2%}"
        )


def main():
    inspect_strategy(
        "models/best_betting_network.npz"
    )


if __name__ == "__main__":
    main()