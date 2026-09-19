# Runs repeated seven-seat tournaments, assigning one betting strategy to each
# seat and aggregating both strategy-level and seat-level results.
import os
import random
from collections import Counter

from agents.all_in_agent import AllInAgent
from agents.basic_strategy_agent import (
    BasicStrategyAgent,
)
from agents.percentage_bet_agent import (
    PercentageBetAgent,
)
from blackjack.player import Player
from tournament.tournament import Tournament


STRATEGIES = (
    "minimum",
    "10_percent",
    "20_percent",
    "30_percent",
    "40_percent",
    "50_percent",
    "all_in",
)


DISPLAY_NAMES = {
    "minimum": "Minimum bet",
    "10_percent": "10% of bankroll",
    "20_percent": "20% of bankroll",
    "30_percent": "30% of bankroll",
    "40_percent": "40% of bankroll",
    "50_percent": "50% of bankroll",
    "all_in": "All-in",
}


def ask_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))

            if value <= 0:
                raise ValueError(
                    "Please enter a positive integer."
                )

            return value

        except ValueError as error:
            print(error)


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please enter 'y' or 'n'.")


def format_money(amount):
    if float(amount).is_integer():
        return f"${int(amount):,}"

    return f"${amount:,.2f}"


def clear_console():
    os.system(
        "cls" if os.name == "nt" else "clear"
    )


def create_bot(strategy):
    # Card-play behavior is shared; these agents differ only in bet sizing.
    if strategy == "minimum":
        return BasicStrategyAgent()

    if strategy == "10_percent":
        return PercentageBetAgent(0.10)

    if strategy == "20_percent":
        return PercentageBetAgent(0.20)

    if strategy == "30_percent":
        return PercentageBetAgent(0.30)

    if strategy == "40_percent":
        return PercentageBetAgent(0.40)

    if strategy == "50_percent":
        return PercentageBetAgent(0.50)

    if strategy == "all_in":
        return AllInAgent()

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


def main():
    number_players = len(STRATEGIES)

    print("Seven-strategy blackjack simulation")
    print("-----------------------------------")
    print("Strategies:")
    print("  1. Minimum bet")
    print("  2. 10% of bankroll")
    print("  3. 20% of bankroll")
    print("  4. 30% of bankroll")
    print("  5. 40% of bankroll")
    print("  6. 50% of bankroll")
    print("  7. All-in")
    print()

    starting_bankroll = ask_positive_integer(
        "Enter the starting bankroll: "
    )

    number_hands = ask_positive_integer(
        "Enter the number of hands per tournament: "
    )

    number_tournaments = ask_positive_integer(
        "Enter the number of tournaments to simulate: "
    )

    number_decks = ask_positive_integer(
        "Enter the number of decks: "
    )

    minimum_bet = ask_positive_integer(
        "Enter the minimum bet: "
    )

    hit_soft_17 = ask_yes_no(
        "Does the dealer hit soft 17? (y/n): "
    )

    if minimum_bet > starting_bankroll:
        print(
            "\nWarning: the minimum bet exceeds the "
            "starting bankroll."
        )

    strategy_entries = Counter()
    strategy_first_places = Counter()
    strategy_win_credits = Counter()
    strategy_bankroll_totals = Counter()
    strategy_bankruptcies = Counter()

    outcome_counts = Counter()

    tied_tournaments = 0
    total_rounds_played = 0

    progress_interval = max(
        1,
        number_tournaments // 10,
    )

    # Shuffle strategies between seats each tournament to reduce seat-order bias.
    for tournament_number in range(
        1,
        number_tournaments + 1,
    ):
        players = [
            Player(starting_bankroll)
            for _ in range(number_players)
        ]

        # Give every strategy a random seat.
        strategy_labels = list(STRATEGIES)
        random.shuffle(strategy_labels)

        bots = [
            create_bot(strategy)
            for strategy in strategy_labels
        ]

        for strategy in strategy_labels:
            strategy_entries[strategy] += 1

        tournament = Tournament(
            players=players,
            bots=bots,
            number_of_rounds=number_hands,
            decks=number_decks,
            minimum_bet=minimum_bet,
            hit_soft_17=hit_soft_17,
            max_hands=4,
        )

        rankings = tournament.play_tournament()

        total_rounds_played += len(
            tournament.round_history
        )

        highest_bankroll = rankings[0][1]

        winner_indices = [
            player_index
            for player_index, bankroll in rankings
            if bankroll == highest_bankroll
        ]

        if len(winner_indices) > 1:
            tied_tournaments += 1

        shared_credit = 1 / len(winner_indices)

        for winner_index in winner_indices:
            winner_strategy = strategy_labels[
                winner_index
            ]

            strategy_first_places[
                winner_strategy
            ] += 1

            strategy_win_credits[
                winner_strategy
            ] += shared_credit

        for player_index, player in enumerate(
            players
        ):
            strategy = strategy_labels[
                player_index
            ]

            strategy_bankroll_totals[
                strategy
            ] += player.bankroll

            if player.bankroll <= 0:
                strategy_bankruptcies[
                    strategy
                ] += 1

        for table_round in tournament.round_history:
            for player_outcomes in (
                table_round.outcomes
            ):
                outcome_counts.update(
                    player_outcomes
                )

        if (
            tournament_number % progress_interval == 0
            or tournament_number
            == number_tournaments
        ):
            print(
                f"Completed {tournament_number:,} "
                f"of {number_tournaments:,} "
                "tournaments"
            )

    clear_console()

    print("Seven-strategy simulation results")
    print("---------------------------------")
    print(
        f"Tournaments simulated: "
        f"{number_tournaments:,}"
    )
    print(
        f"Players per tournament: "
        f"{number_players}"
    )
    print(
        f"Requested hands per tournament: "
        f"{number_hands}"
    )
    print(
        f"Total hands played: "
        f"{total_rounds_played:,}"
    )
    print(
        f"Starting bankroll: "
        f"{format_money(starting_bankroll)}"
    )
    print(
        f"Minimum bet: "
        f"{format_money(minimum_bet)}"
    )
    print(
        f"Tied tournaments: "
        f"{tied_tournaments:,}"
    )

    print("\nResults by strategy")
    print("-------------------")

    result_rows = []

    for strategy in STRATEGIES:
        entries = strategy_entries[strategy]

        first_place_rate = (
            strategy_first_places[strategy]
            / entries
            * 100
        )

        credited_win_rate = (
            strategy_win_credits[strategy]
            / entries
            * 100
        )

        average_bankroll = (
            strategy_bankroll_totals[strategy]
            / entries
        )

        bankruptcy_rate = (
            strategy_bankruptcies[strategy]
            / entries
            * 100
        )

        result_rows.append(
            (
                strategy,
                credited_win_rate,
                first_place_rate,
                average_bankroll,
                bankruptcy_rate,
            )
        )

    # Display strongest tie-adjusted win rate first.
    result_rows.sort(
        key=lambda row: row[1],
        reverse=True,
    )

    for (
        strategy,
        credited_win_rate,
        first_place_rate,
        average_bankroll,
        bankruptcy_rate,
    ) in result_rows:
        print(
            f"\n{DISPLAY_NAMES[strategy]}"
        )
        print(
            f"First-place appearances: "
            f"{strategy_first_places[strategy]:,}"
        )
        print(
            f"First-place rate: "
            f"{first_place_rate:.2f}%"
        )
        print(
            f"Tie-adjusted win rate: "
            f"{credited_win_rate:.2f}%"
        )
        print(
            f"Average ending bankroll: "
            f"{format_money(average_bankroll)}"
        )
        print(
            f"Bankruptcy rate: "
            f"{bankruptcy_rate:.2f}%"
        )

    print("\nHand outcomes")
    print("-------------")

    for outcome, count in sorted(
        outcome_counts.items()
    ):
        print(f"{outcome}: {count:,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
