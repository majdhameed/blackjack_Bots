import os
import random
from collections import Counter

from agents.all_in_agent import AllInAgent
from agents.basic_strategy_agent import (
    BasicStrategyAgent,
)
from blackjack.player import Player
from tournament.tournament import Tournament


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


def ask_integer_in_range(
    prompt,
    minimum,
    maximum,
):
    while True:
        try:
            value = int(input(prompt))

            if value < minimum or value > maximum:
                raise ValueError(
                    f"Please enter a value from "
                    f"{minimum} to {maximum}."
                )

            return value

        except ValueError as error:
            print(error)


def ask_player_count():
    while True:
        number_players = ask_positive_integer(
            "Enter the number of players: "
        )

        if number_players >= 2:
            return number_players

        print(
            "A tournament requires at least two players."
        )


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
    if strategy == "basic":
        return BasicStrategyAgent()

    if strategy == "all_in":
        return AllInAgent()

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


def main():
    print("Mixed blackjack tournament simulator")
    print("------------------------------------")

    number_players = ask_player_count()

    number_all_in = ask_integer_in_range(
        "How many players should use the "
        "all-in strategy? ",
        0,
        number_players,
    )

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
        print(
            "Players will wager their entire bankroll "
            "on the first hand."
        )

    seat_win_credits = [
        0.0 for _ in range(number_players)
    ]

    seat_first_places = [
        0 for _ in range(number_players)
    ]

    seat_bankroll_totals = [
        0.0 for _ in range(number_players)
    ]

    strategy_entries = Counter()
    strategy_first_places = Counter()
    strategy_win_credits = Counter()
    strategy_bankroll_totals = Counter()
    strategy_bankruptcies = Counter()

    total_bankruptcies = 0
    tied_tournaments = 0
    outcome_counts = Counter()

    progress_interval = max(
        1,
        number_tournaments // 10,
    )

    for tournament_number in range(
        1,
        number_tournaments + 1,
    ):
        players = [
            Player(starting_bankroll)
            for _ in range(number_players)
        ]

        strategy_labels = (
            ["all_in"] * number_all_in
            + ["basic"]
            * (number_players - number_all_in)
        )

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
            strategy = strategy_labels[
                winner_index
            ]

            seat_first_places[
                winner_index
            ] += 1

            seat_win_credits[
                winner_index
            ] += shared_credit

            strategy_first_places[
                strategy
            ] += 1

            strategy_win_credits[
                strategy
            ] += shared_credit

        for player_index, player in enumerate(
            players
        ):
            strategy = strategy_labels[
                player_index
            ]

            seat_bankroll_totals[
                player_index
            ] += player.bankroll

            strategy_bankroll_totals[
                strategy
            ] += player.bankroll

            if player.bankroll <= 0:
                total_bankruptcies += 1
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

    print("Mixed tournament simulation results")
    print("-----------------------------------")
    print(
        f"Tournaments simulated: "
        f"{number_tournaments:,}"
    )
    print(
        f"Players per tournament: "
        f"{number_players:,}"
    )
    print(
        f"Basic-strategy bettors: "
        f"{number_players - number_all_in}"
    )
    print(
        f"All-in bettors: {number_all_in}"
    )
    print(
        f"Hands per tournament: "
        f"{number_hands:,}"
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
    print(
        f"Total bankruptcies: "
        f"{total_bankruptcies:,}"
    )

    print("\nResults by strategy")
    print("-------------------")

    for strategy in ("basic", "all_in"):
        entries = strategy_entries[strategy]

        if entries == 0:
            continue

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

        display_name = {
            "basic": "Minimum-bet basic strategy",
            "all_in": "All-in basic strategy",
        }[strategy]

        print(f"\n{display_name}")
        print(f"Entries: {entries:,}")
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

    print("\nResults by seat")
    print("---------------")

    for player_index in range(number_players):
        average_bankroll = (
            seat_bankroll_totals[player_index]
            / number_tournaments
        )

        first_place_rate = (
            seat_first_places[player_index]
            / number_tournaments
            * 100
        )

        credited_win_rate = (
            seat_win_credits[player_index]
            / number_tournaments
            * 100
        )

        print(f"\nSeat {player_index + 1}")
        print(
            f"First-place appearances: "
            f"{seat_first_places[player_index]:,}"
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