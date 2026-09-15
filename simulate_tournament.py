import os
from collections import Counter

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


def main():
    print("Blackjack tournament simulator")
    print("------------------------------")

    number_players = ask_player_count()

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
            "Players will be forced to wager their "
            "entire bankroll on the first hand."
        )

    win_credits = [
        0.0 for _ in range(number_players)
    ]

    first_place_appearances = [
        0 for _ in range(number_players)
    ]

    total_ending_bankrolls = [
        0.0 for _ in range(number_players)
    ]

    total_bankruptcies = 0
    tied_tournaments = 0
    outcome_counts = Counter()

    for tournament_number in range(
        1,
        number_tournaments + 1,
    ):
        players = [
            Player(starting_bankroll)
            for _ in range(number_players)
        ]

        bots = [
            BasicStrategyAgent()
            for _ in range(number_players)
        ]

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
            first_place_appearances[
                winner_index
            ] += 1

            win_credits[
                winner_index
            ] += shared_credit

        for player_index, player in enumerate(
            players
        ):
            total_ending_bankrolls[
                player_index
            ] += player.bankroll

            if player.bankroll <= 0:
                total_bankruptcies += 1

        for table_round in tournament.round_history:
            for player_outcomes in (
                table_round.outcomes
            ):
                outcome_counts.update(
                    player_outcomes
                )

        progress_interval = max(
            1,
            number_tournaments // 10,
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

    print("Tournament simulation results")
    print("-----------------------------")
    print(
        f"Tournaments simulated: "
        f"{number_tournaments:,}"
    )
    print(
        f"Players per tournament: "
        f"{number_players:,}"
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

    print("\nResults by seat")
    print("---------------")

    for player_index in range(number_players):
        average_bankroll = (
            total_ending_bankrolls[player_index]
            / number_tournaments
        )

        first_place_rate = (
            first_place_appearances[player_index]
            / number_tournaments
            * 100
        )

        credited_win_rate = (
            win_credits[player_index]
            / number_tournaments
            * 100
        )

        print(f"\nPlayer {player_index + 1}")
        print(
            f"First-place appearances: "
            f"{first_place_appearances[player_index]:,}"
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