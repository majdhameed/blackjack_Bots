import random
from pathlib import Path

from agents.all_in_agent import AllInAgent
from agents.basic_strategy_agent import (
    BasicStrategyAgent,
)
from agents.neural_betting_agent import (
    NeuralBettingAgent,
)
from agents.percentage_bet_agent import (
    PercentageBetAgent,
)
from blackjack.player import Player
from ml.betting_network import BettingNetwork
from tournament.tournament import Tournament


STRATEGY_NAMES = (
    "neural",
    "minimum",
    "10_percent",
    "20_percent",
    "30_percent",
    "50_percent",
    "all_in",
)


def create_competitors(saved_network_path):
    network = BettingNetwork.load(
        saved_network_path
    )

    # Factories create a fresh agent for each tournament.
    return [
        (
            "neural",
            lambda: NeuralBettingAgent(network),
        ),
        (
            "minimum",
            lambda: BasicStrategyAgent(),
        ),
        (
            "10_percent",
            lambda: PercentageBetAgent(0.10),
        ),
        (
            "20_percent",
            lambda: PercentageBetAgent(0.20),
        ),
        (
            "30_percent",
            lambda: PercentageBetAgent(0.30),
        ),
        (
            "50_percent",
            lambda: PercentageBetAgent(0.50),
        ),
        (
            "all_in",
            lambda: AllInAgent(),
        ),
    ]


def create_statistics():
    return {
        strategy_name: {
            "win_credit": 0.0,
            "position_total": 0.0,
            "bankroll_total": 0.0,
            "bankruptcies": 0,
        }
        for strategy_name in STRATEGY_NAMES
    }


def record_win_credits(
    rankings,
    seat_names,
    statistics,
):
    highest_bankroll = rankings[0][1]

    winning_seats = [
        seat_index
        for seat_index, bankroll in rankings
        if bankroll == highest_bankroll
    ]

    shared_credit = 1.0 / len(winning_seats)

    for seat_index in winning_seats:
        strategy_name = seat_names[seat_index]

        statistics[strategy_name][
            "win_credit"
        ] += shared_credit


def record_bankrolls(
    rankings,
    seat_names,
    statistics,
):
    for seat_index, bankroll in rankings:
        strategy_name = seat_names[seat_index]

        statistics[strategy_name][
            "bankroll_total"
        ] += bankroll

        if bankroll <= 0:
            statistics[strategy_name][
                "bankruptcies"
            ] += 1


def record_finishing_positions(
    rankings,
    seat_names,
    statistics,
):
    ranking_index = 0

    while ranking_index < len(rankings):
        bankroll = rankings[ranking_index][1]

        tie_end = ranking_index

        while (
            tie_end < len(rankings)
            and rankings[tie_end][1] == bankroll
        ):
            tie_end += 1

        # Positions are one-indexed.
        occupied_positions = range(
            ranking_index + 1,
            tie_end + 1,
        )

        average_position = (
            sum(occupied_positions)
            / (tie_end - ranking_index)
        )

        for tied_index in range(
            ranking_index,
            tie_end,
        ):
            seat_index = rankings[tied_index][0]
            strategy_name = seat_names[
                seat_index
            ]

            statistics[strategy_name][
                "position_total"
            ] += average_position

        ranking_index = tie_end


def record_head_to_head(
    rankings,
    seat_names,
    head_to_head,
):
    bankroll_by_strategy = {}

    for seat_index, bankroll in rankings:
        strategy_name = seat_names[seat_index]
        bankroll_by_strategy[strategy_name] = (
            bankroll
        )

    neural_bankroll = bankroll_by_strategy[
        "neural"
    ]

    for strategy_name in STRATEGY_NAMES:
        if strategy_name == "neural":
            continue

        opponent_bankroll = bankroll_by_strategy[
            strategy_name
        ]

        if neural_bankroll > opponent_bankroll:
            head_to_head[strategy_name] += 1.0
        elif neural_bankroll == opponent_bankroll:
            head_to_head[strategy_name] += 0.5


def run_one_tournament(
    competitors,
    statistics,
    head_to_head,
    random_generator,
    starting_bankroll,
    rounds_per_tournament,
    decks,
    minimum_bet,
    hit_soft_17,
    max_hands,
):
    shuffled_competitors = competitors.copy()

    random_generator.shuffle(
        shuffled_competitors
    )

    seat_names = [
        strategy_name
        for strategy_name, _ in (
            shuffled_competitors
        )
    ]

    bots = [
        agent_factory()
        for _, agent_factory in (
            shuffled_competitors
        )
    ]

    players = [
        Player(starting_bankroll)
        for _ in shuffled_competitors
    ]

    tournament = Tournament(
        players=players,
        bots=bots,
        number_of_rounds=rounds_per_tournament,
        decks=decks,
        minimum_bet=minimum_bet,
        hit_soft_17=hit_soft_17,
        max_hands=max_hands,
    )

    rankings = tournament.play_tournament()

    record_win_credits(
        rankings,
        seat_names,
        statistics,
    )

    record_bankrolls(
        rankings,
        seat_names,
        statistics,
    )

    record_finishing_positions(
        rankings,
        seat_names,
        statistics,
    )

    record_head_to_head(
        rankings,
        seat_names,
        head_to_head,
    )

    return rankings


def evaluate_network(
    saved_network_path,
    number_of_tournaments,
    starting_bankroll=10_000,
    rounds_per_tournament=12,
    decks=6,
    minimum_bet=100,
    hit_soft_17=True,
    max_hands=4,
    seed=123,
):
    if (
        isinstance(number_of_tournaments, bool)
        or not isinstance(
            number_of_tournaments,
            int,
        )
    ):
        raise TypeError(
            "number_of_tournaments must be an integer"
        )

    if number_of_tournaments <= 0:
        raise ValueError(
            "number_of_tournaments must be positive"
        )

    competitors = create_competitors(
        saved_network_path
    )

    statistics = create_statistics()

    head_to_head = {
        strategy_name: 0.0
        for strategy_name in STRATEGY_NAMES
        if strategy_name != "neural"
    }

    random_generator = random.Random(seed)

    progress_interval = max(
        1,
        number_of_tournaments // 10,
    )

    for tournament_number in range(
        1,
        number_of_tournaments + 1,
    ):
        run_one_tournament(
            competitors=competitors,
            statistics=statistics,
            head_to_head=head_to_head,
            random_generator=random_generator,
            starting_bankroll=starting_bankroll,
            rounds_per_tournament=(
                rounds_per_tournament
            ),
            decks=decks,
            minimum_bet=minimum_bet,
            hit_soft_17=hit_soft_17,
            max_hands=max_hands,
        )

        if (
            tournament_number % progress_interval
            == 0
            or tournament_number
            == number_of_tournaments
        ):
            print(
                f"Completed {tournament_number:,} "
                f"of {number_of_tournaments:,} "
                "tournaments"
            )

    return {
        "number_of_tournaments": (
            number_of_tournaments
        ),
        "statistics": statistics,
        "head_to_head": head_to_head,
    }


def print_results(results):
    number_of_tournaments = results[
        "number_of_tournaments"
    ]

    statistics = results["statistics"]

    print()
    print("Betting strategy evaluation")
    print("---------------------------")
    print(
        f"Tournaments: "
        f"{number_of_tournaments:,}"
    )
    print()

    header = (
        f"{'Strategy':<14}"
        f"{'Win rate':>12}"
        f"{'Avg position':>16}"
        f"{'Avg bankroll':>18}"
        f"{'Bankruptcy':>14}"
    )

    print(header)
    print("-" * len(header))

    for strategy_name in STRATEGY_NAMES:
        strategy_statistics = statistics[
            strategy_name
        ]

        win_rate = (
            strategy_statistics["win_credit"]
            / number_of_tournaments
            * 100
        )

        average_position = (
            strategy_statistics["position_total"]
            / number_of_tournaments
        )

        average_bankroll = (
            strategy_statistics["bankroll_total"]
            / number_of_tournaments
        )

        bankruptcy_rate = (
            strategy_statistics["bankruptcies"]
            / number_of_tournaments
            * 100
        )

        print(
            f"{strategy_name:<14}"
            f"{win_rate:>11.2f}%"
            f"{average_position:>16.2f}"
            f"{average_bankroll:>18,.2f}"
            f"{bankruptcy_rate:>13.2f}%"
        )

    print()
    print("Neural head-to-head results")
    print("---------------------------")

    for strategy_name, credit in results[
        "head_to_head"
    ].items():
        rate = (
            credit
            / number_of_tournaments
            * 100
        )

        print(
            f"Neural vs {strategy_name:<10}: "
            f"{rate:6.2f}%"
        )


def ask_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))

            if value <= 0:
                raise ValueError

            return value

        except ValueError:
            print(
                "Please enter a positive integer."
            )


def main():
    saved_network_path = Path(
        "models/best_betting_network.npz"
    )

    if not saved_network_path.exists():
        raise FileNotFoundError(
            "Saved network was not found at "
            f"{saved_network_path}"
        )

    number_of_tournaments = ask_positive_integer(
        "Number of evaluation tournaments: "
    )

    results = evaluate_network(
        saved_network_path=saved_network_path,
        number_of_tournaments=(
            number_of_tournaments
        ),
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        seed=123,
    )

    print_results(results)

    results = evaluate_against_six_minimum(
        saved_network_path=saved_network_path,
        number_of_tournaments=(
            number_of_tournaments
        ),
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        seed=123,
    )

    print_six_minimum_results(results)

def evaluate_against_six_minimum(
    saved_network_path,
    number_of_tournaments,
    starting_bankroll=10_000,
    rounds_per_tournament=12,
    decks=6,
    minimum_bet=100,
    hit_soft_17=True,
    max_hands=4,
    seed=123,
):
    if (
        isinstance(number_of_tournaments, bool)
        or not isinstance(
            number_of_tournaments,
            int,
        )
    ):
        raise TypeError(
            "number_of_tournaments must be an integer"
        )

    if number_of_tournaments <= 0:
        raise ValueError(
            "number_of_tournaments must be positive"
        )

    network = BettingNetwork.load(
        saved_network_path
    )

    random_generator = random.Random(seed)

    first_place_credit = 0.0
    top_two_credit = 0.0
    position_total = 0.0
    bankroll_total = 0.0
    bankruptcies = 0

    progress_interval = max(
        1,
        number_of_tournaments // 10,
    )

    for tournament_number in range(
        1,
        number_of_tournaments + 1,
    ):
        competitors = [
            (
                "neural",
                NeuralBettingAgent(network),
            ),
            (
                "minimum_1",
                BasicStrategyAgent(),
            ),
            (
                "minimum_2",
                BasicStrategyAgent(),
            ),
            (
                "minimum_3",
                BasicStrategyAgent(),
            ),
            (
                "minimum_4",
                BasicStrategyAgent(),
            ),
            (
                "minimum_5",
                BasicStrategyAgent(),
            ),
            (
                "minimum_6",
                BasicStrategyAgent(),
            ),
        ]

        random_generator.shuffle(competitors)

        seat_names = [
            name
            for name, _ in competitors
        ]

        bots = [
            bot
            for _, bot in competitors
        ]

        players = [
            Player(starting_bankroll)
            for _ in competitors
        ]

        tournament = Tournament(
            players=players,
            bots=bots,
            number_of_rounds=(
                rounds_per_tournament
            ),
            decks=decks,
            minimum_bet=minimum_bet,
            hit_soft_17=hit_soft_17,
            max_hands=max_hands,
        )

        rankings = tournament.play_tournament()

        neural_seat_index = seat_names.index(
            "neural"
        )

        neural_ranking_index = None
        neural_bankroll = None

        for ranking_index, (
            seat_index,
            bankroll,
        ) in enumerate(rankings):
            if seat_index == neural_seat_index:
                neural_ranking_index = (
                    ranking_index
                )

                neural_bankroll = bankroll
                break

        if neural_ranking_index is None:
            raise RuntimeError(
                "Neural player was missing "
                "from tournament rankings"
            )

        # Find every player tied with the neural bot.
        tie_start = neural_ranking_index

        while (
            tie_start > 0
            and rankings[tie_start - 1][1]
            == neural_bankroll
        ):
            tie_start -= 1

        tie_end = neural_ranking_index + 1

        while (
            tie_end < len(rankings)
            and rankings[tie_end][1]
            == neural_bankroll
        ):
            tie_end += 1

        tie_size = tie_end - tie_start

        # Rankings use zero-based indexes here.
        # A tie occupying indexes 0 and 1 shares
        # first-place credit equally.
        if tie_start == 0:
            first_place_credit += (
                1.0 / tie_size
            )

        # Determine how many of the tied positions
        # fall within the top two.
        top_two_slots = max(
            0,
            min(tie_end, 2) - tie_start,
        )

        top_two_credit += (
            top_two_slots / tie_size
        )

        # Convert occupied indexes to one-based
        # finishing positions.
        occupied_positions = range(
            tie_start + 1,
            tie_end + 1,
        )

        average_position = (
            sum(occupied_positions)
            / tie_size
        )

        position_total += average_position
        bankroll_total += neural_bankroll

        if neural_bankroll <= 0:
            bankruptcies += 1

        if (
            tournament_number % progress_interval
            == 0
            or tournament_number
            == number_of_tournaments
        ):
            print(
                f"Completed {tournament_number:,} "
                f"of {number_of_tournaments:,} "
                "six-minimum tournaments"
            )

    return {
        "number_of_tournaments": (
            number_of_tournaments
        ),
        "first_place_rate": (
            first_place_credit
            / number_of_tournaments
            * 100
        ),
        "top_two_rate": (
            top_two_credit
            / number_of_tournaments
            * 100
        ),
        "average_position": (
            position_total
            / number_of_tournaments
        ),
        "average_bankroll": (
            bankroll_total
            / number_of_tournaments
        ),
        "bankruptcy_rate": (
            bankruptcies
            / number_of_tournaments
            * 100
        ),
    }

def print_six_minimum_results(results):
    print()
    print("Neural versus six minimum bettors")
    print("----------------------------------")

    print(
        f"Tournaments: "
        f"{results['number_of_tournaments']:,}"
    )

    print(
        f"First-place rate: "
        f"{results['first_place_rate']:.2f}%"
    )

    print(
        f"Top-two advancement rate: "
        f"{results['top_two_rate']:.2f}%"
    )

    print(
        f"Average position: "
        f"{results['average_position']:.2f}"
    )

    print(
        f"Average bankroll: "
        f"${results['average_bankroll']:,.2f}"
    )

    print(
        f"Bankruptcy rate: "
        f"{results['bankruptcy_rate']:.2f}%"
    )

    print()
    print("Equal-strategy reference values")
    print("-------------------------------")
    print("First-place rate: 14.29%")
    print("Top-two rate: 28.57%")
    print("Average position: 4.00")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation stopped.")

