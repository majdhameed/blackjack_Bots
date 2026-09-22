from agents.neural_betting_agent import NeuralBettingAgent
from blackjack.player import Player
from tournament.tournament import Tournament
from agents.all_in_agent import AllInAgent
from agents.basic_strategy_agent import (
    BasicStrategyAgent,
)
from agents.percentage_bet_agent import (
    PercentageBetAgent,
)

class Trainer:
    def __init__(self, population, starting_bankroll, rounds_per_tournament, decks, minimum_bet, hit_soft_17, max_hands):
        self.population = population
        self.starting_bankroll = starting_bankroll
        self.rounds_per_tournament = (
            rounds_per_tournament
        )
        self.decks = decks
        self.minimum_bet = minimum_bet
        self.hit_soft_17 = hit_soft_17
        self.max_hands = max_hands

    def evaluate_group(self, network_indices):
        if len(network_indices) != 7:
            raise ValueError("There must be 7 players per table")

        players = [Player(self.starting_bankroll) for _ in network_indices]
        agents = []
        
        for network_index in network_indices:
            network = self.population.networks[network_index]

            agent = NeuralBettingAgent(network)

            agents.append(agent)


        tournament = Tournament(
            players=players,
            bots=agents,
            number_of_rounds=self.rounds_per_tournament,
            decks=self.decks,
            minimum_bet=self.minimum_bet,
            hit_soft_17=self.hit_soft_17,
            max_hands=self.max_hands
        )

        rankings = tournament.play_tournament()


        self.award_fitness(network_indices, rankings)

        return rankings

    def award_fitness(
        self,
        network_indices,
        rankings,
    ):
        position = 0

        while position < len(rankings):
            bankroll = rankings[position][1]

            tie_end = position

            while (
                tie_end < len(rankings)
                and rankings[tie_end][1]
                == bankroll
            ):
                tie_end += 1

            tie_size = tie_end - position

            # Rankings use indexes 0 and 1 for the
            # two advancement positions.
            top_two_slots = max(
                0,
                min(tie_end, 2) - position,
            )

            shared_credit = (
                top_two_slots / tie_size
            )

            for ranking_index in range(
                position,
                tie_end,
            ):
                seat_index = rankings[
                    ranking_index
                ][0]

                population_index = (
                    network_indices[seat_index]
                )

                self.population.add_fitness(
                    population_index,
                    shared_credit,
                )

            position = tie_end
    def evaluate_generation(
        self,
        tournaments_per_network,
        baseline_tournaments_per_network=0,
    ):
        if (
            isinstance(tournaments_per_network, bool)
            or not isinstance(
                tournaments_per_network,
                int,
            )
        ):
            raise TypeError(
                "tournaments_per_network must be an integer"
            )

        if tournaments_per_network <= 0:
            raise ValueError(
                "tournaments_per_network must be positive"
            )

        if (
            isinstance(
                baseline_tournaments_per_network,
                bool,
            )
            or not isinstance(
                baseline_tournaments_per_network,
                int,
            )
        ):
            raise TypeError(
                "baseline_tournaments_per_network "
                "must be an integer"
            )

        if baseline_tournaments_per_network < 0:
            raise ValueError(
                "baseline_tournaments_per_network "
                "cannot be negative"
            )

        self.population.reset_fitness()

        for _ in range(tournaments_per_network):
            network_indices = list(
                range(
                    self.population.population_size
                )
            )

            self.population.random_generator.shuffle(
                network_indices
            )

            for start_index in range(
                0,
                len(network_indices),
                7,
            ):
                group = network_indices[
                    start_index:start_index + 7
                ]

                self.evaluate_group(group)

        if baseline_tournaments_per_network > 0:
            self.evaluate_all_against_baselines(
                baseline_tournaments_per_network
            )

        return self.population.fitness_scores.copy()


    def train_generation(
        self,
        tournaments_per_network,
        baseline_tournaments_per_network=0,
    ):
        evaluated_generation = (
            self.population.generation_number
        )

        fitness_scores = self.evaluate_generation(
            tournaments_per_network=(
                tournaments_per_network
            ),
            baseline_tournaments_per_network=(
                baseline_tournaments_per_network
            ),
        )

        ranked_indices = (
            self.population.get_ranked_indices()
        )

        best_network_index = ranked_indices[0]

        best_fitness = float(
            fitness_scores[best_network_index]
        )

        average_fitness = float(
            fitness_scores.mean()
        )

        best_network = self.population.networks[
            best_network_index
        ].clone()

        self.population.create_next_generation()

        return {
            "generation": evaluated_generation,
            "best_network_index": (
                best_network_index
            ),
            "best_fitness": best_fitness,
            "average_fitness": average_fitness,
            "best_network": best_network,
        }

    def evaluate_network_against_baselines(
        self,
        network_index,
        number_of_tournaments,
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

        if number_of_tournaments < 0:
            raise ValueError(
                "number_of_tournaments cannot be negative"
            )

        if (
            isinstance(network_index, bool)
            or not isinstance(network_index, int)
        ):
            raise TypeError(
                "network_index must be an integer"
            )

        if not (
            0
            <= network_index
            < self.population.population_size
        ):
            raise IndexError(
                "network_index is outside the population"
            )

        network = self.population.networks[
            network_index
        ]

        total_fitness = 0.0

        for _ in range(number_of_tournaments):
            competitors = [
                (
                    "neural",
                    NeuralBettingAgent(network),
                ),
                (
                    "minimum",
                    BasicStrategyAgent(),
                ),
                (
                    "minimum",
                    BasicStrategyAgent(),
                ),
                (
                    "minimum",
                    BasicStrategyAgent(),
                ),
                (
                    "minimum",
                    BasicStrategyAgent(),
                ),
                (
                    "minimum",
                    BasicStrategyAgent(),
                ),
                (
                    "minimum",
                    BasicStrategyAgent(),
                ),
            ]

            seat_order = (
                self.population.random_generator.permutation(
                    len(competitors)
                )
            )

            shuffled_competitors = [
                competitors[int(index)]
                for index in seat_order
            ]

            seat_names = [
                strategy_name
                for strategy_name, _ in (
                    shuffled_competitors
                )
            ]

            bots = [
                bot
                for _, bot in shuffled_competitors
            ]

            players = [
                Player(self.starting_bankroll)
                for _ in shuffled_competitors
            ]

            tournament = Tournament(
                players=players,
                bots=bots,
                number_of_rounds=(
                    self.rounds_per_tournament
                ),
                decks=self.decks,
                minimum_bet=self.minimum_bet,
                hit_soft_17=self.hit_soft_17,
                max_hands=self.max_hands,
            )

            rankings = tournament.play_tournament()

            neural_seat_index = seat_names.index(
                "neural"
            )

            neural_bankroll = None

            for seat_index, bankroll in rankings:
                if seat_index == neural_seat_index:
                    neural_bankroll = bankroll
                    break

            if neural_bankroll is None:
                raise RuntimeError(
                    "Neural player was missing "
                    "from tournament rankings"
                )

            neural_ranking_index = None

            for ranking_index, (
                seat_index,
                bankroll,
            ) in enumerate(rankings):
                if seat_index == neural_seat_index:
                    neural_ranking_index = (
                        ranking_index
                    )
                    break

            if neural_ranking_index is None:
                raise RuntimeError(
                    "Neural player was missing "
                    "from tournament rankings"
                )

            neural_bankroll = rankings[
                neural_ranking_index
            ][1]

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

            top_two_slots = max(
                0,
                min(tie_end, 2) - tie_start,
            )

            advancement_credit = (
                top_two_slots / tie_size
            )

            total_fitness += advancement_credit

        self.population.add_fitness(
            network_index,
            total_fitness,
        )

        return total_fitness
    def evaluate_all_against_baselines(
        self,
        tournaments_per_network,
    ):
        baseline_scores = []

        for network_index in range(
            self.population.population_size
        ):
            score = (
                self.evaluate_network_against_baselines(
                    network_index,
                    tournaments_per_network,
                )
            )

            baseline_scores.append(score)

        return baseline_scores