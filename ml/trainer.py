import copy
import math
import random
from pathlib import Path
import statistics

import numpy as np

from agents.neural_betting_agent import NeuralBettingAgent
from blackjack.player import Player
from tournament.tournament import Tournament
from agents.all_in_agent import AllInAgent
from agents.basic_strategy_agent import (
    BasicStrategyAgent,
)
from agents.chasing_agent import ChasingAgent
from agents.count_aware_agent import CountAwareAgent
from agents.controlled_lead_martingale_agent import (
    ControlledLeadMartingaleAgent,
)
from agents.early_lead_agent import EarlyLeadAgent
from agents.lead_protection_agent import (
    LeadProtectionAgent,
)
from agents.human_behavior_agent import HumanBehaviorAgent
from agents.percentage_bet_agent import (
    PercentageBetAgent,
)
from agents.unpredictable_betting_agent import (
    UnpredictableBettingAgent,
)

from ml.strategy_diversity import (
    measure_strategy_diversity
)


FIRST_PLACE_FITNESS = 1.05
SECOND_PLACE_FITNESS = 1.0


def advancement_fitness_for_tie(
    tie_start,
    tie_end,
):
    position_fitness = (
        FIRST_PLACE_FITNESS,
        SECOND_PLACE_FITNESS,
    )

    occupied_fitness = [
        (
            position_fitness[position]
            if position < len(position_fitness)
            else 0.0
        )
        for position in range(tie_start, tie_end)
    ]

    return sum(occupied_fitness) / len(occupied_fitness)


def training_progress_fitness(
    starting_bankrolls,
    final_bankrolls,
    seat_index,
):
    """Small dense reward used only while evolving networks."""
    starting_bankroll = starting_bankrolls[seat_index]
    final_bankroll = final_bankrolls[seat_index]
    starting_rank = 1 + sum(
        bankroll > starting_bankroll
        for bankroll in starting_bankrolls
    )
    final_rank = 1 + sum(
        bankroll > final_bankroll
        for bankroll in final_bankrolls
    )
    rank_credit = min(
        0.20,
        0.05 * max(0, starting_rank - final_rank),
    )

    starting_second = sorted(
        starting_bankrolls,
        reverse=True,
    )[1]
    final_second = sorted(
        final_bankrolls,
        reverse=True,
    )[1]
    starting_gap = max(
        0,
        starting_second - starting_bankroll,
    )
    final_gap = max(
        0,
        final_second - final_bankroll,
    )

    gap_credit = 0.0
    if starting_gap > 0:
        closure_fraction = max(
            0.0,
            min(
                1.0,
                (starting_gap - final_gap)
                / starting_gap,
            ),
        )
        gap_credit = 0.25 * closure_fraction

    def top_two_safety_margin(bankrolls):
        if len(bankrolls) < 3:
            return 0.0
        third_bankroll = sorted(
            bankrolls,
            reverse=True,
        )[2]
        return max(
            0.0,
            bankrolls[seat_index] - third_bankroll,
        )

    def lead_margin(bankrolls):
        opponent_bankrolls = [
            bankroll
            for index, bankroll in enumerate(bankrolls)
            if index != seat_index
        ]
        return max(
            0.0,
            bankrolls[seat_index]
            - max(opponent_bankrolls),
        )

    margin_scale = max(
        starting_bankroll * 0.05,
        1.0,
    )
    safety_improvement = max(
        0.0,
        top_two_safety_margin(final_bankrolls)
        - top_two_safety_margin(starting_bankrolls),
    )
    lead_improvement = max(
        0.0,
        lead_margin(final_bankrolls)
        - lead_margin(starting_bankrolls),
    )
    safety_credit = 0.15 * min(
        1.0,
        safety_improvement / margin_scale,
    )
    lead_credit = 0.10 * min(
        1.0,
        lead_improvement / margin_scale,
    )

    return (
        rank_credit
        + gap_credit
        + safety_credit
        + lead_credit
    )

class Trainer:
    def __init__(
        self,
        population,
        starting_bankroll,
        rounds_per_tournament,
        decks,
        minimum_bet,
        hit_soft_17,
        max_hands,
        randomize_training_rounds=False,
        league_directory=None,
        maximum_league_size=8,
    ):
        self.population = population
        self.starting_bankroll = starting_bankroll
        self.rounds_per_tournament = (
            rounds_per_tournament
        )
        self.decks = decks
        self.minimum_bet = minimum_bet
        self.hit_soft_17 = hit_soft_17
        self.max_hands = max_hands
        self.randomize_training_rounds = (
            randomize_training_rounds
        )
        self.maximum_league_size = maximum_league_size
        self.league_directory = (
            Path(league_directory)
            if league_directory is not None
            else None
        )
        self.champion_league = []
        self.league_best_score = float("-inf")
        self._load_champion_league()

    def _load_champion_league(self):
        if (
            self.league_directory is None
            or not self.league_directory.exists()
        ):
            return

        from ml.betting_network import BettingNetwork

        league_paths = sorted(
            self.league_directory.glob("champion_*.npz")
        )[-self.maximum_league_size:]
        for league_path in league_paths:
            self.champion_league.append(
                BettingNetwork.load(league_path)
            )
            try:
                name_parts = league_path.stem.split("_")
                archived_score = (
                    float(name_parts[2])
                    if len(name_parts) >= 3
                    else 0.0
                )
            except (ValueError, IndexError):
                archived_score = 0.0
            self.league_best_score = max(
                self.league_best_score,
                archived_score,
            )

    def maybe_promote_to_league(
        self,
        network,
        selection_fitness,
        normal_fitness,
        mid_fitness,
        late_fitness,
        holdout_fitness,
        minimum_holdout_fitness,
        generation,
    ):
        """Archive only broad performers, not one-table lucky winners."""
        required_scores = (
            normal_fitness,
            mid_fitness,
            late_fitness,
            holdout_fitness,
            minimum_holdout_fitness,
        )
        if any(score is None or score <= 0 for score in required_scores):
            return False
        if holdout_fitness < 0.25 or minimum_holdout_fitness < 0.25:
            return False
        if selection_fitness < self.league_best_score + 0.01:
            return False

        champion = network.clone()
        self.champion_league.append(champion)
        self.champion_league = self.champion_league[
            -self.maximum_league_size:
        ]
        self.league_best_score = selection_fitness

        if self.league_directory is not None:
            self.league_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            champion.save(
                self.league_directory
                / (
                    f"champion_{generation:04d}_"
                    f"{selection_fitness:.4f}.npz"
                )
            )

        return True

    def select_training_table_kind(self):
        """Sample the requested 40/25/20/10/5 training mixture."""
        roll = self.population.random_generator.random()
        if roll < 0.40:
            return "human"
        if roll < 0.65:
            return "proven"
        if roll < 0.85:
            return "league"
        if roll < 0.95:
            return "adversarial"
        return "minimum"

    def training_round_count(self):
        if not self.randomize_training_rounds:
            return self.rounds_per_tournament

        minimum_rounds = max(
            2,
            self.rounds_per_tournament * 2 // 3,
        )

        maximum_rounds = max(
            minimum_rounds,
            self.rounds_per_tournament * 4 // 3,
        )

        return int(
            self.population.random_generator.integers(
                minimum_rounds,
                maximum_rounds + 1,
            )
        )

    def training_shaping_multiplier(self):
        """Fade dense rewards out; final generations use advancement only."""
        return max(
            0.0,
            1.0
            - self.population.generation_number / 40,
        )

    def create_training_baseline_competitors(
        self,
        network,
        table_kind=None,
    ):
        """Build a table from the human/strategy/champion league mixture."""
        if table_kind is None:
            table_kind = self.select_training_table_kind()

        if table_kind not in {
            "adaptive",
            "human",
            "extreme",
            "proven",
            "league",
            "adversarial",
            "minimum",
        }:
            raise ValueError(
                "table_kind must be adaptive, human, extreme, proven, "
                "league, adversarial, or minimum"
            )

        if table_kind == "proven":
            table_kind = "adaptive"
        elif table_kind == "adversarial":
            table_kind = "extreme"

        competitors = [
            ("neural", NeuralBettingAgent(network))
        ]

        if table_kind == "minimum":
            competitors.extend(
                (
                    f"minimum_{index}",
                    BasicStrategyAgent(),
                )
                for index in range(1, 7)
            )
            return competitors

        if table_kind == "league":
            if not self.champion_league:
                return self.create_training_baseline_competitors(
                    network,
                    table_kind="adaptive",
                )

            champion_count = min(
                3,
                len(self.champion_league),
            )
            champion_indices = (
                self.population.random_generator.choice(
                    len(self.champion_league),
                    size=champion_count,
                    replace=False,
                )
            )
            for opponent_number, champion_index in enumerate(
                champion_indices,
                start=1,
            ):
                competitors.append(
                    (
                        f"league_champion_{opponent_number}",
                        NeuralBettingAgent(
                            self.champion_league[
                                int(champion_index)
                            ].clone()
                        ),
                    )
                )

            league_fillers = (
                (
                    "controlled_lead_martingale",
                    ControlledLeadMartingaleAgent,
                ),
                ("lead_protector", LeadProtectionAgent),
                (
                    "league_chaser",
                    lambda: ChasingAgent(0.45, 0.50),
                ),
                (
                    "league_human",
                    lambda: self.create_human_behavior_agent(
                        base_fraction=0.05,
                        maximum_fraction=0.30,
                    ),
                ),
                (
                    "league_early_lead",
                    lambda: EarlyLeadAgent(0.30, 0.50),
                ),
                ("league_count_aware", CountAwareAgent),
            )
            for name, factory in league_fillers:
                if len(competitors) == 7:
                    break
                competitors.append((name, factory()))
            return competitors

        if table_kind == "adaptive":
            opponent_factories = [
                (
                    "conservative_chaser",
                    lambda: ChasingAgent(0.65, 0.30),
                ),
                (
                    "aggressive_chaser",
                    lambda: ChasingAgent(0.35, 0.65),
                ),
                (
                    "controlled_lead_martingale",
                    ControlledLeadMartingaleAgent,
                ),
                ("lead_protector", LeadProtectionAgent),
                (
                    "half_bankroll_leader",
                    lambda: EarlyLeadAgent(0.25, 0.50),
                ),
                (
                    "adaptive_human",
                    lambda: self.create_human_behavior_agent(
                        base_fraction=0.05,
                        maximum_fraction=0.25,
                        loss_multiplier=1.5,
                        win_multiplier=1.25,
                    ),
                ),
            ]
        elif table_kind == "human":
            opponent_factories = [
                (
                    "balanced_chaser",
                    lambda: ChasingAgent(0.50, 0.45),
                ),
                (
                    "human_cautious",
                    lambda: self.create_human_behavior_agent(
                        base_fraction=0.03,
                        maximum_fraction=0.15,
                        loss_multiplier=1.25,
                        win_multiplier=1.10,
                    ),
                ),
                (
                    "human_balanced",
                    lambda: self.create_human_behavior_agent(
                        base_fraction=0.05,
                        maximum_fraction=0.25,
                        loss_multiplier=1.5,
                        win_multiplier=1.25,
                    ),
                ),
                (
                    "human_bold",
                    lambda: self.create_human_behavior_agent(
                        base_fraction=0.08,
                        maximum_fraction=0.40,
                        loss_multiplier=1.75,
                        win_multiplier=1.5,
                        impulse_probability=0.15,
                    ),
                ),
                (
                    "human_unpredictable",
                    lambda: self.create_unpredictable_agent(
                        0.40,
                        0.25,
                    ),
                ),
                (
                    "controlled_lead_martingale",
                    ControlledLeadMartingaleAgent,
                ),
            ]
        else:
            opponent_factories = [
                (
                    "five_percent",
                    lambda: PercentageBetAgent(0.05),
                ),
                (
                    "fifteen_percent",
                    lambda: PercentageBetAgent(0.15),
                ),
                ("all_in", AllInAgent),
                (
                    "extreme_human",
                    lambda: self.create_unpredictable_agent(
                        0.20,
                        0.50,
                    ),
                ),
                (
                    "aggressive_early_lead",
                    lambda: EarlyLeadAgent(0.60, 0.30),
                ),
                (
                    "desperate_chaser",
                    lambda: ChasingAgent(0.25, 0.80),
                ),
            ]

        competitors.extend(
            (
                strategy_name,
                agent_factory(),
            )
            for strategy_name, agent_factory in (
                opponent_factories
            )
        )

        return competitors

    def create_human_behavior_agent(
        self,
        **kwargs,
    ):
        seed = int(
            self.population.random_generator.integers(
                0,
                2**32,
            )
        )
        return HumanBehaviorAgent(seed=seed, **kwargs)

    def create_fixed_benchmark_competitors(
        self,
        network,
        lineup_kind,
    ):
        """Build one of several strict, repeatable holdout lineups."""
        competitors = [
            ("neural", NeuralBettingAgent(network))
        ]

        if lineup_kind == "adaptive":
            opponents = [
                (
                    "conservative_chaser",
                    ChasingAgent(0.65, 0.30),
                ),
                (
                    "aggressive_chaser",
                    ChasingAgent(0.35, 0.65),
                ),
                (
                    "controlled_lead_martingale",
                    ControlledLeadMartingaleAgent(),
                ),
                ("lead_protector", LeadProtectionAgent()),
                (
                    "half_bankroll_leader",
                    EarlyLeadAgent(0.25, 0.50),
                ),
                (
                    "adaptive_human",
                    self.create_human_behavior_agent(
                        base_fraction=0.05,
                        maximum_fraction=0.25,
                    ),
                ),
            ]
        elif lineup_kind == "disciplined":
            opponents = [
                ("minimum_control", BasicStrategyAgent()),
                (
                    "conservative_chaser",
                    ChasingAgent(0.70, 0.25),
                ),
                ("lead_protector", LeadProtectionAgent()),
                (
                    "controlled_lead_martingale",
                    ControlledLeadMartingaleAgent(),
                ),
                ("count_aware", CountAwareAgent()),
                (
                    "cautious_human",
                    self.create_human_behavior_agent(
                        base_fraction=0.03,
                        maximum_fraction=0.15,
                    ),
                ),
            ]
        elif lineup_kind == "human":
            opponents = [
                (
                    "controlled_lead_martingale",
                    ControlledLeadMartingaleAgent(),
                ),
                ("lead_protector", LeadProtectionAgent()),
                (
                    "cautious_human",
                    self.create_human_behavior_agent(
                        base_fraction=0.03,
                        maximum_fraction=0.15,
                    ),
                ),
                (
                    "balanced_human",
                    self.create_human_behavior_agent(
                        base_fraction=0.05,
                        maximum_fraction=0.25,
                    ),
                ),
                (
                    "bold_human",
                    self.create_human_behavior_agent(
                        base_fraction=0.08,
                        maximum_fraction=0.40,
                        loss_multiplier=1.75,
                        win_multiplier=1.5,
                        impulse_probability=0.15,
                    ),
                ),
                (
                    "unpredictable_human",
                    self.create_unpredictable_agent(
                        0.40,
                        0.25,
                    ),
                ),
            ]
        elif lineup_kind == "risk_taker":
            opponents = [
                ("minimum_control", BasicStrategyAgent()),
                (
                    "half_bankroll_leader_1",
                    EarlyLeadAgent(0.25, 0.50),
                ),
                (
                    "half_bankroll_leader_2",
                    EarlyLeadAgent(0.25, 0.50),
                ),
                (
                    "opening_all_in_leader",
                    EarlyLeadAgent(0.10, 1.00),
                ),
                ("lead_protector", LeadProtectionAgent()),
                (
                    "aggressive_chaser",
                    ChasingAgent(0.35, 0.65),
                ),
            ]
        elif lineup_kind == "minimum":
            opponents = [
                (f"minimum_{index}", BasicStrategyAgent())
                for index in range(1, 7)
            ]
        else:
            raise ValueError(
                "lineup_kind must be adaptive, disciplined, human, risk_taker, or minimum"
            )

        competitors.extend(opponents)
        return competitors

    def create_unpredictable_agent(
        self,
        minimum_probability,
        maximum_fraction,
    ):
        seed = int(
            self.population.random_generator.integers(
                0,
                2**32,
            )
        )

        return UnpredictableBettingAgent(
            seed=seed,
            minimum_bet_probability=(
                minimum_probability
            ),
            maximum_bankroll_fraction=maximum_fraction,
        )

    def create_tournament_scenario(
        self,
        player_count,
        focal_seat=None,
        stage=None,
        late_stage=None,
    ):
        if stage is None and late_stage is not None:
            stage = "late" if late_stage else "normal"

        if stage is None:
            stage = self.select_training_stage()

        if stage not in {
            "normal",
            "mid",
            "late",
            "extreme",
        }:
            raise ValueError(
                "stage must be normal, mid, late, or extreme"
            )

        if stage == "normal":
            return (
                [
                    Player(self.starting_bankroll)
                    for _ in range(player_count)
                ],
                self.training_round_count(),
                0,
            )

        difficulty = min(
            1.0,
            self.population.generation_number / 49,
        )
        scenario_focal_seat = focal_seat
        if scenario_focal_seat is None:
            scenario_focal_seat = int(
                self.population.random_generator.integers(
                    0,
                    player_count,
                )
            )

        if stage == "mid":
            rounds_remaining = int(
                self.population.random_generator.integers(
                    5,
                    9,
                )
            )
            leader_count = int(
                self.population.random_generator.integers(
                    2,
                    4,
                )
            )
            maximum_second_gap = 0.10 + 0.10 * difficulty
            second_gap = float(
                self.population.random_generator.uniform(
                    0.05,
                    maximum_second_gap,
                )
            )
        elif stage == "late":
            rounds_remaining = int(
                self.population.random_generator.integers(
                    2,
                    4,
                )
            )
            leader_count = 2
            maximum_second_gap = 0.05 + 0.03 * difficulty
            second_gap = float(
                self.population.random_generator.uniform(
                    0.03,
                    maximum_second_gap,
                )
            )
        else:
            rounds_remaining = int(
                self.population.random_generator.integers(
                    2,
                    7,
                )
            )
            bankrolls = (
                self.population.random_generator.uniform(
                    0.40,
                    1.70,
                    size=player_count,
                )
                * self.starting_bankroll
            )
            leader_count = 2
            leader_range = (1.30, 1.80)
            focal_range = (0.55, 1.00)

        if stage in {"mid", "late"}:
            focal_bankroll = float(
                self.population.random_generator.uniform(
                    0.88,
                    1.02,
                )
                * self.starting_bankroll
            )
            bankrolls = (
                self.population.random_generator.uniform(
                    0.72,
                    0.98,
                    size=player_count,
                )
                * focal_bankroll
            )
            bankrolls[scenario_focal_seat] = focal_bankroll
            opponent_seats = [
                seat
                for seat in range(player_count)
                if seat != scenario_focal_seat
            ]
            higher_seats = (
                self.population.random_generator.choice(
                    opponent_seats,
                    size=leader_count,
                    replace=False,
                )
            )
            leader_gaps = [
                second_gap
                + float(
                    self.population.random_generator.uniform(
                        0.06,
                        0.16,
                    )
                ),
                second_gap,
            ]
            if leader_count == 3:
                leader_gaps.append(
                    max(0.02, second_gap * 0.60)
                )
            for seat, gap in zip(
                higher_seats,
                leader_gaps,
            ):
                bankrolls[int(seat)] = (
                    focal_bankroll * (1.0 + gap)
                )
        elif focal_seat is not None:
            focal_bankroll = float(
                self.population.random_generator.uniform(
                    focal_range[0],
                    focal_range[1],
                )
                * self.starting_bankroll
            )
            bankrolls[focal_seat] = focal_bankroll
            opponent_seats = [
                seat
                for seat in range(player_count)
                if seat != focal_seat
            ]
            higher_seats = (
                self.population.random_generator.choice(
                    opponent_seats,
                    size=leader_count,
                    replace=False,
                )
            )
            for seat in higher_seats:
                bankrolls[int(seat)] = (
                    self.population.random_generator.uniform(
                        leader_range[0],
                        leader_range[1],
                    )
                    * self.starting_bankroll
                )
        else:
            leader_seats = (
                self.population.random_generator.choice(
                    player_count,
                    size=leader_count,
                    replace=False,
                )
            )

            for seat in leader_seats:
                bankrolls[int(seat)] = (
                    self.population.random_generator.uniform(
                        leader_range[0],
                        leader_range[1],
                    )
                    * self.starting_bankroll
                )

        players = [
            Player(max(self.minimum_bet, int(bankroll)))
            for bankroll in bankrolls
        ]

        total_rounds = self.rounds_per_tournament
        completed_rounds = max(
            0,
            total_rounds - rounds_remaining,
        )

        return players, total_rounds, completed_rounds

    def select_training_stage(self):
        stage_roll = (
            self.population.random_generator.random()
        )

        if stage_roll < 0.45:
            return "normal"
        if stage_roll < 0.75:
            return "mid"
        if stage_roll < 0.95:
            return "late"
        return "extreme"

    def evaluate_group(self, network_indices):
        if len(network_indices) != 7:
            raise ValueError("There must be 7 players per table")

        players, total_rounds, completed_rounds = (
            self.create_tournament_scenario(
                len(network_indices)
            )
        )
        agents = []
        
        for network_index in network_indices:
            network = self.population.networks[network_index]

            agent = NeuralBettingAgent(network)

            agents.append(agent)


        tournament = Tournament(
            players=players,
            bots=agents,
            number_of_rounds=total_rounds,
            decks=self.decks,
            minimum_bet=self.minimum_bet,
            hit_soft_17=self.hit_soft_17,
            max_hands=self.max_hands
        )

        tournament.current_round_number = (
            completed_rounds
        )

        starting_bankrolls = [
            player.bankroll
            for player in players
        ]

        rankings = tournament.play_tournament()

        self.award_fitness(
            network_indices,
            rankings,
            starting_bankrolls=starting_bankrolls,
        )

        return rankings

    def award_fitness(
        self,
        network_indices,
        rankings,
        starting_bankrolls=None,
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

            shared_credit = (
                advancement_fitness_for_tie(
                    position,
                    tie_end,
                )
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

        if starting_bankrolls is not None:
            final_bankrolls = [0] * len(rankings)
            for seat_index, bankroll in rankings:
                final_bankrolls[seat_index] = bankroll

            for seat_index, population_index in enumerate(
                network_indices
            ):
                self.population.add_fitness(
                    population_index,
                    training_progress_fitness(
                        starting_bankrolls,
                        final_bankrolls,
                        seat_index,
                    ) * self.training_shaping_multiplier(),
                )
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
        benchmark_tournaments=0,
        benchmark_candidate_count=1,
    ):
        if (
            isinstance(benchmark_tournaments, bool)
            or not isinstance(benchmark_tournaments, int)
        ):
            raise TypeError(
                "benchmark_tournaments must be an integer"
            )

        if benchmark_tournaments < 0:
            raise ValueError(
                "benchmark_tournaments cannot be negative"
            )

        if (
            isinstance(benchmark_candidate_count, bool)
            or not isinstance(
                benchmark_candidate_count,
                int,
            )
        ):
            raise TypeError(
                "benchmark_candidate_count must be an integer"
            )

        if benchmark_candidate_count <= 0:
            raise ValueError(
                "benchmark_candidate_count must be positive"
            )

        evaluated_generation = (
            self.population.generation_number
        )

        if baseline_tournaments_per_network > 0:
            fitness_scores = self.evaluate_generation(
                tournaments_per_network=(
                    tournaments_per_network
                ),
                baseline_tournaments_per_network=(
                    baseline_tournaments_per_network
                ),
            )
        else:
            fitness_scores = self.evaluate_generation(
                tournaments_per_network
            )

        ranked_indices = (
            self.population.get_ranked_indices()
        )

        average_fitness = float(
            fitness_scores.mean()
        )

        best_network_index = ranked_indices[0]
        best_network = None
        benchmark_fitness = None
        benchmark_normal_fitness = None
        benchmark_mid_fitness = None
        benchmark_late_fitness = None
        benchmark_holdout_fitness = None
        benchmark_minimum_holdout_fitness = None
        benchmark_selection_fitness = None
        best_benchmark_key = None
        league_promoted = False

        if benchmark_tournaments > 0:
            candidate_count = min(
                benchmark_candidate_count,
                len(ranked_indices),
            )

            # Every candidate starts from the same scenario, seat, human-agent
            # and shoe random states.  This makes candidate differences much
            # less likely to be caused by one network receiving easier cards.
            benchmark_numpy_state = copy.deepcopy(
                self.population.random_generator.bit_generator.state
            )
            benchmark_python_state = random.getstate()

            for candidate_index in ranked_indices[
                :candidate_count
            ]:
                self.population.random_generator.bit_generator.state = (
                    copy.deepcopy(benchmark_numpy_state)
                )
                random.setstate(benchmark_python_state)

                candidate_network = (
                    self.population.networks[
                        candidate_index
                    ].clone()
                )

                normal_tournaments = (
                    benchmark_tournaments * 2 // 5
                )
                late_tournaments = (
                    benchmark_tournaments // 4
                )
                mid_tournaments = (
                    benchmark_tournaments
                    - normal_tournaments
                    - late_tournaments
                )

                candidate_normal = None
                candidate_mid = None
                candidate_late = None
                candidate_holdout = None
                candidate_minimum_holdout = None

                if normal_tournaments > 0:
                    candidate_normal = (
                        self.evaluate_network_against_fixed_benchmark(
                            candidate_network,
                            normal_tournaments,
                            stage="normal",
                        )
                    )

                if mid_tournaments > 0:
                    candidate_mid = (
                        self.evaluate_network_against_fixed_benchmark(
                            candidate_network,
                            mid_tournaments,
                            stage="mid",
                        )
                    )

                if late_tournaments > 0:
                    candidate_late = (
                        self.evaluate_network_against_fixed_benchmark(
                            candidate_network,
                            late_tournaments,
                            stage="late",
                        )
                    )

                holdout_tournaments = max(
                    10,
                    benchmark_tournaments // 5,
                )
                candidate_holdout = (
                    self.evaluate_network_against_fixed_benchmark(
                        candidate_network,
                        holdout_tournaments,
                        stage="normal",
                        lineup_kind="disciplined",
                        full_tournament=True,
                    )
                )
                candidate_minimum_holdout = (
                    self.evaluate_network_against_fixed_benchmark(
                        candidate_network,
                        holdout_tournaments,
                        stage="normal",
                        lineup_kind="minimum",
                        full_tournament=True,
                    )
                )

                candidate_total = (
                    (candidate_normal or 0.0)
                    * normal_tournaments
                    + (candidate_mid or 0.0)
                    * mid_tournaments
                    + (candidate_late or 0.0)
                    * late_tournaments
                )

                candidate_benchmark = (
                    candidate_total
                    / benchmark_tournaments
                )
                candidate_selection = (
                    0.45 * candidate_benchmark
                    + 0.45 * candidate_holdout
                    + 0.10 * candidate_minimum_holdout
                )

                candidate_key = (
                    candidate_late is not None
                    and candidate_late > 0,
                    candidate_selection,
                )

                if (
                    best_benchmark_key is None
                    or candidate_key > best_benchmark_key
                ):
                    best_benchmark_key = candidate_key
                    best_network_index = candidate_index
                    best_network = candidate_network
                    benchmark_fitness = (
                        candidate_benchmark
                    )
                    benchmark_normal_fitness = (
                        candidate_normal
                    )
                    benchmark_mid_fitness = (
                        candidate_mid
                    )
                    benchmark_late_fitness = (
                        candidate_late
                    )
                    benchmark_holdout_fitness = (
                        candidate_holdout
                    )
                    benchmark_minimum_holdout_fitness = (
                        candidate_minimum_holdout
                    )
                    benchmark_selection_fitness = (
                        candidate_selection
                    )

            league_promoted = self.maybe_promote_to_league(
                best_network,
                benchmark_selection_fitness,
                benchmark_normal_fitness,
                benchmark_mid_fitness,
                benchmark_late_fitness,
                benchmark_holdout_fitness,
                benchmark_minimum_holdout_fitness,
                evaluated_generation,
            )
        else:
            best_network = self.population.networks[
                best_network_index
            ].clone()

        best_fitness = float(
            fitness_scores[best_network_index]
        )

        strategy_diversity = measure_strategy_diversity(self.population.networks)

        population_best_fitness = float(max(fitness_scores))

        population_worst_fitness = float(min(fitness_scores))

        population_fitness_std = float(np.std(fitness_scores))

        self.population.create_next_generation()


        return {
            "generation": evaluated_generation,
            "best_network_index": (
                best_network_index
            ),
            "best_fitness": best_fitness,
            "average_fitness": average_fitness,
            "best_network": best_network,
            "benchmark_fitness": benchmark_fitness,
            "benchmark_normal_fitness": (
                benchmark_normal_fitness
            ),
            "benchmark_mid_fitness": (
                benchmark_mid_fitness
            ),
            "benchmark_late_fitness": (
                benchmark_late_fitness
            ),
            "benchmark_holdout_fitness": (
                benchmark_holdout_fitness
            ),
            "benchmark_minimum_holdout_fitness": (
                benchmark_minimum_holdout_fitness
            ),
            "benchmark_selection_fitness": (
                benchmark_selection_fitness
            ),
            "league_promoted": league_promoted,
            "league_size": len(self.champion_league),
            "strategy_diversity": strategy_diversity,
            "population_best_fitness": population_best_fitness,
            "population_worst_fitness": population_worst_fitness,
            "population_fitness_std": population_fitness_std
        }

    def evaluate_robust_checkpoint(
        self,
        network,
        benchmark_tournaments,
    ):
        """Run the stable benchmark used only to verify a checkpoint."""
        if (
            isinstance(benchmark_tournaments, bool)
            or not isinstance(benchmark_tournaments, int)
            or benchmark_tournaments <= 0
        ):
            raise ValueError(
                "benchmark_tournaments must be a positive integer"
            )

        normal_tournaments = benchmark_tournaments * 2 // 5
        late_tournaments = benchmark_tournaments // 4
        mid_tournaments = (
            benchmark_tournaments
            - normal_tournaments
            - late_tournaments
        )
        holdout_tournaments = max(
            10,
            benchmark_tournaments // 5,
        )

        normal_fitness = (
            self.evaluate_network_against_fixed_benchmark(
                network,
                normal_tournaments,
                stage="normal",
            )
        )
        mid_fitness = (
            self.evaluate_network_against_fixed_benchmark(
                network,
                mid_tournaments,
                stage="mid",
            )
        )
        late_fitness = (
            self.evaluate_network_against_fixed_benchmark(
                network,
                late_tournaments,
                stage="late",
            )
        )
        holdout_fitness = (
            self.evaluate_network_against_fixed_benchmark(
                network,
                holdout_tournaments,
                stage="normal",
                lineup_kind="disciplined",
                full_tournament=True,
            )
        )
        minimum_holdout_fitness = (
            self.evaluate_network_against_fixed_benchmark(
                network,
                holdout_tournaments,
                stage="normal",
                lineup_kind="minimum",
                full_tournament=True,
            )
        )
        staged_fitness = (
            normal_fitness * normal_tournaments
            + mid_fitness * mid_tournaments
            + late_fitness * late_tournaments
        ) / benchmark_tournaments
        selection_fitness = (
            0.45 * staged_fitness
            + 0.45 * holdout_fitness
            + 0.10 * minimum_holdout_fitness
        )

        return {
            "benchmark_fitness": staged_fitness,
            "benchmark_normal_fitness": normal_fitness,
            "benchmark_mid_fitness": mid_fitness,
            "benchmark_late_fitness": late_fitness,
            "benchmark_holdout_fitness": holdout_fitness,
            "benchmark_minimum_holdout_fitness": (
                minimum_holdout_fitness
            ),
            "benchmark_selection_fitness": selection_fitness,
        }

    def compare_checkpoint_networks(
        self,
        challenger,
        incumbent,
        benchmark_tournaments,
    ):
        """Compare checkpoint candidates on an independent common batch."""
        numpy_state = copy.deepcopy(
            self.population.random_generator.bit_generator.state
        )
        python_state = random.getstate()
        challenger_metrics = self.evaluate_robust_checkpoint(
            challenger,
            benchmark_tournaments,
        )

        incumbent_metrics = None
        if incumbent is not None:
            self.population.random_generator.bit_generator.state = (
                copy.deepcopy(numpy_state)
            )
            random.setstate(python_state)
            incumbent_metrics = self.evaluate_robust_checkpoint(
                incumbent,
                benchmark_tournaments,
            )

        return challenger_metrics, incumbent_metrics

    def evaluate_network_against_fixed_benchmark(
        self,
        network,
        number_of_tournaments,
        stage=None,
        lineup_kind=None,
        full_tournament=False,
    ):
        if (
            isinstance(number_of_tournaments, bool)
            or not isinstance(number_of_tournaments, int)
        ):
            raise TypeError(
                "number_of_tournaments must be an integer"
            )

        if number_of_tournaments <= 0:
            raise ValueError(
                "number_of_tournaments must be positive"
            )

        if lineup_kind not in {
            None,
            "adaptive",
            "disciplined",
            "human",
            "risk_taker",
            "minimum",
        }:
            raise ValueError(
                "lineup_kind must be adaptive, disciplined, human, risk_taker, minimum, or None"
            )

        if full_tournament and stage not in {
            None,
            "normal",
        }:
            raise ValueError(
                "full_tournament can only be used with normal tables"
            )

        total_fitness = 0.0
        lineup_kinds = (
            "adaptive",
            "disciplined",
            "human",
            "risk_taker",
        )

        for tournament_index in range(number_of_tournaments):
            selected_lineup = lineup_kind
            if selected_lineup is None:
                selected_lineup = lineup_kinds[
                    tournament_index % len(lineup_kinds)
                ]
            competitors = (
                self.create_fixed_benchmark_competitors(
                    network,
                    selected_lineup,
                )
            )

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
                for strategy_name, _ in shuffled_competitors
            ]

            neural_seat = seat_names.index("neural")

            players, total_rounds, completed_rounds = (
                self.create_tournament_scenario(
                    len(shuffled_competitors),
                    focal_seat=neural_seat,
                    stage=stage,
                )
            )
            if full_tournament:
                total_rounds = self.rounds_per_tournament
                completed_rounds = 0

            tournament = Tournament(
                players=players,
                bots=[
                    bot
                    for _, bot in shuffled_competitors
                ],
                number_of_rounds=total_rounds,
                decks=self.decks,
                minimum_bet=self.minimum_bet,
                hit_soft_17=self.hit_soft_17,
                max_hands=self.max_hands,
            )

            tournament.current_round_number = (
                completed_rounds
            )

            rankings = tournament.play_tournament()
            ranking_index = next(
                index
                for index, (seat, _) in enumerate(rankings)
                if seat == neural_seat
            )

            neural_bankroll = rankings[ranking_index][1]
            tie_start = ranking_index
            tie_end = ranking_index + 1

            while (
                tie_start > 0
                and rankings[tie_start - 1][1]
                == neural_bankroll
            ):
                tie_start -= 1

            while (
                tie_end < len(rankings)
                and rankings[tie_end][1]
                == neural_bankroll
            ):
                tie_end += 1

            total_fitness += advancement_fitness_for_tie(
                tie_start,
                tie_end,
            )

        return total_fitness / number_of_tournaments

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
            stage = self.select_training_stage()
            table_kind = self.select_training_table_kind()

            competitors = (
                self.create_training_baseline_competitors(
                    network,
                    table_kind=table_kind,
                )
            )

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

            neural_seat_index = seat_names.index(
                "neural"
            )

            players, total_rounds, completed_rounds = (
                self.create_tournament_scenario(
                    len(shuffled_competitors),
                    focal_seat=neural_seat_index,
                    stage=stage,
                )
            )

            tournament = Tournament(
                players=players,
                bots=bots,
                number_of_rounds=total_rounds,
                decks=self.decks,
                minimum_bet=self.minimum_bet,
                hit_soft_17=self.hit_soft_17,
                max_hands=self.max_hands,
            )

            tournament.current_round_number = (
                completed_rounds
            )

            starting_bankrolls = [
                player.bankroll
                for player in players
            ]

            rankings = tournament.play_tournament()

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

            total_fitness += advancement_fitness_for_tie(
                tie_start,
                tie_end,
            )
            final_bankrolls = [0] * len(rankings)
            for seat_index, bankroll in rankings:
                final_bankrolls[seat_index] = bankroll
            total_fitness += training_progress_fitness(
                starting_bankrolls,
                final_bankrolls,
                neural_seat_index,
            ) * self.training_shaping_multiplier()

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
