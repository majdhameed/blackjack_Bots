from datetime import datetime
from pathlib import Path

from ml.population import Population
from ml.trainer import Trainer
from ml.training_metrics import (
    save_training_history,
    build_generation_record,
)


def create_run_artifact_paths(base_directory="runs", timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()

    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    run_directory = Path(base_directory) / run_id
    run_directory.mkdir(parents=True, exist_ok=True)

    return {
        "run_directory": run_directory,
        "model_path": run_directory / "best_betting_network.npz",
        "metrics_path": run_directory / "metrics.csv",
    }

def create_population(population_size, elite_count, mutation_rate, mutation_strength, seed=None):

    population = Population(population_size, elite_count, mutation_rate, mutation_strength, seed)
    return population

def create_trainer(
    population,
    starting_bankroll,
    rounds_per_tournament,
    decks,
    minimum_bet,
    hit_soft_17,
    max_hands,
    randomize_training_rounds=False,
    league_directory=None,
):
    trainer = Trainer(
        population,
        starting_bankroll,
        rounds_per_tournament,
        decks,
        minimum_bet,
        hit_soft_17,
        max_hands,
        randomize_training_rounds,
        league_directory=league_directory,
    )
    return trainer

def run_training(
    trainer,
    generations,
    tournaments_per_network,
    output_path,
    baseline_tournaments_per_network=0,
    print_fn=print,
    benchmark_tournaments_per_generation=0,
    benchmark_candidate_count=1,
    checkpoint_verification_tournaments=0,
    metrics_output_path=None,
):

    if type(generations) is not int:
        raise TypeError("Generations must be an integer")
    if generations <= 0:
        raise ValueError("There must be atleast 1 generation")

    if type(tournaments_per_network) is not int:
        raise TypeError("Generations must be an integer")
    if tournaments_per_network <= 0:
        raise ValueError("There must be atleast 1 generation")

    if (
        isinstance(
            benchmark_tournaments_per_generation,
            bool,
        )
        or not isinstance(
            benchmark_tournaments_per_generation,
            int,
        )
    ):
        raise TypeError(
            "benchmark_tournaments_per_generation must be an integer"
        )

    if benchmark_tournaments_per_generation < 0:
        raise ValueError(
            "benchmark_tournaments_per_generation cannot be negative"
        )

    if (
        isinstance(benchmark_candidate_count, bool)
        or not isinstance(benchmark_candidate_count, int)
    ):
        raise TypeError(
            "benchmark_candidate_count must be an integer"
        )

    if benchmark_candidate_count <= 0:
        raise ValueError(
            "benchmark_candidate_count must be positive"
        )

    if (
        isinstance(checkpoint_verification_tournaments, bool)
        or not isinstance(
            checkpoint_verification_tournaments,
            int,
        )
    ):
        raise TypeError(
            "checkpoint_verification_tournaments must be an integer"
        )
    if checkpoint_verification_tournaments < 0:
        raise ValueError(
            "checkpoint_verification_tournaments cannot be negative"
        )
    if 0 < checkpoint_verification_tournaments < 4:
        raise ValueError(
            "checkpoint_verification_tournaments must be zero or at least four"
        )
    
    history = []
    best_network = None
    best_fitness = float('-inf')
    best_comparison_key = None
    best_benchmark_fitness = None
    best_benchmark_normal_fitness = None
    best_benchmark_mid_fitness = None
    best_benchmark_late_fitness = None
    best_benchmark_holdout_fitness = None
    best_benchmark_minimum_holdout_fitness = None
    best_benchmark_selection_fitness = None
    best_generation = None
    metrics_history = []


    for _ in range(generations):
        verified_score = None
        if (
            baseline_tournaments_per_network > 0
            or benchmark_tournaments_per_generation > 0
        ):
            result = trainer.train_generation(
                tournaments_per_network=(
                    tournaments_per_network
                ),
                baseline_tournaments_per_network=(
                    baseline_tournaments_per_network
                ),
                benchmark_tournaments=(
                    benchmark_tournaments_per_generation
                ),
                benchmark_candidate_count=(
                    benchmark_candidate_count
                ),
            )
        else:
            result = trainer.train_generation(
                tournaments_per_network
            )

        history_entry = {
            "generation": result["generation"],
            "best_network_index": result[
                "best_network_index"
            ],
            "best_fitness": result["best_fitness"],
            "average_fitness": result[
                "average_fitness"
            ],
            "benchmark_fitness": result.get(
                "benchmark_fitness"
            ),
            "benchmark_normal_fitness": result.get(
                "benchmark_normal_fitness"
            ),
            "benchmark_mid_fitness": result.get(
                "benchmark_mid_fitness"
            ),
            "benchmark_late_fitness": result.get(
                "benchmark_late_fitness"
            ),
            "benchmark_holdout_fitness": result.get(
                "benchmark_holdout_fitness"
            ),
            "benchmark_minimum_holdout_fitness": result.get(
                "benchmark_minimum_holdout_fitness"
            ),
            "benchmark_selection_fitness": result.get(
                "benchmark_selection_fitness"
            ),
        }

        history.append(history_entry)

        
        print_fn(
            f"Generation {result['generation']}: "
            f"best fitness = "
            f"{result['best_fitness']:.2f}, "
            f"average fitness = "
            f"{result['average_fitness']:.2f}"
        )

        if result.get("benchmark_fitness") is not None:
            print_fn(
                "Fixed benchmark average advancement "
                f"fitness = {result['benchmark_fitness']:.2f}"
            )

            print_fn(
                "  normal tables = "
                f"{result['benchmark_normal_fitness']:.2f}, "
                "mid-stage tables = "
                f"{result['benchmark_mid_fitness']:.2f}, "
                "late-stage tables = "
                f"{result['benchmark_late_fitness']:.2f}"
            )

            print_fn(
                "  full-tournament holdout = "
                f"{result['benchmark_holdout_fitness']:.2f}, "
                "six-minimum holdout = "
                f"{result['benchmark_minimum_holdout_fitness']:.2f}, "
                "robust selection score = "
                f"{result['benchmark_selection_fitness']:.2f}"
            )

            if result.get("league_promoted"):
                print_fn(
                    "  promoted to opponent league; "
                    f"league size = {result['league_size']}"
                )

        selection_fitness = result.get(
            "benchmark_selection_fitness"
        )
        if selection_fitness is None:
            selection_fitness = result["best_fitness"]
            comparison_key = (True, selection_fitness)
        else:
            comparison_key = (
                result.get("benchmark_late_fitness", 0) > 0,
                selection_fitness,
            )

        should_save = (
            best_comparison_key is None
            or comparison_key > best_comparison_key
        )

    
        metrics_for_checkpoint = result


        if (
            should_save
            and selection_fitness is not None
            and checkpoint_verification_tournaments > 0
        ):
            compare_networks = getattr(
                trainer,
                "compare_checkpoint_networks",
                None,
            )
            if callable(compare_networks):
                challenger_metrics, incumbent_metrics = (
                    compare_networks(
                        result["best_network"],
                        best_network,
                        checkpoint_verification_tournaments,
                    )
                )

                challenger_key = (
                    challenger_metrics[
                        "benchmark_late_fitness"
                    ] > 0,
                    challenger_metrics[
                        "benchmark_selection_fitness"
                    ],
                )
                incumbent_key = None
                if incumbent_metrics is not None:
                    incumbent_key = (
                        incumbent_metrics[
                            "benchmark_late_fitness"
                        ] > 0,
                        incumbent_metrics[
                            "benchmark_selection_fitness"
                        ],
                    )
                should_save = (
                    incumbent_key is None
                    or challenger_key > incumbent_key
                )
                comparison_key = challenger_key
                metrics_for_checkpoint = challenger_metrics
                incumbent_text = (
                    "none"
                    if incumbent_metrics is None
                    else f"{incumbent_key[1]:.3f}"
                )
                print_fn(
                    "  independent checkpoint verification: "
                    "challenger = "
                    f"{challenger_key[1]:.3f}, incumbent = "
                    f"{incumbent_text}"
                )
                if not should_save:
                    print_fn(
                        "  checkpoint rejected by independent verification"
                    )
                verified_score = challenger_metrics["benchmark_selection_fitness"]
            
                

        if should_save:
            best_comparison_key = comparison_key
            best_fitness = result["best_fitness"]
            best_benchmark_fitness = metrics_for_checkpoint.get(
                "benchmark_fitness"
            )
            best_benchmark_normal_fitness = metrics_for_checkpoint.get(
                "benchmark_normal_fitness"
            )
            best_benchmark_mid_fitness = metrics_for_checkpoint.get(
                "benchmark_mid_fitness"
            )
            best_benchmark_late_fitness = metrics_for_checkpoint.get(
                "benchmark_late_fitness"
            )
            best_benchmark_holdout_fitness = metrics_for_checkpoint.get(
                "benchmark_holdout_fitness"
            )
            best_benchmark_minimum_holdout_fitness = metrics_for_checkpoint.get(
                "benchmark_minimum_holdout_fitness"
            )
            best_benchmark_selection_fitness = metrics_for_checkpoint.get(
                "benchmark_selection_fitness"
            )
            best_generation = result["generation"]
            best_network = result["best_network"]
            best_network.save(output_path)
            print_fn(
                "Checkpoint saved: "
                f"{output_path} "
                f"(generation {best_generation})"
            )
       

        if metrics_output_path:
            record = build_generation_record(result, verified_score, should_save, )
            metrics_history.append(record)
            save_training_history(metrics_output_path, metrics_history)


    return{
        "best_generation": best_generation,
        "best_fitness": best_fitness,
        "best_benchmark_fitness": (
            best_benchmark_fitness
        ),
        "best_benchmark_normal_fitness": (
            best_benchmark_normal_fitness
        ),
        "best_benchmark_mid_fitness": (
            best_benchmark_mid_fitness
        ),
        "best_benchmark_late_fitness": (
            best_benchmark_late_fitness
        ),
        "best_benchmark_holdout_fitness": (
            best_benchmark_holdout_fitness
        ),
        "best_benchmark_minimum_holdout_fitness": (
            best_benchmark_minimum_holdout_fitness
        ),
        "best_benchmark_selection_fitness": (
            best_benchmark_selection_fitness
        ),
        "best_network": best_network,
        "output_path": output_path,
        "history": history,
        "metrics_history": metrics_history
    }


def main():
    artifact_paths = create_run_artifact_paths()

    population = create_population(
        population_size=70,
        elite_count=7,
        mutation_rate=0.05,
        mutation_strength=0.1,
        seed=123,
    )

    trainer = create_trainer(
        population=population,
        starting_bankroll=10_000,
        rounds_per_tournament=12,
        decks=6,
        minimum_bet=100,
        hit_soft_17=True,
        max_hands=4,
        randomize_training_rounds=True,
        league_directory="models/league",
    )
    summary = run_training(
        trainer=trainer,
        generations=10,
        tournaments_per_network=30,
        baseline_tournaments_per_network=50,
        benchmark_tournaments_per_generation=300,
        benchmark_candidate_count=10,
        checkpoint_verification_tournaments=1_000,
        output_path=artifact_paths["model_path"],
        metrics_output_path=artifact_paths["metrics_path"],
    )

    print()
    print(
        "Best generation:",
        summary["best_generation"],
    )
    print(
        "Best fitness:",
        summary["best_fitness"],
    )
    print(
        "Best strict benchmark:",
        summary["best_benchmark_fitness"],
    )
    print(
        "Best benchmark stages:",
        "normal =",
        summary["best_benchmark_normal_fitness"],
        "mid =",
        summary["best_benchmark_mid_fitness"],
        "late =",
        summary["best_benchmark_late_fitness"],
    )
    print(
        "Best full-tournament holdout:",
        summary["best_benchmark_holdout_fitness"],
    )
    print(
        "Best six-minimum holdout:",
        summary["best_benchmark_minimum_holdout_fitness"],
    )
    print(
        "Best robust selection score:",
        summary["best_benchmark_selection_fitness"],
    )
    print(
        "Saved network:",
        summary["output_path"],
    )


if __name__ == "__main__":
    main()
