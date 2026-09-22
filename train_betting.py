from ml.population import Population
from ml.trainer import Trainer


def create_population(population_size, elite_count, mutation_rate, mutation_strength, seed=None):

    population = Population(population_size, elite_count, mutation_rate, mutation_strength, seed)
    return population

def create_trainer(population, starting_bankroll, rounds_per_tournament, decks, minimum_bet, hit_soft_17, max_hands):
    trainer = Trainer(population, starting_bankroll, rounds_per_tournament, decks, minimum_bet, hit_soft_17, max_hands)
    return trainer

def run_training(trainer, generations, tournaments_per_network, output_path, baseline_tournaments_per_network=0, print_fn=print):

    if type(generations) is not int:
        raise TypeError("Generations must be an integer")
    if generations <= 0:
        raise ValueError("There must be atleast 1 generation")

    if type(tournaments_per_network) is not int:
        raise TypeError("Generations must be an integer")
    if tournaments_per_network <= 0:
        raise ValueError("There must be atleast 1 generation")
    
    history = []
    best_network = None
    best_fitness = float('-inf')
    best_generation = None


    for _ in range(generations):
        result = trainer.train_generation(
            tournaments_per_network=(
                tournaments_per_network
            ),
            baseline_tournaments_per_network=(
                baseline_tournaments_per_network
            ),
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
        }

        history.append(history_entry)

        
        print_fn(
            f"Generation {result['generation']}: "
            f"best fitness = "
            f"{result['best_fitness']:.2f}, "
            f"average fitness = "
            f"{result['average_fitness']:.2f}"
        )

        if result["best_fitness"] > best_fitness:
            best_fitness = result["best_fitness"]
            best_generation = result["generation"]
            best_network = result["best_network"]

    best_network.save(output_path)

    return{
        "best_generation": best_generation,
        "best_fitness": best_fitness,
        "best_network": best_network,
        "output_path": output_path,
        "history": history
    }


def main():
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
    )
    summary = run_training(
        trainer=trainer,
        generations=50,
        tournaments_per_network=50,
        baseline_tournaments_per_network=0,
        output_path=(
            "models/best_betting_network.npz"
        ),
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
        "Saved network:",
        summary["output_path"],
    )


if __name__ == "__main__":
    main()