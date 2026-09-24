import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from pathlib import Path
from collections import defaultdict
from itertools import islice

FITNESS_VALUES = [
    "training_best",
    "training_mean",
    "training_worst",
]

EVALUATION_VALUES = [
    "benchmark_score",
    "normal_score",
    "mid_score",
    "late_score",
    "full_tournament_score",
    "minimum_control_score",
    "robust_score",
    "verified_score",
]


def create_fitness_plot(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lists = defaultdict(list)

    for record in records:
        lists["generation"].append(record["generation"])
        for value in FITNESS_VALUES:
            
            lists[value].append(record[value])
        
        lists["lower"].append(
            record["training_mean"] - record["training_std"]
        )
        lists["upper"].append(
            record["training_mean"] + record["training_std"]
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )
    
    for value in FITNESS_VALUES:
        ax.plot(lists["generation"], lists[value], label=value, linewidth=2)

    ax.fill_between(lists["generation"], lists["lower"], lists["upper"], interpolate=False, step=None)

    ax.set_title("Metrics Graph")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

def create_evaluation_plot(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lists = defaultdict(list)

    for record in records:
        lists["generation"].append(record["generation"])
        for value in EVALUATION_VALUES:
            lists[value].append(record[value])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    for value in EVALUATION_VALUES:
        ax.plot(lists["generation"], lists[value], label=value, linewidth=2)

        ax.set_title("Evaluation and Holdout Scores")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Score")

    checkpoint_generations = []
    checkpoint_scores = []

    for record in records:
        if record["checkpoint_saved"] == True:
            checkpoint_generations.append(record["generation"])
            checkpoint_scores.append(record["robust_score"])

    
    ax.scatter(checkpoint_generations, checkpoint_scores)



    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

