import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from pathlib import Path
from collections import defaultdict
from itertools import islice


from ml.betting_network import BETTING_ACTION_NAMES

from ml.training_metrics import load_training_history

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

DIVERSITY_VALUES = [
    "action_entropy",
    "unique_policy_rate"
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

def create_diversity_plot(records, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lists = defaultdict(list)

    for record in records:
        lists["generation"].append(record["generation"])
        for value in DIVERSITY_VALUES:
            lists[value].append(record[value])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    for value in DIVERSITY_VALUES:
        ax.plot(lists["generation"], lists[value], label=value, linewidth=2)

    ax.set_ybound(lower=0, upper=1)

    ax.set_title("Population Strategy Diversity")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized Diversity")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

def create_action_plot(records, output_path):
    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )    

    genlist = []

    lists = defaultdict(list)

    action_lists = [
        lists[action]
        for action in BETTING_ACTION_NAMES
    ]

    for record in records:
        genlist.append(record["generation"])
        for action in BETTING_ACTION_NAMES:
            lists[action].append(record["action_"+action])


    fig, ax = plt.subplots(figsize=(12, 7))

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )


    ax.stackplot(genlist, *action_lists, labels=BETTING_ACTION_NAMES)

    ax.legend(loc="upper left")


    ax.set_ybound(lower=0, upper=1)

    ax.set_title("Population Betting Actions")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Action Percentage")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

def generate_training_plots(metrics_path):
    metrics_path = Path(metrics_path)

    records = load_training_history(metrics_path)

    directory = metrics_path.parent

    output_path_dict = {
        "fitness": directory / "fitness.png",
        "evaluation": directory / "evaluation.png",
        "diversity": directory / "diversity.png",
        "actions": directory / "actions.png",
}

    create_fitness_plot(records, output_path_dict["fitness"])
    create_evaluation_plot(records, output_path_dict["evaluation"])
    create_diversity_plot(records, output_path_dict["diversity"])
    create_action_plot(records, output_path_dict["actions"])

    return output_path_dict

