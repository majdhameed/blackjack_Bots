import csv
from pathlib import Path
import shutil

from ml.betting_network import BETTING_ACTION_NAMES

def build_generation_record(generation_result, verified_score=None, checkpoint_saved=False):
    strategy_diversity = generation_result["strategy_diversity"]

    
    record = {
        "generation": generation_result["generation"],
        "training_best": generation_result["population_best_fitness"],
        "training_mean": generation_result["average_fitness"],
        "training_worst": generation_result["population_worst_fitness"],
        "training_std": generation_result["population_fitness_std"],
        "selected_candidate_fitness": generation_result["best_fitness"],
        "benchmark_score": generation_result["benchmark_fitness"],
        "normal_score": generation_result["benchmark_normal_fitness"],
        "mid_score": generation_result["benchmark_mid_fitness"],
        "late_score": generation_result["benchmark_late_fitness"],
        "full_tournament_score": generation_result["benchmark_holdout_fitness"],
        "minimum_control_score": generation_result["benchmark_minimum_holdout_fitness"],
        "robust_score": generation_result["benchmark_selection_fitness"],
        "verified_score": verified_score,
        "checkpoint_saved": checkpoint_saved,
        "league_promoted": generation_result["league_promoted"],
        "league_size": generation_result["league_size"],
        "total_probe_decisions": strategy_diversity["total_decisions"],
        "action_entropy": strategy_diversity["action_entropy"],
        "unique_policy_count": strategy_diversity["unique_policy_count"],
        "unique_policy_rate": strategy_diversity["unique_policy_rate"],
    }

    for action in BETTING_ACTION_NAMES:
        name = f"action_{action}"
        value = strategy_diversity["action_percentages"][action]
        record[name] = value

    return record

def save_training_history(output_path, records):

    output_path = Path(output_path)
    if len(records) == 0:
        raise ValueError("no records")
    keys = records[0].keys()

    for record in records:
        if record.keys() != keys:

            raise ValueError("Record keys don't match")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = output_path.with_name(f".{output_path.name}.tmp")

    with temp.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)

        writer.writeheader()
        writer.writerows(records)
    
    temp.replace(output_path)



