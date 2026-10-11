"""Pool three evaluation seeds over explicitly selected RoboMME tasks."""

import argparse
from collections import Counter
import json
from pathlib import Path

from openpi.t2mem.config import TASKS


def summarize(paths, tasks):
    records = [record for path in paths for record in json.loads(path.read_text())["episodes"]]
    identity = [(record["task"], record["seed"], record["episode"]) for record in records]
    if len(identity) != len(set(identity)):
        raise ValueError("Duplicate task/seed/episode; pass one checkpoint and three distinct evaluation seeds")
    scores = {}
    for task in tasks:
        episodes = [record for record in records if record["task"] == task]
        if len(episodes) != 150 or len({record["seed"] for record in episodes}) != 3:
            raise ValueError(f"{task}: protocol requires 50 episodes x 3 seeds")
        for seed in {record["seed"] for record in episodes}:
            if {record["episode"] for record in episodes if record["seed"] == seed} != set(range(50)):
                raise ValueError(f"{task}, seed {seed}: incomplete test split")
            difficulty = Counter(record["difficulty"] for record in episodes if record["seed"] == seed)
            if difficulty != {"easy": 26, "medium": 12, "hard": 12}:
                raise ValueError(f"{task}, seed {seed}: unexpected test difficulty distribution {difficulty}")
        scores[task] = 100.0 * sum(record["success"] for record in episodes) / len(episodes)
    suites = {
        name: sum(scores[task] for task in TASKS[i : i + 4]) / 4
        for i, name in zip(range(0, 16, 4), ("Counting", "Permanence", "Reference", "Imitation"), strict=True)
        if all(task in scores for task in TASKS[i : i + 4])
    }
    return {
        "scope": "all_tasks" if set(tasks) == set(TASKS) else "selected_tasks",
        "evaluated_tasks": tasks,
        "task_success_percent": scores,
        "suite_success_percent": suites,
        "AVG_percent": sum(scores.values()) / len(scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--compare", nargs="+", type=Path, help="Second model's three-seed result files")
    args = parser.parse_args()
    result = summarize(args.results, args.tasks)
    if args.compare:
        comparison = summarize(args.compare, args.tasks)
        result = {
            "baseline": result,
            "comparison": comparison,
            "delta_percentage_points": {
                task: comparison["task_success_percent"][task] - result["task_success_percent"][task]
                for task in args.tasks
            },
            "AVG_delta_percentage_points": comparison["AVG_percent"] - result["AVG_percent"],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
