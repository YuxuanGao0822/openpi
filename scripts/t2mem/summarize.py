"""Pool three model-seed rollouts per task, then macro-average task success."""

import argparse
from collections import Counter
import json
from pathlib import Path

from openpi.t2mem.config import TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    records = [record for path in args.results for record in json.loads(path.read_text())["episodes"]]
    identity = [(record["task"], record["seed"], record["episode"]) for record in records]
    if len(identity) != len(set(identity)):
        raise ValueError("Duplicate task/seed/episode; pass one checkpoint and three distinct evaluation seeds")
    scores = {}
    for task in TASKS:
        episodes = [record for record in records if record["task"] == task]
        if len(episodes) != 150 or len({record["seed"] for record in episodes}) != 3:
            raise ValueError(f"{task}: main-paper protocol requires 50 episodes x 3 seeds")
        for seed in {record["seed"] for record in episodes}:
            if {record["episode"] for record in episodes if record["seed"] == seed} != set(range(50)):
                raise ValueError(f"{task}, seed {seed}: incomplete test split")
            difficulty = Counter(record["difficulty"] for record in episodes if record["seed"] == seed)
            if difficulty != {"easy": 26, "medium": 12, "hard": 12}:
                raise ValueError(f"{task}, seed {seed}: unexpected test difficulty distribution {difficulty}")
        scores[task] = 100.0 * sum(record["success"] for record in episodes) / len(episodes)
    print(
        json.dumps(
            {
                "task_success_percent": scores,
                "suite_success_percent": {
                    name: sum(scores[task] for task in TASKS[i : i + 4]) / 4
                    for i, name in zip(
                        range(0, 16, 4), ("Counting", "Permanence", "Reference", "Imitation"), strict=True
                    )
                },
                "AVG_percent": sum(scores.values()) / len(TASKS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
