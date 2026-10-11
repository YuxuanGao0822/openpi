"""Synthetic summary fixtures; these numbers are not experimental results."""

import json

import pytest

from scripts.t2mem.summarize import summarize


def seed_files(tmp_path, *, successes):
    paths = []
    for seed in range(3):
        records = [
            {
                "task": "PickXtimes",
                "seed": seed,
                "episode": episode,
                "difficulty": "easy" if episode < 26 else "medium" if episode < 38 else "hard",
                "success": episode < successes,
            }
            for episode in range(50)
        ]
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps({"episodes": records}))
        paths.append(path)
    return paths


def test_single_task_summary_does_not_require_other_tasks(tmp_path):
    paths = seed_files(tmp_path, successes=25)
    result = summarize(paths, ["PickXtimes"])
    assert result["task_success_percent"] == {"PickXtimes": 50.0}
    assert result["AVG_percent"] == 50.0
    assert result["scope"] == "selected_tasks"
    assert result["suite_success_percent"] == {}


def test_partial_test_split_is_rejected(tmp_path):
    paths = seed_files(tmp_path, successes=25)
    with pytest.raises(ValueError, match="50 episodes x 3 seeds"):
        summarize(paths[:2], ["PickXtimes"])


def test_duplicate_seed_results_are_rejected(tmp_path):
    paths = seed_files(tmp_path, successes=25)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([*paths, paths[0]], ["PickXtimes"])
