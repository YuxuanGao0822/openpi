"""Convert official training HDF5 without subgoal/task-progress annotations."""

import argparse
import json
import lzma
from pathlib import Path
import shutil
import tarfile

import h5py
import numpy as np

from openpi.t2mem.config import TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for archive in args.input.rglob("*.h5.tar.xz"):
        with tarfile.open(archive, "r:xz") as bundle:
            for member in bundle.getmembers():
                if member.isfile() and member.name.endswith(".h5"):
                    target = archive.parent / Path(member.name).name
                    if not target.exists():
                        with bundle.extractfile(member) as source, target.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
    for archive in args.input.rglob("*.h5.xz"):
        target = archive.with_suffix("")
        if not target.exists():
            with lzma.open(archive, "rb") as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    manifest, states, actions = [], [], []
    for path in sorted(args.input.rglob("*.h5")):
        task = path.stem.split("_")[-1]
        if task not in TASKS:
            continue
        with h5py.File(path, "r") as file:
            for name in sorted(file, key=lambda key: int(key.split("_")[-1])):
                ep = file[name]
                indices = sorted(int(key.split("_")[-1]) for key in ep if key.startswith("timestep_"))
                steps = [ep[f"timestep_{index}"] for index in indices]
                demo = np.array([bool(step["info"]["is_video_demo"][()]) for step in steps])
                start = int(np.flatnonzero(~demo)[0])
                ep_state = np.stack(
                    [
                        np.concatenate((step["obs"]["joint_state"][()], step["obs"]["gripper_state"][()][:1]))
                        for step in steps
                    ]
                ).astype(np.float32)
                ep_actions = np.stack([step["action"]["joint_action"][()] for step in steps]).astype(np.float32)
                goal = ep["setup"]["task_goal"][()][0].decode("utf-8")
                output = args.output / task / f"{name}.npz"
                output.parent.mkdir(exist_ok=True)
                np.savez_compressed(
                    output,
                    front=np.stack([step["obs"]["front_rgb"][()] for step in steps]),
                    wrist=np.stack([step["obs"]["wrist_rgb"][()] for step in steps]),
                    state=ep_state,
                    actions=ep_actions,
                    execution_start=np.asarray(start),
                    prompt=np.asarray(goal),
                )
                manifest.append(
                    {
                        "task": task,
                        "path": str(output.relative_to(args.output)),
                        "frames": len(steps),
                        "execution_start": start,
                    }
                )
                states.append(ep_state[start:])
                actions.append(ep_actions[start:])
    norm_stats = {}
    for key, arrays in (("state", states), ("actions", actions)):
        values = np.concatenate(arrays)
        norm_stats[key] = {
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "norm_stats.json").write_text(json.dumps({"norm_stats": norm_stats}, indent=2))
    print(f"Prepared {len(manifest)} episodes across {len({x['task'] for x in manifest})} tasks")


if __name__ == "__main__":
    main()
