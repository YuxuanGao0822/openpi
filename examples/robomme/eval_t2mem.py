"""Run inside the official RoboMME simulator environment, separate from openpi."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robomme.env_record_wrapper import BenchmarkEnvBuilder
import robomme.robomme_env  # noqa: F401 -- register tasks

TASKS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "ButtonUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)


def request(obs, prompt, index, *, op, write):
    state = np.concatenate(
        (np.asarray(obs["joint_state_list"][index]), np.asarray(obs["gripper_state_list"][index])[:1])
    ).astype(np.float32)
    return {
        "op": op,
        "front": np.asarray(obs["front_rgb_list"][index]),
        "wrist": np.asarray(obs["wrist_rgb_list"][index]),
        "state": state,
        "prompt": prompt,
        "write": write,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--seed", type=int, required=True, help="Evaluation model/RNG seed; benchmark test seeds remain fixed"
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--demo-content", choices=["correct", "empty"], default="correct")
    args = parser.parse_args()
    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    stride, offset = metadata["write_stride"], metadata["demo_write_offset"]
    execute = metadata["execute"]
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        builder = BenchmarkEnvBuilder(
            env_id=task, dataset="test", action_space="joint_angle", gui_render=False, max_steps=args.max_steps
        )
        for episode in range(builder.get_episode_num()):
            env = builder.make_env_for_episode(episode)
            obs, info = env.reset()
            prompt = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]
            # Same model seed/call cadence across interventions; no fast-state leakage.
            client.infer({"op": "reset", "seed": args.seed})
            for frame in range(len(obs["front_rgb_list"]) - 1):
                if frame >= offset and (frame - offset) % stride == 0:
                    client.infer(request(obs, prompt, frame, op="observe", write=args.demo_content == "correct"))
            plan = deque()
            stopped, steps = False, 0
            while not stopped and steps < args.max_steps:
                if not plan:
                    response = client.infer(request(obs, prompt, -1, op="infer", write=steps % stride == 0))
                    plan.extend(response["actions"][:execute])
                elif steps % stride == 0:
                    # Midpoint observation updates memory without denoising/replanning.
                    client.infer(request(obs, prompt, -1, op="observe", write=True))
                obs, _, terminated, truncated, info = env.step(plan.popleft())
                steps += 1
                stopped = bool(terminated) or bool(truncated)
            record = {
                "task": task,
                "episode": episode,
                "seed": args.seed,
                "difficulty": env.unwrapped.difficulty,
                "success": info["status"] == "success",
                "status": info["status"],
                "steps": steps,
            }
            records.append(record)
            env.close()
            # Persist explicit terminal results. Simulator/model errors propagate.
            args.output.write_text(
                json.dumps(
                    {
                        "protocol": {
                            "split": "test",
                            "horizon": metadata["horizon"],
                            "execute": execute,
                            "write_stride": stride,
                            "demo_content": args.demo_content,
                        },
                        "episodes": records,
                    },
                    indent=2,
                )
            )
            print(record, flush=True)


if __name__ == "__main__":
    main()
