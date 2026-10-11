"""Stage 1 and alternating 2A/2B; torchrun DDP, task-homogeneous updates."""

import argparse
from contextlib import nullcontext
import dataclasses
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from openpi.t2mem.config import TASKS
from openpi.t2mem.config import MemoryConfig
from openpi.t2mem.data import Episodes
from openpi.t2mem.data import ObservationTransform
from openpi.t2mem.data import sequence_indices
from openpi.t2mem.data import write_due
from openpi.t2mem.model import T2Mem


class SequenceLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, episode, indices, transform, supervise_stride, device):
        model = self.model
        start = int(episode["execution_start"])
        memory = model.reset_memory(1)
        losses = []
        for index in indices:
            obs = transform.observation(
                episode["front"][index],
                episode["wrist"][index],
                episode["state"][index],
                str(episode["prompt"]),
                stage1=model.stage == "stage1",
                device=device,
            )
            due = model.stage != "stage1" and write_due(index, start, model.config)
            encoded, updated = model.encode(obs, memory, torch.tensor([due], device=device), retain_graph=True)
            if index >= start and (model.stage == "stage1" or (index - start) % supervise_stride == 0):
                actions, mask = transform.chunk(episode, index, device)
                losses.append(model.action_loss(encoded, obs, actions, mask).mean())
            memory = updated
        return torch.stack(losses).mean()


def optimizer(model, group, lr):
    if group == "memory":
        gates, others = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (gates if name.endswith(".alpha") else others).append(param)
        return torch.optim.AdamW([{"params": others, "lr": lr}, {"params": gates, "lr": 5e-4}], weight_decay=0)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path)
    parser.add_argument("--init", type=Path, help="Stage-1 checkpoint; introduces fresh memory")
    parser.add_argument("--resume", type=Path, help="Full checkpoint, including both optimizers and RNG")
    parser.add_argument("--stage", choices=["stage1", "alternating"], required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument(
        "--updates", type=int, required=True, help="Stage 1: 20000; alternating total is not disclosed in the paper"
    )
    parser.add_argument("--batch-size", type=int, help="Global effective batch; defaults to 64 frames or 32 sequences")
    parser.add_argument(
        "--supervise-stride",
        type=int,
        default=32,
        help="Execution supervision interval; paper does not specify training sampling",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--write-stride", type=int, default=16)
    parser.add_argument("--non-adaptive", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.stage == "stage1" and not (args.base_weights or args.resume):
        parser.error("Stage 1 requires --base-weights or --resume")
    if args.stage == "alternating" and not (args.init or args.resume):
        parser.error("Alternating training requires --init or --resume")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    config = MemoryConfig(write_stride=args.write_stride, adaptive=not args.non_adaptive)
    restored = (
        torch.load(args.resume or args.init, map_location="cpu", weights_only=False)
        if args.resume or args.init
        else None
    )
    if args.resume:
        config = MemoryConfig(**restored["config"])
    model = T2Mem(config, base_weights=str(args.base_weights) if args.base_weights else None).to(device)
    if args.resume:
        if restored["stage"] != args.stage:
            raise ValueError("Resume stage differs from checkpoint; use --init to transition")
        model.load_state_dict(restored["model"])
    elif args.init:
        if restored["stage"] != "stage1" or restored["step"] != 20000:
            raise ValueError("--init must be the Stage-1 step-20000 checkpoint (Appendix B)")
        model.base.load_state_dict(
            {name.removeprefix("base."): value for name, value in restored["model"].items() if name.startswith("base.")}
        )
    model.activation_checkpointing = args.activation_checkpointing
    dataset = Episodes(args.data, args.tasks)
    transform = ObservationTransform(json.loads((args.data / "norm_stats.json").read_text()), config)
    batch = args.batch_size or (64 if args.stage == "stage1" else 32)
    if batch % world:
        raise ValueError("Global batch size must be divisible by the number of workers")
    accumulation = batch // world
    optimizers = {}
    for phase, lr in [("stage1", 2e-5)] if args.stage == "stage1" else [("memory", 7.5e-5), ("policy", 3e-5)]:
        model.set_stage(phase)
        optimizers[phase] = optimizer(model, phase, lr)
    initial_step = restored["step"] if args.resume else 0
    if args.resume:
        for phase, state in restored["optimizers"].items():
            optimizers[phase].load_state_dict(state)
        rng = restored["rng"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output / "recipe.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "data": str(args.data),
                    "output": str(args.output),
                    "base_weights": str(args.base_weights),
                    "init": str(args.init),
                    "resume": str(args.resume),
                    "config": dataclasses.asdict(config),
                    "sequence": "Full chronological prefix through a sampled execution endpoint; no detach or TBPTT",
                },
                indent=2,
            )
        )
        (args.output / "norm_stats.json").write_text((args.data / "norm_stats.json").read_text())
    current_phase, runner = None, None
    task_rng = random.Random(args.seed)
    # Task schedule is shared across ranks and restored by step index.
    task_schedule = [task_rng.choice(args.tasks) for _ in range(args.updates)]
    for step in range(initial_step, args.updates):
        phase = "stage1" if args.stage == "stage1" else ("memory" if (step // 500) % 2 == 0 else "policy")
        if phase != current_phase:
            del runner
            model.set_stage(phase)
            runner = SequenceLoss(model)
            if world > 1:
                runner = DistributedDataParallel(
                    runner, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False
                )
            current_phase = phase
        opt = optimizers[phase]
        opt.zero_grad(set_to_none=True)
        task = task_schedule[step]
        total_loss = 0.0
        for micro in range(accumulation):
            episode = dataset[random.choice(dataset.by_task[task])]
            start = int(episode["execution_start"])
            if phase == "stage1":
                indices = [random.randrange(start, len(episode["actions"]))]
            else:
                endpoint = random.choice(list(range(start, len(episode["actions"]), args.supervise_stride)))
                indices = sequence_indices(episode, endpoint, config, args.supervise_stride)
            sync = runner.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
            with sync:
                loss = runner(episode, indices, transform, args.supervise_stride, device) / accumulation
                loss.backward()
            total_loss += loss.detach().item()
        opt.step()
        if rank == 0:
            print(f"step={step + 1} phase={phase} task={task} loss={total_loss:.6f}", flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.updates:
            rng = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device),
            }
            rngs = [None] * world
            if world > 1:
                dist.all_gather_object(rngs, rng)
            else:
                rngs[0] = rng
            if rank == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": dataclasses.asdict(config),
                        "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
                        "stage": args.stage,
                        "step": step + 1,
                        "rng": rngs,
                    },
                    args.output / f"step_{step + 1:06d}.pt",
                )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
