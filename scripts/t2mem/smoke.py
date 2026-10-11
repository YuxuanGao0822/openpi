"""GPU model checks for the server: parameter ledger, groups, backward, rollout."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch

from openpi.t2mem.config import TASKS
from openpi.t2mem.data import Episodes
from openpi.t2mem.data import ObservationTransform
from openpi.t2mem.model import T2Mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--task", choices=TASKS, default="VideoUnmask")
    parser.add_argument("--activation-offload", action="store_true")
    args = parser.parse_args()
    model = T2Mem(base_weights=args.base_weights).cuda()
    model.activation_checkpointing = args.activation_checkpointing
    added = sum(p.numel() for p in model.memory_parameters()) + sum(p.numel() for p in model.proprio.parameters())
    assert added == 67_898_386, added
    dataset = Episodes(args.data, [args.task])
    episode = dataset[0]
    transform = ObservationTransform(json.loads((args.data / "norm_stats.json").read_text()))
    start = int(episode["execution_start"])
    for phase in ("stage1", "memory", "policy"):
        model.set_stage(phase)
        model.zero_grad(set_to_none=True)
        model.train()
        state = model.reset_memory(1)
        # A preceding observation writes; a later frame supplies the action loss.
        # Counting tasks have no demonstration prefix (execution_start == 0).
        first = min(model.config.demo_write_offset, start - 1) if start > 0 else 0
        target = (
            start
            if phase == "stage1"
            else min(max(start, first + model.config.write_stride), len(episode["actions"]) - 1)
        )
        indices = [target] if phase == "stage1" else [first, target]
        torch.cuda.reset_peak_memory_stats()
        offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if args.activation_offload else nullcontext()
        with offload:
            for index in indices:
                obs = transform.observation(
                    episode["front"][index],
                    episode["wrist"][index],
                    episode["state"][index],
                    str(episode["prompt"]),
                    stage1=phase == "stage1",
                    device="cuda",
                )
                encoded, state = model.encode(
                    obs, state, torch.tensor([phase != "stage1"], device="cuda"), retain_graph=True
                )
            actions, mask = transform.chunk(episode, target, "cuda")
            loss = model.action_loss(encoded, obs, actions, mask).mean()
        loss.backward()
        assert torch.isfinite(loss)
        active = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert active
        assert all(torch.isfinite(p.grad).all() for p in active)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        if phase == "memory":
            assert model.interface[0].memory.value.weight.grad.abs().sum() > 0
        print(
            phase,
            "loss",
            loss.item(),
            "active gradient tensors",
            len(active),
            "peak_allocated_gib",
            torch.cuda.max_memory_allocated() / 2**30,
        )
    model.eval()
    model.requires_grad_(requires_grad=False)
    initial = model.reset_memory(1)
    actions, updated = model.sample_actions(obs, initial, torch.tensor([True], device="cuda"))
    assert actions.shape == (1, 64, 32)
    assert torch.isfinite(actions).all()
    assert all(not weight.requires_grad and weight.dtype == torch.float32 for layer in updated for weight in layer)
    print("GPU smoke checks passed; added parameters", added)


if __name__ == "__main__":
    main()
