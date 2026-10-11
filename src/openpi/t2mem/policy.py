"""Single-episode policy with explicit reset / observe / infer operations."""

import json
from pathlib import Path

import torch

from openpi.t2mem.config import MemoryConfig
from openpi.t2mem.data import ObservationTransform
from openpi.t2mem.model import T2Mem


class T2MemPolicy:
    def __init__(self, checkpoint, *, device="cuda", seed=0, disable_writes=False, num_steps=10):
        checkpoint = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.model = T2Mem(MemoryConfig(**saved["config"])).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.set_stage("stage1" if saved["stage"] == "stage1" else "policy")
        self.model.requires_grad_(requires_grad=False)
        self.model.eval()
        self.transform = ObservationTransform(
            json.loads((checkpoint.parent / "norm_stats.json").read_text()), self.model.config
        )
        self.seed = seed
        self.disable_writes = disable_writes
        self.num_steps = num_steps
        self.reset(seed)

    def reset(self, seed=None):
        torch.manual_seed(self.seed if seed is None else seed)
        self.memory = self.model.reset_memory(1)

    def _observation(self, request):
        return self.transform.observation(
            request["front"],
            request["wrist"],
            request["state"],
            request["prompt"],
            stage1=self.model.stage == "stage1",
            device=self.device,
        )

    def infer(self, request):
        operation = request.get("op", "infer")
        if operation == "reset":
            self.reset(request.get("seed"))
            return {"reset_finished": True}
        obs = self._observation(request)
        write = request["write"] and not self.disable_writes and self.model.stage != "stage1"
        mask = torch.tensor([write], device=self.device)
        if operation == "observe":
            with torch.no_grad():
                _, self.memory = self.model.encode(obs, self.memory, mask, retain_graph=False)
            return {"observed": True}
        actions, self.memory = self.model.sample_actions(obs, self.memory, mask, num_steps=self.num_steps)
        actions = actions[0, :, : self.model.config.supervised_dim].cpu().numpy()
        return {"actions": self.transform.unnormalize_actions(actions).astype("float32")}
