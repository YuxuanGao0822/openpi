"""Official RoboMME HDF5 episodes, preserved as chronological NPZ arrays."""

from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.t2mem.config import MemoryConfig


def write_due(frame: int, execution_start: int, config: MemoryConfig):
    if frame < execution_start:
        return frame >= config.demo_write_offset and (frame - config.demo_write_offset) % config.write_stride == 0
    return (frame - execution_start) % config.write_stride == 0


class ObservationTransform:
    def __init__(self, norm_stats, config=None):
        self.stats = norm_stats["norm_stats"]
        self.tokenizer = PaligemmaTokenizer(max_len=200)
        self.config = config or MemoryConfig()

    def normalize(self, value, key):
        stats = self.stats[key]
        return (
            2.0 * (value - np.asarray(stats["q01"])) / (np.asarray(stats["q99"]) - np.asarray(stats["q01"]) + 1e-6)
            - 1.0
        )

    def unnormalize_actions(self, value):
        stats = self.stats["actions"]
        return (value + 1.0) / 2.0 * (np.asarray(stats["q99"]) - np.asarray(stats["q01"]) + 1e-6) + np.asarray(
            stats["q01"]
        )

    def observation(self, front, wrist, state, prompt, *, stage1, device):
        normalized = self.normalize(np.asarray(state, dtype=np.float32), "state").astype(np.float32)
        padded = np.pad(normalized, (0, self.config.action_dim - len(normalized)))
        # Stage 2 excludes proprioception from VL keys and adds a suffix token.
        tokens, mask = self.tokenizer.tokenize(prompt, padded if stage1 else None)

        def tensor(value):
            return torch.as_tensor(value, device=device)[None]

        front = np.asarray(front, dtype=np.float32) / 127.5 - 1.0
        wrist = np.asarray(wrist, dtype=np.float32) / 127.5 - 1.0
        return SimpleNamespace(
            images={
                "base_0_rgb": tensor(front).permute(0, 3, 1, 2),
                "left_wrist_0_rgb": tensor(wrist).permute(0, 3, 1, 2),
            },
            image_masks={
                key: tensor(np.asarray(value)) for key, value in (("base_0_rgb", True), ("left_wrist_0_rgb", True))
            },
            state=tensor(padded),
            tokenized_prompt=tensor(tokens).long(),
            tokenized_prompt_mask=tensor(mask).bool(),
        )

    def chunk(self, episode, index, device):
        actions = episode["actions"][index : index + self.config.horizon]
        normalized = self.normalize(actions, "actions").astype(np.float32)
        chunk = np.zeros((self.config.horizon, self.config.action_dim), dtype=np.float32)
        mask = np.zeros_like(chunk, dtype=bool)
        chunk[: len(actions), : self.config.supervised_dim] = normalized
        mask[: len(actions), : self.config.supervised_dim] = True
        return torch.as_tensor(chunk, device=device)[None], torch.as_tensor(mask, device=device)[None]


class Episodes(Dataset):
    def __init__(self, root, tasks):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.records = [record for record in self.manifest if record["task"] in tasks]
        self.by_task = {task: [i for i, record in enumerate(self.records) if record["task"] == task] for task in tasks}

    def __len__(self):
        return len(self.records)

    @lru_cache(maxsize=2)  # noqa: B019 -- one dataset instance for the lifetime of training
    def __getitem__(self, index):
        with np.load(self.root / self.records[index]["path"], allow_pickle=False) as archive:
            return dict(archive)


def sequence_indices(episode, endpoint, config, supervise_stride):
    start = int(episode["execution_start"])
    return [
        index
        for index in range(endpoint + 1)
        if write_due(index, start, config) or (index >= start and (index - start) % supervise_stride == 0)
    ]
