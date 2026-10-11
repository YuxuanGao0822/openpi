from dataclasses import dataclass

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


@dataclass(frozen=True)
class MemoryConfig:
    layers: int = 18
    width: int = 1024
    tokens: int = 16
    heads: int = 16
    head_dim: int = 64
    hidden_dim: int = 256
    adapter_rank: int = 16
    adaptive: bool = True
    write_stride: int = 16
    demo_write_offset: int = 2
    horizon: int = 64
    execute: int = 32
    action_dim: int = 32
    supervised_dim: int = 8
    vlm_lora_rank: int = 16
