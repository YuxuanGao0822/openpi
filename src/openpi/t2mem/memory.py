"""FP32 nonlinear fast weights; T²Mem equations (10)--(15).

All operations are functional: checkpoint recomputation never commits a write.
The exact MLP gradient is explicit, retaining second-order outer derivatives.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.t2mem.config import MemoryConfig

FastState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def rope(x):
    """Interleaved RoPE on interface positions; x is B,H,N,D."""
    positions = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)
    inv_freq = 10000.0 ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype) / x.shape[-1])
    angle = positions[:, None] * inv_freq[None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * angle.cos() - odd * angle.sin(), even * angle.sin() + odd * angle.cos()), -1).flatten(-2)


def mlp(state: FastState, x):
    w1, b1, w2, b2 = state
    pre = torch.einsum("bhnd,bhdf->bhnf", x, w1) + b1[:, :, None]
    hidden = F.gelu(pre, approximate="none")
    out = torch.einsum("bhnf,bhfd->bhnd", hidden, w2) + b2[:, :, None]
    return out, pre, hidden


def gelu_derivative(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) + x * torch.exp(-x.square() / 2.0) / math.sqrt(2.0 * math.pi)


def head_sum(xs):
    return sum(x.flatten(2).sum(-1) for x in xs)


def scale_heads(x, factor):
    return x * factor.reshape(*factor.shape, *([1] * (x.ndim - 2)))


class FastMemory(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        h, d, f = config.heads, config.head_dim, config.hidden_dim
        self.initial = nn.ParameterList(
            [
                nn.Parameter(torch.randn(h, d, f) * 0.02),
                nn.Parameter(torch.zeros(h, f)),
                nn.Parameter(torch.randn(h, f, d) * 0.02),
                nn.Parameter(torch.zeros(h, d)),
            ]
        )
        self.query = nn.Linear(config.width, config.width)
        self.key = nn.Linear(config.width, config.width)
        self.value = nn.Linear(config.width, config.width)
        self.alpha = nn.Parameter(torch.full((config.width,), math.atanh(0.01)))
        self.rho = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def reset(self, batch_size: int) -> FastState:
        return tuple(p.float().unsqueeze(0).expand(batch_size, *p.shape) for p in self.initial)

    def bindings(self, summary):
        b, n, _ = summary.shape

        def project(layer):
            return rms(layer(summary.float()).reshape(b, n, self.config.heads, self.config.head_dim).transpose(1, 2))

        return rope(project(self.query)), rope(project(self.key)), project(self.value)

    def read(self, state, query):
        output = mlp(state, query)[0]
        return output.transpose(1, 2).flatten(2)

    def write(self, state: FastState, key, value, write_mask, *, retain_graph: bool) -> FastState:
        if not bool(write_mask.any()):
            return state if retain_graph else tuple(w.detach() for w in state)
        # Sum per-example losses; the H*N*D denominator matches equation (10).
        with torch.autocast(device_type=key.device.type, enabled=False):
            output, pre, hidden = mlp(state, key.float())
            residual = output - value.float()
            denominator = self.config.heads * key.shape[-2] * self.config.head_dim
            dy = 2.0 * residual / denominator
            w2 = state[2]
            dpre = torch.einsum("bhnd,bhfd->bhnf", dy, w2) * gelu_derivative(pre)
            gradients = (
                torch.einsum("bhnd,bhnf->bhdf", key, dpre),
                dpre.sum(-2),
                torch.einsum("bhnf,bhnd->bhfd", hidden, dy),
                dy.sum(-2),
            )
            # Calibration and adaptive scaling are explicitly stop-gradient.
            with torch.no_grad():
                g = tuple(x.detach() for x in gradients)
                delta = tuple(w.detach() - p[None] for w, p in zip(state, self.initial, strict=True))
                gnorm = head_sum(tuple(x.square() for x in g))
                dnorm = head_sum(tuple(x.square() for x in delta))
                alignment = -head_sum(tuple(x * d for x, d in zip(g, delta, strict=True))) / (
                    gnorm * dnorm
                ).sqrt().clamp_min(1e-12)
                novelty = 1.0 - alignment.clamp(0.0, 1.0)
                novelty = torch.where(dnorm <= 1e-24, 1.0, novelty)
                novelty = torch.where(gnorm <= 1e-24, 0.0, novelty)
                ratio = residual.square().mean((-2, -1)).sqrt() / value.square().mean((-2, -1)).sqrt().clamp_min(1e-8)
                z = novelty * (1.0 + ratio.clamp(0.0, 1.0)) / 2.0
                beta = 0.1 + 0.9 * (3.0 * z.square() - 2.0 * z.pow(3)) if self.config.adaptive else torch.ones_like(z)
                # Exact J_f g for the two-layer GeLU network.
                directional_pre = torch.einsum("bhnd,bhdf->bhnf", key, g[0]) + g[1][:, :, None]
                jg = torch.einsum("bhnf,bhfd->bhnd", directional_pre * gelu_derivative(pre), w2)
                jg = jg + torch.einsum("bhnf,bhfd->bhnd", hidden, g[2]) + g[3][:, :, None]
                curvature = 2.0 * jg.square().sum((-2, -1)) / denominator
                q = (gnorm / curvature.clamp_min(1e-30)).clamp(1e-12, 1e3)
                trial = tuple(w - scale_heads(grad, q) for w, grad in zip(state, g, strict=True))
                loss_head = residual.square().sum((-2, -1)) / denominator
                trial_loss = (mlp(trial, key)[0] - value).square().sum((-2, -1)) / denominator
                b = (trial_loss - loss_head + gnorm * q) / q.square()
                length = torch.where(b > 0, gnorm / (2.0 * b).clamp_min(1e-30), q).clamp_max(1e3)
            eta = 0.5 * F.softplus(self.rho) * length * beta * write_mask[:, None]
            updated = tuple(w - scale_heads(grad, eta) for w, grad in zip(state, gradients, strict=True))
        return updated if retain_graph else tuple(w.detach() for w in updated)
