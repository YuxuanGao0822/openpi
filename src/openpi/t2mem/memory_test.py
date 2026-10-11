"""Server-side numerical checks of the paper update, without loading π0.5."""

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.t2mem.config import MemoryConfig
from openpi.t2mem.memory import FastMemory
from openpi.t2mem.memory import head_sum
from openpi.t2mem.memory import mlp
from openpi.t2mem.memory import scale_heads


def small_memory(*, adaptive=True):
    config = MemoryConfig(layers=1, width=8, tokens=3, heads=2, head_dim=4, hidden_dim=7, adaptive=adaptive)
    return FastMemory(config)


def reference_write(memory, state, key, value):
    """Autograd/JVP reference independent of the explicit GeLU gradient."""
    denom = memory.config.heads * key.shape[-2] * key.shape[-1]
    out = mlp(state, key)[0]
    residual = out - value
    loss = residual.square().sum() / denom
    grads = torch.autograd.grad(loss, state, create_graph=True)
    with torch.no_grad():
        g = tuple(x.detach() for x in grads)
        delta = tuple(w - p[None] for w, p in zip(state, memory.initial, strict=True))
        gnorm = head_sum(tuple(x.square() for x in g))
        dnorm = head_sum(tuple(x.square() for x in delta))
        align = -head_sum(tuple(x * d for x, d in zip(g, delta, strict=True))) / (gnorm * dnorm).sqrt().clamp_min(1e-12)
        novelty = 1 - align.clamp(0, 1)
        novelty = torch.where(dnorm <= 1e-24, 1.0, novelty)
        novelty = torch.where(gnorm <= 1e-24, 0.0, novelty)
        r = (residual.square().mean((-2, -1)).sqrt() / value.square().mean((-2, -1)).sqrt().clamp_min(1e-8)).clamp(0, 1)
        z = novelty * (1 + r) / 2
        beta = 0.1 + 0.9 * (3 * z.square() - 2 * z.pow(3)) if memory.config.adaptive else torch.ones_like(z)
    # JVP is computed via autodiff, not the explicit implementation's derivative.
    _, jg = torch.autograd.functional.jvp(lambda *weights: mlp(weights, key)[0], state, g)
    with torch.no_grad():
        c = 2 * jg.square().sum((-2, -1)) / denom
        q = (gnorm / c.clamp_min(1e-30)).clamp(1e-12, 1e3)
        trial = tuple(w - scale_heads(grad, q) for w, grad in zip(state, g, strict=True))
        trial_loss = (mlp(trial, key)[0] - value).square().sum((-2, -1)) / denom
        current_loss = residual.square().sum((-2, -1)) / denom
        b = (trial_loss - current_loss + gnorm * q) / q.square()
        length = torch.where(b > 0, gnorm / (2 * b).clamp_min(1e-30), q).clamp_max(1e3)
    eta = 0.5 * F.softplus(memory.rho) * length * beta
    return tuple(w - scale_heads(grad, eta) for w, grad in zip(state, grads, strict=True))


def test_exact_write_and_outer_gradients():
    torch.manual_seed(2)
    memory = small_memory()
    state = tuple(p[None] + torch.randn_like(p[None]) * 0.1 for p in memory.initial)
    key = torch.randn(1, 2, 3, 4, requires_grad=True)
    value = torch.randn(1, 2, 3, 4, requires_grad=True)
    actual = memory.write(state, key, value, torch.tensor([True]), retain_graph=True)
    expected = reference_write(memory, state, key, value)
    for result, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, ref, atol=2e-6, rtol=2e-5)
    objective = sum(x.square().sum() for x in actual)
    reference_objective = sum(x.square().sum() for x in expected)
    inputs = (key, value, memory.rho, *memory.initial)
    actual_grad = torch.autograd.grad(objective, inputs, retain_graph=True)
    reference_grad = torch.autograd.grad(reference_objective, inputs)
    for result, ref in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(result, ref, atol=1e-5, rtol=5e-5)


def test_episode_reset_and_write_mask():
    torch.manual_seed(3)
    memory = small_memory()
    state = memory.reset(2)
    original = tuple(x.clone() for x in state)
    key, value = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)
    updated = memory.write(state, key, value, torch.tensor([True, False]), retain_graph=False)
    for before, after in zip(state, updated, strict=True):
        torch.testing.assert_close(before[1], after[1])
        assert not after.requires_grad
    assert any(not torch.equal(before[0], after[0]) for before, after in zip(state, updated, strict=True))
    for reset, before in zip(memory.reset(2), original, strict=True):
        torch.testing.assert_close(reset, before)
    query = torch.randn(2, 2, 3, 4)
    torch.testing.assert_close(memory.read(state, query), memory.read(original, query))


def test_earlier_write_receives_action_gradient():
    torch.manual_seed(4)
    memory = small_memory()
    key = torch.randn(1, 2, 3, 4, requires_grad=True)
    value = torch.randn(1, 2, 3, 4, requires_grad=True)
    updated = memory.write(memory.reset(1), key, value, torch.tensor([True]), retain_graph=True)
    readout = memory.read(updated, torch.randn(1, 2, 3, 4))
    key_grad, value_grad, rho_grad = torch.autograd.grad(readout.square().sum(), (key, value, memory.rho))
    assert key_grad.abs().sum() > 0
    assert value_grad.abs().sum() > 0
    assert rho_grad.abs() > 0


def test_nonadaptive_keeps_curvature_and_fp32():
    torch.manual_seed(5)
    memory = small_memory(adaptive=False)
    state = memory.reset(1)
    key, value = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    actual = memory.write(state, key, value, torch.tensor([True]), retain_graph=True)
    expected = reference_write(memory, state, key, value)
    for result, ref in zip(actual, expected, strict=True):
        assert result.dtype == torch.float32
        torch.testing.assert_close(result, ref, atol=2e-6, rtol=2e-5)


def test_reference_memory_parameter_counts():
    config = MemoryConfig()
    memory = FastMemory(config)
    assert sum(p.numel() for p in memory.initial) * config.layers == 9_529_344
    assert (
        sum(p.numel() for layer in (memory.query, memory.key, memory.value) for p in layer.parameters()) * config.layers
        == 56_678_400
    )
