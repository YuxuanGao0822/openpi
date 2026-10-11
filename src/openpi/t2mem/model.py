"""Observation-grounded, per-layer interface alongside the native π0.5 path."""

from safetensors.torch import load_model
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
from openpi.t2mem.config import MemoryConfig
from openpi.t2mem.memory import FastMemory
from openpi.t2mem.memory import rms


class LowRank(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.float()))


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.adapter = LowRank(base.in_features, base.out_features, rank)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x.to(self.base.weight.dtype)) + self.adapter(x).to(x.dtype)


def install_vlm_lora(module, rank):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank))
        else:
            install_vlm_lora(child, rank)


def linear(layer, x):
    return layer(x.to(layer.weight.dtype))


def qkv(layer, hidden, adapter=None):
    outputs = []
    for name in ("q_proj", "k_proj", "v_proj"):
        y = linear(getattr(layer.self_attn, name), hidden)
        if adapter is not None:
            y = y + adapter[name](hidden).to(y.dtype)
        outputs.append(y.reshape(*hidden.shape[:-1], -1, layer.self_attn.head_dim).transpose(1, 2))
    return outputs


def rotate(model, query, key, positions):
    cos, sin = model.rotary_emb(query, positions)
    return modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)


def attention(query, key, value, mask, scaling):
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, 1)
    value = value.repeat_interleave(repeats, 1)
    return (
        F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None], scale=scaling)
        .transpose(1, 2)
        .flatten(2)
    )


def finish_layer(layer, hidden, attended, gate, cond):
    hidden = hidden + linear(layer.self_attn.o_proj, attended) * (1 if gate is None else gate)
    normed, gate = layer.post_attention_layernorm(hidden, cond=cond)
    update = layer.mlp(normed.to(layer.mlp.up_proj.weight.dtype))
    return hidden + update * (1 if gate is None else gate)


class InterfaceLayer(nn.Module):
    def __init__(self, config, action_layer):
        super().__init__()
        self.memory = FastMemory(config)
        self.adapters = nn.ModuleDict(
            {
                name: LowRank(config.width, getattr(action_layer.self_attn, name).out_features, config.adapter_rank)
                for name in ("q_proj", "k_proj", "v_proj")
            }
        )


class T2Mem(nn.Module):
    def __init__(self, config: MemoryConfig | None = None, *, base_weights: str | None = None):
        super().__init__()
        config = config or MemoryConfig()
        self.config = config
        self.base = PI0Pytorch(Pi0Config(pi05=True, action_horizon=config.horizon, pytorch_compile_mode=None))
        if base_weights is not None:
            load_model(self.base, base_weights)
        # Restore openpi's norm precision after loading converted BF16 weights.
        self.base.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        for module in (
            self.base.action_in_proj,
            self.base.action_out_proj,
            self.base.time_mlp_in,
            self.base.time_mlp_out,
        ):
            module.float()
        install_vlm_lora(self.vlm.layers, config.vlm_lora_rank)
        self.interface_tokens = nn.Parameter(torch.randn(config.tokens, config.width) * 0.02)
        self.interface = nn.ModuleList([InterfaceLayer(config, layer) for layer in self.expert.layers])
        self.proprio = nn.Linear(config.action_dim, config.width)
        self.stage = "stage1"
        self.activation_checkpointing = False
        self.set_stage(self.stage)

    @property
    def vlm(self):
        return self.base.paligemma_with_expert.paligemma.language_model

    @property
    def expert(self):
        return self.base.paligemma_with_expert.gemma_expert.model

    def memory_parameters(self):
        yield self.interface_tokens
        yield from self.interface.parameters()

    def policy_parameters(self):
        for name, param in self.base.named_parameters():
            if (
                ("language_model.layers." in name and ".adapter." in name)
                or "gemma_expert." in name
                or name.startswith(("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out."))
            ):
                yield param
        yield from self.proprio.parameters()

    def set_stage(self, stage):
        self.stage = stage
        self.requires_grad_(requires_grad=False)
        for p in self.memory_parameters():
            p.requires_grad_(stage in ("memory", "joint"))
        for p in self.policy_parameters():
            p.requires_grad_(stage in ("stage1", "policy", "joint"))
        if stage == "stage1":
            self.proprio.requires_grad_(requires_grad=False)

    def reset_memory(self, batch_size):
        return tuple(layer.memory.reset(batch_size) for layer in self.interface)

    def _call(self, fn, *args):
        if self.activation_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def embed(self, observation):
        processed = preprocess_observation_pytorch(
            observation, train=False, image_keys=("base_0_rgb", "left_wrist_0_rgb")
        )
        prefix, valid, _ = self.base.embed_prefix(
            list(processed.images.values()),
            list(processed.image_masks.values()),
            processed.tokenized_prompt,
            processed.tokenized_prompt_mask,
        )
        return prefix.to(self.vlm.layers[0].self_attn.q_proj.weight.dtype), valid

    def encode(self, observation, fast_state, write_mask, *, retain_graph):
        """Compute VL caches and interface contexts once per physical observation.

        Context l is the interface entering l. The update produced at l is first
        consumed at l+1. Fast-state writes never change this observation's reads.
        """
        prefix, valid = self.embed(observation)
        positions = valid.long().cumsum(-1) - 1
        batch = prefix.shape[0]
        enabled = self.stage != "stage1"
        interface = self.interface_tokens[None].expand(batch, -1, -1)
        interface_positions = (
            positions.max(-1).values[:, None] + 1 + torch.arange(self.config.tokens, device=prefix.device)
        )
        vl_mask = valid[:, None, :].expand(-1, prefix.shape[1], -1)
        caches, updated = [], []
        for index, (vl_layer, ae_layer, block) in enumerate(
            zip(self.vlm.layers, self.expert.layers, self.interface, strict=True)
        ):

            def encode_layer(prefix, interface, state, vl_layer=vl_layer, ae_layer=ae_layer, block=block):
                normed, gate = vl_layer.input_layernorm(prefix)
                query, key, value = qkv(vl_layer, normed)
                query, key = rotate(self.vlm, query, key, positions)
                next_prefix = finish_layer(
                    vl_layer, prefix, attention(query, key, value, vl_mask, vl_layer.self_attn.scaling), gate, None
                )
                if not enabled:
                    return next_prefix, interface, state, (key, value, None, None)
                # Independent interface normalization: no flow-time/proprio/action conditioning.
                interface_norm = rms(interface.float())
                iq, ik, iv = qkv(ae_layer, interface_norm, block.adapters)
                iq, ik = rotate(self.expert, iq, ik, interface_positions)
                summary = linear(
                    ae_layer.self_attn.o_proj,
                    attention(
                        iq, key, value, valid[:, None, :].expand(-1, self.config.tokens, -1), ae_layer.self_attn.scaling
                    ),
                ).float()
                mq, mk, mv = block.memory.bindings(summary)
                readout = block.memory.read(state, mq)
                fused = summary + block.memory.alpha.tanh() * readout
                next_interface = interface.float() + fused
                next_interface = (
                    next_interface + ae_layer.mlp(rms(next_interface).to(ae_layer.mlp.up_proj.weight.dtype)).float()
                )
                next_state = block.memory.write(state, mk, mv, write_mask, retain_graph=retain_graph)
                return next_prefix, next_interface, next_state, (key, value, ik, iv)

            prefix, interface, new_state, cache = self._call(encode_layer, prefix, interface, fast_state[index])
            caches.append(cache)
            updated.append(new_state)
        return (tuple(caches), valid), tuple(updated)

    def velocity(self, encoded, state, noisy_actions, time):
        caches, valid = encoded
        suffix, _, _, cond = self.base.embed_suffix(state, noisy_actions, time)
        enabled = self.stage != "stage1"
        if enabled:
            suffix = torch.cat((self.proprio(state.float())[:, None], suffix), 1)
        suffix = suffix.to(self.expert.layers[0].self_attn.q_proj.weight.dtype)
        prefix_len = valid.sum(-1)[:, None]
        positions = (
            prefix_len + (self.config.tokens if enabled else 0) + torch.arange(suffix.shape[1], device=suffix.device)
        )
        suffix_mask = torch.ones(
            state.shape[0], suffix.shape[1], suffix.shape[1], dtype=torch.bool, device=state.device
        )
        if enabled:
            # Native prefix-LM suffix: proprio cannot see future action tokens.
            suffix_mask[:, 0, 1:] = False
        for layer, cache in zip(self.expert.layers, caches, strict=True):

            def decode_layer(suffix, layer=layer, cache=cache):
                normed, gate = layer.input_layernorm(suffix, cond=cond)
                query, key, value = qkv(layer, normed)
                query, key = rotate(self.expert, query, key, positions)
                vl_key, vl_value, interface_key, interface_value = cache
                keys, values = [vl_key], [vl_value]
                masks = [valid[:, None, :].expand(-1, suffix.shape[1], -1)]
                if enabled:
                    keys.append(interface_key)
                    values.append(interface_value)
                    masks.append(
                        torch.ones(
                            state.shape[0], suffix.shape[1], self.config.tokens, dtype=torch.bool, device=state.device
                        )
                    )
                keys.append(key)
                values.append(value)
                masks.append(suffix_mask)
                attended = attention(
                    query, torch.cat(keys, 2), torch.cat(values, 2), torch.cat(masks, -1), layer.self_attn.scaling
                )
                return finish_layer(layer, suffix, attended, gate, cond)

            suffix = self._call(decode_layer, suffix)
        suffix, _ = self.expert.norm(suffix, cond=cond)
        return self.base.action_out_proj(suffix[:, -self.config.horizon :].float())

    def action_loss(self, encoded, observation, actions, action_mask):
        noise = self.base.sample_noise(actions.shape, actions.device)
        time = self.base.sample_time(actions.shape[0], actions.device)
        mask = action_mask.to(actions.dtype)
        noisy = (time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions) * mask
        predicted = self.velocity(encoded, observation.state, noisy, time)
        error = (predicted - (noise - actions)).square() * mask
        return error.sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)

    @torch.no_grad()
    def sample_actions(self, observation, fast_state, write_mask, *, num_steps=10):
        encoded, updated = self.encode(observation, fast_state, write_mask, retain_graph=False)
        b = observation.state.shape[0]
        x = self.base.sample_noise((b, self.config.horizon, self.config.action_dim), observation.state.device)
        x[..., self.config.supervised_dim :] = 0
        # Frozen pre-write caches/readouts are reused throughout denoising.
        for step in range(num_steps):
            time = torch.full((b,), 1.0 - step / num_steps, device=x.device)
            x = x - self.velocity(encoded, observation.state, x, time) / num_steps
            x[..., self.config.supervised_dim :] = 0
        return x, updated
