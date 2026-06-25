import jax
import jax.numpy as jnp

def _cdist(x, y, eps=1e-8):
    """Pairwise L2 distance: [B, N, D] x [B, M, D] -> [B, N, M]."""
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))

def drift_loss(gen, fixed_pos, fixed_neg=None, weight_gen=None, weight_pos=None,
               weight_neg=None, R_list=(0.02, 0.05, 0.2)):
    """JAX implementation of the debiased Sinkhorn/W-Flow loss."""
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = jnp.zeros((B, 0, S), dtype=gen.dtype)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = jnp.ones((B, C_g), dtype=gen.dtype)
    if weight_pos is None:
        weight_pos = jnp.ones((B, C_p), dtype=gen.dtype)
    if weight_neg is None:
        weight_neg = jnp.ones((B, C_n), dtype=gen.dtype)

    gen = gen.astype(jnp.float32)
    fixed_pos = fixed_pos.astype(jnp.float32)
    fixed_neg = fixed_neg.astype(jnp.float32)
    weight_gen = weight_gen.astype(jnp.float32)
    weight_pos = weight_pos.astype(jnp.float32)
    weight_neg = weight_neg.astype(jnp.float32)

    old_gen = jax.lax.stop_gradient(gen)
    targets = jnp.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
    targets_w = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)

    # Goal computation (no gradients)
    dist = _cdist(old_gen, targets)
    weighted_dist = dist * targets_w[:, None, :]
    scale = weighted_dist.mean() / targets_w.mean()

    scale_inputs = jnp.clip(scale / (S ** 0.5), a_min=1e-3)
    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs

    dist_normed = dist / jnp.clip(scale, a_min=1e-3)

    # Mask self-connections for gen block
    mask_val = 100.0
    diag_mask = jnp.eye(C_g, dtype=gen.dtype)
    block_mask = jnp.pad(diag_mask, ((0, 0), (0, C_n + C_p)))
    block_mask = block_mask[None, :, :]
    dist_normed = dist_normed + block_mask * mask_val

    force_across_R = jnp.zeros_like(old_gen_scaled)
    info = {"scale": scale}

    for R in R_list:
        logits = -dist_normed / R

        # softmax along axis=-1 and axis=-2
        affinity = jax.nn.softmax(logits, axis=-1)
        aff_transpose = jax.nn.softmax(logits, axis=-2)
        affinity = jnp.sqrt(jnp.clip(affinity * aff_transpose, a_min=1e-6))

        affinity = affinity * targets_w[:, None, :]

        split_idx = C_g + C_n
        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]

        sum_pos = aff_pos.sum(axis=-1, keepdims=True)
        r_coeff_neg = -aff_neg * sum_pos
        sum_neg = aff_neg.sum(axis=-1, keepdims=True)
        r_coeff_pos = aff_pos * sum_neg

        R_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)

        total_force_R = jnp.einsum("biy,byx->bix", R_coeff, targets_scaled)

        total_coeffs = R_coeff.sum(axis=-1)
        total_force_R = total_force_R - total_coeffs[:, :, None] * old_gen_scaled

        f_norm_val = jnp.mean(jnp.square(total_force_R))
        info[f"loss_{R}"] = f_norm_val

        force_scale = jnp.sqrt(jnp.clip(f_norm_val, a_min=1e-8))
        force_across_R = force_across_R + total_force_R / force_scale

    goal_scaled = old_gen_scaled + force_across_R

    # Stop gradient on target variables
    goal_scaled = jax.lax.stop_gradient(goal_scaled)
    scale_inputs = jax.lax.stop_gradient(scale_inputs)

    # Loss with gradients through gen
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = jnp.mean(jnp.square(diff), axis=(-1, -2))

    # average the info dict entries
    info = {k: jnp.mean(v) for k, v in info.items()}

    return loss, info
