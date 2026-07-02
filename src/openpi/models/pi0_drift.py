import einops
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.models.drifting_util import drift_loss
from openpi.shared import array_typing as at
from openpi.models.pi0_drift_config import Pi0DriftConfig


class Pi0Drift(Pi0):
    def __init__(self, config: Pi0DriftConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.config = config

    def _maybe_drop_language(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, train: bool
    ) -> _model.Observation:
        """LCDF Scheme C: randomly drop language tokens during training for CFG.

        With probability cfg_drop_rate, replaces the tokenized prompt with zeros
        and the prompt mask with False, simulating the unconditional case.
        This allows CFG-style guidance at inference time.
        """
        cfg_drop_rate = self.config.cfg_drop_rate
        if not train or cfg_drop_rate <= 0.0 or observation.tokenized_prompt is None:
            return observation

        B = observation.state.shape[0]
        # Per-sample Bernoulli mask: True where we DROP the language condition
        drop_mask = jax.random.bernoulli(rng, cfg_drop_rate, (B,))
        # Expand for broadcasting: [B, 1] for sequence dim
        drop_mask_expanded = drop_mask[:, None]

        # Zero out prompt tokens and mask where dropped
        new_prompt = jnp.where(
            drop_mask_expanded,
            jnp.zeros_like(observation.tokenized_prompt),
            observation.tokenized_prompt,
        )
        new_prompt_mask = jnp.where(
            drop_mask_expanded,
            jnp.zeros_like(observation.tokenized_prompt_mask),
            observation.tokenized_prompt_mask,
        )

        return _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=new_prompt,
            tokenized_prompt_mask=new_prompt_mask,
        )

    def _build_cross_lang_negatives(
        self, actions: _model.Actions, observation: _model.Observation
    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
        """LCDF Scheme B: collect GT actions from other samples in the batch as negatives.

        For each sample i, actions from all other samples j (j != i) serve as
        semantically different (cross-language) negatives for the drift field.
        If they share the same prompt, we mask them out by returning a weight of 0.0.

        Args:
            actions: [B, action_horizon, action_dim] ground truth actions.
            observation: The observation dict containing prompts.

        Returns:
            cross_neg: [B, B-1, D] flattened cross-language negative actions, or None.
            neg_weights: [B, B-1] mask of weights (0.0 for same prompt/self, 1.0 for valid negatives).
        """
        if not self.config.use_cross_lang_negatives:
            return None, None

        B = actions.shape[0]
        D = self.action_horizon * self.action_dim
        actions_flat = actions.reshape((B, D))

        # 1. Gather other actions
        all_actions = jnp.tile(actions_flat[None, :, :], (B, 1, 1))

        # Vectorized gather indices of shape [B, B-1]
        r = jnp.arange(B)[:, None]
        c = jnp.arange(B - 1)[None, :]
        gather_indices = c + (c >= r)

        cross_neg = jnp.take_along_axis(all_actions, gather_indices[:, :, None], axis=1)

        # 2. Compute dynamic weights based on prompt equality
        prompt = observation.tokenized_prompt
        if prompt is not None:
            # Gather corresponding prompts for comparison
            # prompt has shape [B, L], gathered_prompts has shape [B, B-1, L]
            gathered_prompts = jnp.take(prompt, gather_indices, axis=0)
            source_prompts = prompt[:, None, :] # [B, 1, L]
            
            # Check if prompts are identical along the token sequence axis (-1)
            is_same_prompt = jnp.all(gathered_prompts == source_prompts, axis=-1)
            # Weight is 0.0 if prompts are the same, otherwise 1.0
            neg_weights = jnp.where(is_same_prompt, 0.0, 1.0)
        else:
            # Unconditional case, all gathered negatives are valid (weight 1.0)
            neg_weights = jnp.ones((B, B - 1), dtype=jnp.float32)

        return cross_neg, neg_weights

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, cfg_drop_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # LCDF Scheme C: conditionally drop language tokens during training
        observation = self._maybe_drop_language(cfg_drop_rng, observation, train=train)

        # Batch size
        B = actions.shape[0]
        # Number of generated hypotheses per observation
        G = self.config.gen_per_label

        # 1. Repeat observation components G times along batch dimension (axis 0)
        repeated_images = {k: jnp.repeat(v, G, axis=0) for k, v in observation.images.items()}
        repeated_image_masks = {k: jnp.repeat(v, G, axis=0) for k, v in observation.image_masks.items()}
        repeated_state = jnp.repeat(observation.state, G, axis=0)

        repeated_tokenized_prompt = (
            jnp.repeat(observation.tokenized_prompt, G, axis=0)
            if observation.tokenized_prompt is not None
            else None
        )
        repeated_tokenized_prompt_mask = (
            jnp.repeat(observation.tokenized_prompt_mask, G, axis=0)
            if observation.tokenized_prompt_mask is not None
            else None
        )

        repeated_observation = _model.Observation(
            images=repeated_images,
            image_masks=repeated_image_masks,
            state=repeated_state,
            tokenized_prompt=repeated_tokenized_prompt,
            tokenized_prompt_mask=repeated_tokenized_prompt_mask,
        )

        # 2. Sample noise for generating G hypotheses: shape [B * G, action_horizon, action_dim]
        noise = jax.random.normal(noise_rng, (B * G, self.action_horizon, self.action_dim))

        # 3. Evaluate model at t=1.0 to get generated actions
        time = jnp.ones((B * G,))

        # Build prefix and suffix embeddings
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(repeated_observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            repeated_observation, noise, time
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # One-step predicted action: x = noise - v_t
        gen_actions = noise - v_t

        # Flatten actions to shape [B, G, D] where D = action_horizon * action_dim
        D = self.action_horizon * self.action_dim
        gen_actions_flat = gen_actions.reshape((B, G, D))
        actions_flat = actions.reshape((B, 1, D))

        # LCDF Scheme B: build cross-language negatives from batch
        cross_neg, neg_weights = self._build_cross_lang_negatives(actions, observation)

        # 4. Compute drift loss
        loss, info = drift_loss(
            gen=gen_actions_flat,
            fixed_pos=actions_flat,
            fixed_neg=cross_neg,
            weight_neg=neg_weights,
            R_list=self.config.drift_temps,
            plus_only=self.config.drift_plus_only,
            use_neg_only=self.config.drift_use_neg_only,
        )

        # 5. Compute MSE error between generated actions and ground truth
        mse_error = jnp.mean(jnp.square(gen_actions_flat - actions_flat))
        aux = {"mse_error": mse_error}
        # Propagate per-R drift loss breakdown for detailed monitoring
        for k, v in info.items():
            aux[f"drift_{k}"] = v

        # 6. Broadcast loss from shape [B] to shape [B, action_horizon]
        return jnp.tile(loss[:, None], (1, self.action_horizon)), aux

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 1,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """One-step action generation with optional Language-Conditioned CFG.

        When cfg_scale > 1.0, performs two forward passes (conditional +
        unconditional) and applies classifier-free guidance to the velocity
        prediction before computing the final action.

        When cfg_scale == 1.0 (default), this is equivalent to a standard
        single forward pass — identical to the parent Pi0.sample_actions
        with num_steps=1.
        """
        cfg_scale = self.config.cfg_scale

        # If no CFG needed, delegate to parent's multi-step sampler (with num_steps=1)
        if cfg_scale <= 1.0:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        # --- CFG inference path (cfg_scale > 1.0) ---
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        time = jnp.ones((batch_size,))

        # Pass 1: Conditional forward (with language)
        v_cond = self._single_forward(observation, noise, time)

        # Pass 2: Unconditional forward (language dropped)
        uncond_observation = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=jnp.zeros_like(observation.tokenized_prompt) if observation.tokenized_prompt is not None else None,
            tokenized_prompt_mask=jnp.zeros_like(observation.tokenized_prompt_mask) if observation.tokenized_prompt_mask is not None else None,
        )
        v_uncond = self._single_forward(uncond_observation, noise, time)

        # CFG formula: v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

        # One-step action: x_0 = noise - v_guided (at t=1, dt=-1)
        return noise - v_guided

    def _single_forward(
        self,
        observation: _model.Observation,
        noisy_actions: at.Float[at.Array, "b ah ad"],
        time: at.Float[at.Array, " b"],
    ) -> at.Float[at.Array, "b ah ad"]:
        """Single forward pass through the VLM backbone + action head.

        Returns the velocity prediction v_t (not the final action).
        Used by both the conditional and unconditional CFG passes.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, noisy_actions, time
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return v_t
