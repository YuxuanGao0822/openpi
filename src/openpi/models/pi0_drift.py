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

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng = jax.random.split(rng, 2)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Batch size
        B = actions.shape[0]
        # Number of generated hypotheses per observation
        G = getattr(self.config, "gen_per_label", 8)
        
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
        
        # 4. Compute drift loss
        drift_temps = getattr(self.config, "drift_temps", (0.02, 0.05, 0.2))
        drift_plus_only = getattr(self.config, "drift_plus_only", False)
        drift_use_neg_only = getattr(self.config, "drift_use_neg_only", False)
        loss, info = drift_loss(
            gen=gen_actions_flat,
            fixed_pos=actions_flat,
            R_list=drift_temps,
            plus_only=drift_plus_only,
            use_neg_only=drift_use_neg_only,
        )
        
        # 5. Compute MSE error between generated actions and ground truth
        mse_error = jnp.mean(jnp.square(gen_actions_flat - actions_flat))
        aux = {"mse_error": mse_error}
        
        # 6. Broadcast loss from shape [B] to shape [B, action_horizon]
        return jnp.tile(loss[:, None], (1, self.action_horizon)), aux
