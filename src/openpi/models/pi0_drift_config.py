import dataclasses
import flax.nnx as nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at

@dataclasses.dataclass(frozen=True)
class Pi0DriftConfig(Pi0Config):
    gen_per_label: int = 8
    drift_temps: tuple[float, ...] = (0.02, 0.05, 0.2)
    drift_plus_only: bool = False
    drift_use_neg_only: bool = False

    # --- Language-Conditioned Drift Field (LCDF) ---
    # Scheme C: Drift-Native CFG — drop language condition during training,
    # apply classifier-free guidance at inference.
    cfg_drop_rate: float = 0.0  # Probability of dropping language tokens during training (0.0 = disabled)
    cfg_scale: float = 1.0  # Guidance scale at inference (1.0 = no guidance, >1.0 = amplified)

    # Scheme B: In-batch cross-language negative sampling — use GT actions
    # from other language instructions in the same batch as drift negatives.
    use_cross_lang_negatives: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_DRIFT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Drift":
        from openpi.models.pi0_drift import Pi0Drift

        return Pi0Drift(self, rngs=nnx.Rngs(rng))
