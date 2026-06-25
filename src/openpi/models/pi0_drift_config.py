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

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_DRIFT

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Drift":
        from openpi.models.pi0_drift import Pi0Drift

        return Pi0Drift(self, rngs=nnx.Rngs(rng))
