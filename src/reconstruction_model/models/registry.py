"""Lazy registry for reconstruction model architecture variants."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class ModelEntry:
    """Metadata needed to lazily construct a model variant."""

    name: str
    module: str
    model_class: str = "Transformer"
    config_class: str = "TransformerConfig"
    source: str = ""
    description: str = ""
    output_signature: str = ""
    requires_pairwise_cache: bool = False
    optional_dependencies: tuple[str, ...] = ()


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "current_compact": ModelEntry(
        name="current_compact",
        module="reconstruction_model.models.current_compact",
        source="current project: reconstruction_model/model.py",
        description="Compact temporal/channel Transformer used by the current root training pipeline.",
        output_signature="spatial_pred, energy_pred",
    ),
    "tidmad_stft": ModelEntry(
        name="tidmad_stft",
        module="tidmad.model",
        model_class="TidmadTransformer",
        config_class="TransformerConfig",
        source="tidmad package: current_compact backbone + reconstruction head (PLAN_04)",
        description="Band-frame reconstruction Transformer for the TIDMAD benchmark; reconstruction head instead of spatial/energy heads.",
        output_signature="reconstructed_spectrogram",
        requires_pairwise_cache=False,
    ),
    "original": ModelEntry(
        name="original",
        module="reconstruction_model.models.original",
        source="DELight_reconstruction-dev1: reconstruction/models/model_original.py",
        description="Legacy baseline Transformer with temporal blocks, channel blocks, and regression/classification heads.",
        output_signature="spatial_pred, energy_pred, class_logits",
        requires_pairwise_cache=True,
    ),
    "pairwise": ModelEntry(
        name="pairwise",
        module="reconstruction_model.models.pairwise",
        source="DELight_reconstruction-dev1: reconstruction/models/model_pairwise.py",
        description="Pairwise spatial Transformer that updates channel-pair features and uses them as attention bias.",
        output_signature="spatial_pred, energy_pred, class_logits",
        requires_pairwise_cache=True,
    ),
    "pairwise_channel_masking": ModelEntry(
        name="pairwise_channel_masking",
        module="reconstruction_model.models.pairwise_channel_masking",
        source="DELight_reconstruction-dev1: reconstruction/models/model_pairwise_channel_masking.py",
        description="Pairwise model with stochastic top/bottom channel masking and inference mask helpers.",
        output_signature="spatial_pred, energy_pred, class_logits",
        requires_pairwise_cache=True,
    ),
    "triangular_pairwise": ModelEntry(
        name="triangular_pairwise",
        module="reconstruction_model.models.triangular_pairwise",
        source="DELight_reconstruction-dev1: reconstruction/models/model_triangular_pairwise.py",
        description="Pairwise model with AlphaFold-style triangular multiplication/attention updates.",
        output_signature="spatial_pred, energy_pred, class_logits",
        requires_pairwise_cache=True,
    ),
    "cnn_transformer": ModelEntry(
        name="cnn_transformer",
        module="reconstruction_model.models.cnn_transformer",
        source="current src copy: reconstruction_model/model_transformer_cnn.py",
        description="Grouped-Conv1D feature extractor followed by spatial Transformer blocks.",
        output_signature="spatial_pred, energy_pred, class_logits",
        requires_pairwise_cache=True,
    ),
    "integration_classifier": ModelEntry(
        name="integration_classifier",
        module="reconstruction_model.models.integration_classifier",
        model_class="PositionClassifier",
        config_class="IntegrationConfig",
        source="DELight_reconstruction-dev1: reconstruction/models/integration_classifier.py",
        description="XGBoost baseline over channel-integrated traces; not a Transformer.",
        output_signature="XGBoost classifier predictions and metrics",
        optional_dependencies=("xgboost", "scikit-learn", "seaborn", "tqdm"),
    ),
}


def available_models() -> tuple[str, ...]:
    """Return available model variant names."""

    return tuple(MODEL_REGISTRY)


def get_model_entry(name: str) -> ModelEntry:
    """Return registry metadata for one model variant."""

    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(available_models())
        raise KeyError(f"Unknown model variant {name!r}. Available: {valid}") from exc


def load_model_objects(name: str) -> tuple[type[Any], type[Any]]:
    """Import and return ``(Transformer, TransformerConfig)``-style classes."""

    entry = get_model_entry(name)
    module = import_module(entry.module)
    return getattr(module, entry.model_class), getattr(module, entry.config_class)


def create_model(name: str, **config_overrides: Any):
    """Instantiate a model variant with optional config overrides."""

    model_cls, config_cls = load_model_objects(name)
    config = config_cls(**config_overrides)
    return model_cls(config), config
