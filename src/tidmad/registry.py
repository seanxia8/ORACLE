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
