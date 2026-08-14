"""Model architecture catalog for DELight reconstruction.

The modules in this package are intentionally imported lazily through
``reconstruction_model.models.registry``. Some optional baselines depend on packages
such as XGBoost and scikit-learn that are not needed for normal Transformer
training.
"""

from reconstruction_model.models.registry import (
    MODEL_REGISTRY,
    ModelEntry,
    available_models,
    create_model,
    get_model_entry,
    load_model_objects,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelEntry",
    "available_models",
    "create_model",
    "get_model_entry",
    "load_model_objects",
]
