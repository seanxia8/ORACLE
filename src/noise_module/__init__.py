"""Composable noise generation modules for wk7 experiments."""

from .NoiseGenerator import NoiseGenerator
from .config import (
    CONFIG_SCHEMA_VERSION,
    ArtifactConfig,
    MultiChannelConfig,
    NoiseConfig,
    TemporalNoiseConfig,
    migrate_config,
)
from .artifact_injector import ArtifactInjector
from .multichannel_noise import MultiChannelNoiseGenerator
from .non_gaussian import NonGaussianNoiseGenerator
from .calibration import CalibrationPreset, ReferenceDataset, calibrate_dataset
from .streaming import StreamingNoiseGenerator, benchmark_generation
from .validation import (
    ValidationConfig,
    ValidationResult,
    bootstrap_interval,
    validate_artifacts,
    validate_csd_ensemble,
    validate_local_nonstationarity,
    validate_stationary_gaussian,
)
from .psd_resampling import (
    alias_fold_psd_density,
    inband_resample_psd_density,
    load_psd_density,
    make_target_psd_density,
    save_psd_density,
    synthetic_resample_psd_density,
)
from .temporal_noise import TemporalNoiseWrapper
from .templates import pulse_template_2
from .al2o3_athermal import (
    DEFAULT_SAMPLES as AL2O3_DEFAULT_SAMPLES,
    DEFAULT_SAMPLING_FREQUENCY as AL2O3_DEFAULT_SAMPLING_FREQUENCY,
    OptimalFilter,
    PulseFit,
    build_optimal_filter,
    fit_reference_pulse,
    load_composite as load_al2o3_athermal_composite,
    noise_generator as al2o3_athermal_noise_generator,
    recommend_record_length,
    validate_reference_noise,
)
from .utils import to_jsonable
from .spectral_models import (
    BandLimited,
    CompositeSpectrum,
    Line,
    Lorentzian,
    PowerLaw,
    Resonance,
    RollOff,
    SpectralComponent,
    White,
)

__all__ = [
    "AL2O3_DEFAULT_SAMPLES",
    "AL2O3_DEFAULT_SAMPLING_FREQUENCY",
    "ArtifactInjector",
    "ArtifactConfig",
    "CalibrationPreset",
    "BandLimited",
    "CompositeSpectrum",
    "CONFIG_SCHEMA_VERSION",
    "Line",
    "Lorentzian",
    "MultiChannelNoiseGenerator",
    "MultiChannelConfig",
    "NoiseGenerator",
    "OptimalFilter",
    "NonGaussianNoiseGenerator",
    "NoiseConfig",
    "PowerLaw",
    "PulseFit",
    "ReferenceDataset",
    "Resonance",
    "RollOff",
    "SpectralComponent",
    "TemporalNoiseWrapper",
    "TemporalNoiseConfig",
    "StreamingNoiseGenerator",
    "ValidationConfig",
    "ValidationResult",
    "White",
    "alias_fold_psd_density",
    "inband_resample_psd_density",
    "load_psd_density",
    "make_target_psd_density",
    "save_psd_density",
    "synthetic_resample_psd_density",
    "migrate_config",
    "benchmark_generation",
    "build_optimal_filter",
    "bootstrap_interval",
    "calibrate_dataset",
    "fit_reference_pulse",
    "load_al2o3_athermal_composite",
    "al2o3_athermal_noise_generator",
    "pulse_template_2",
    "recommend_record_length",
    "validate_artifacts",
    "validate_csd_ensemble",
    "validate_local_nonstationarity",
    "validate_stationary_gaussian",
    "validate_reference_noise",
    "to_jsonable",
]

__version__ = "0.3.0"
