# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
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
from .reference_budget import (
    AL2O3_AL_ATHERMAL,
    AthermalNoiseBudget,
    BudgetGrid,
    write_reference_asd,
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
    "AL2O3_AL_ATHERMAL",
    "al2o3_athermal_noise_generator",
    "AL2O3_DEFAULT_SAMPLES",
    "AL2O3_DEFAULT_SAMPLING_FREQUENCY",
    "alias_fold_psd_density",
    "ArtifactConfig",
    "ArtifactInjector",
    "AthermalNoiseBudget",
    "BandLimited",
    "benchmark_generation",
    "bootstrap_interval",
    "BudgetGrid",
    "build_optimal_filter",
    "calibrate_dataset",
    "CalibrationPreset",
    "CompositeSpectrum",
    "CONFIG_SCHEMA_VERSION",
    "fit_reference_pulse",
    "inband_resample_psd_density",
    "Line",
    "load_al2o3_athermal_composite",
    "load_psd_density",
    "Lorentzian",
    "make_target_psd_density",
    "migrate_config",
    "MultiChannelConfig",
    "MultiChannelNoiseGenerator",
    "NoiseConfig",
    "NoiseGenerator",
    "NonGaussianNoiseGenerator",
    "OptimalFilter",
    "PowerLaw",
    "pulse_template_2",
    "PulseFit",
    "recommend_record_length",
    "ReferenceDataset",
    "Resonance",
    "RollOff",
    "save_psd_density",
    "SpectralComponent",
    "StreamingNoiseGenerator",
    "synthetic_resample_psd_density",
    "TemporalNoiseConfig",
    "TemporalNoiseWrapper",
    "to_jsonable",
    "validate_artifacts",
    "validate_csd_ensemble",
    "validate_local_nonstationarity",
    "validate_reference_noise",
    "validate_stationary_gaussian",
    "ValidationConfig",
    "ValidationResult",
    "White",
    "write_reference_asd",
]

__version__ = "0.3.0"
