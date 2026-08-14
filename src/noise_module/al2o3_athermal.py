"""Al2O3/Al athermal reference model, pulse fit, and optimal filter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import welch

from .NoiseGenerator import NoiseGenerator
from .spectral_models import CompositeSpectrum, component_from_config
from .templates import pulse_template_2


DATA_DIR = Path(__file__).resolve().parent / "data" / "Al2O3_Al_athermal"
FIT_FILE = DATA_DIR / "al2o3_athermal_fit.json"
PSD_FILE = DATA_DIR / "al2o3_athermal_total_psd.npy"
DEFAULT_SAMPLING_FREQUENCY = 1_000_000.0
DEFAULT_SAMPLES = 131_072


@dataclass(frozen=True)
class PulseFit:
    """Parameters inferred from the magnitude-only ``signal.dat``."""

    t0: float
    An: float
    At: float
    tau_n: float
    tau_in: float
    tau_t: float
    rms_log_error: float
    identifiable_parameters: tuple[str, ...]
    unidentifiable_parameters: tuple[str, ...]

    def template(self, t: np.ndarray) -> np.ndarray:
        return pulse_template_2(
            t, self.t0, self.An, self.At, self.tau_n, self.tau_in, self.tau_t
        )


def load_composite(fit_file: str | Path = FIT_FILE) -> CompositeSpectrum:
    """Load the Phase 1 absolute-density composite model."""
    config = json.loads(Path(fit_file).read_text())
    return CompositeSpectrum(
        component_from_config(component) for component in config["components"]
    )


def noise_generator(
    *,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
    seed: int | None = None,
    fit_file: str | Path = FIT_FILE,
    psd_file: str | Path | None = None,
) -> NoiseGenerator:
    """Build a Phase 3 generator from the analytic model or Phase 2 artifact."""
    noise_type: CompositeSpectrum | str
    if psd_file is None:
        noise_type = load_composite(fit_file)
    else:
        noise_type = str(Path(psd_file))
    return NoiseGenerator(
        {
            "noise_type": noise_type,
            "noise_power": 1.0,
            "sampling_frequency": sampling_frequency,
            "composite_psd_scaling": "absolute",
            "custom_psd_scaling": "absolute",
            "custom_out_of_band": "error",
            "power_definition": "variance",
        },
        seed=seed,
    )


def _pulse_frequency_magnitude(
    frequency: np.ndarray,
    An: float,
    At: float,
    tau_n: float,
    tau_in: float,
    tau_t: float,
) -> np.ndarray:
    omega = 2.0 * np.pi * frequency

    def transform(tau: float) -> np.ndarray:
        return tau / (1.0 + 1j * omega * tau)

    response = (
        An * (transform(tau_n) - transform(tau_in))
        + At * (transform(tau_t) - transform(tau_n))
    )
    return np.abs(response)


def fit_reference_pulse(
    signal_file: str | Path = DATA_DIR / "signal.dat",
    *,
    t0: float = 0.0,
) -> PulseFit:
    """Fit the reference magnitude response to the physical pulse model.

    ``signal.dat`` has no phase, hence it cannot constrain ``t0``. In this
    dataset ``An == At`` to numerical precision, which also cancels
    ``tau_n``. A canonical geometric-mean ``tau_n`` is returned so callers
    can still use the standard six-parameter template.
    """
    frequency, magnitude = np.loadtxt(signal_file).T

    # The exact reference is a rise/fall pair. Fitting this identifiable
    # reduction avoids reporting an arbitrary intermediate time constant.
    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude, log_tau_in, log_tau_delta = parameters
        tau_in = np.exp(log_tau_in)
        tau_t = tau_in + np.exp(log_tau_delta)
        predicted = np.abs(amplitude) * np.abs(
            tau_t / (1.0 + 2j * np.pi * frequency * tau_t)
            - tau_in / (1.0 + 2j * np.pi * frequency * tau_in)
        )
        return np.log(predicted) - np.log(magnitude)

    result = least_squares(
        residual,
        np.array([2.0, np.log(5e-6), np.log(1e-3 - 5e-6)]),
        max_nfev=10_000,
    )
    amplitude, log_tau_in, log_tau_delta = result.x
    tau_in = float(np.exp(log_tau_in))
    tau_t = float(tau_in + np.exp(log_tau_delta))
    tau_n = float(np.sqrt(tau_in * tau_t))
    error = residual(result.x)
    return PulseFit(
        t0=float(t0),
        An=float(amplitude),
        At=float(amplitude),
        tau_n=tau_n,
        tau_in=tau_in,
        tau_t=tau_t,
        rms_log_error=float(np.sqrt(np.mean(error**2))),
        identifiable_parameters=("An=At", "tau_in", "tau_t"),
        unidentifiable_parameters=("t0", "tau_n"),
    )


def recommend_record_length(
    tau_slowest: float,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
    *,
    decay_constants: float = 10.0,
    minimum_frequency_hz: float = 10.0,
    power_of_two: bool = True,
) -> dict[str, float | int]:
    """Choose a record from pulse containment and low-frequency resolution."""
    if min(tau_slowest, sampling_frequency, decay_constants, minimum_frequency_hz) <= 0:
        raise ValueError("Record-length inputs must be positive.")
    duration = max(
        decay_constants * tau_slowest,
        1.0 / minimum_frequency_hz,
    )
    samples = int(np.ceil(duration * sampling_frequency))
    if power_of_two:
        samples = 1 << int(np.ceil(np.log2(samples)))
    return {
        "n_samples": samples,
        "duration_seconds": samples / sampling_frequency,
        "frequency_resolution_hz": sampling_frequency / samples,
        "pulse_decay_constants": samples / sampling_frequency / tau_slowest,
    }


@dataclass(frozen=True)
class OptimalFilter:
    """Noise-weighted amplitude estimator on one fixed rFFT grid."""

    template: np.ndarray
    psd: np.ndarray
    sampling_frequency: float

    def __post_init__(self) -> None:
        template = np.asarray(self.template, dtype=float)
        psd = np.asarray(self.psd, dtype=float)
        if template.ndim != 1 or psd.shape != (template.size // 2 + 1,):
            raise ValueError("PSD must match the template's rFFT grid.")
        if np.any(np.isnan(psd)) or np.any(psd <= 0.0):
            raise ValueError("Optimal-filter PSD must be positive and not NaN.")
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "psd", psd)

    @property
    def template_fft(self) -> np.ndarray:
        return np.fft.rfft(self.template)

    @property
    def kernel(self) -> np.ndarray:
        """Return QETpy-convention ``conj(template_fft) / PSD``."""
        return np.conj(self.template_fft) / self.psd

    @property
    def normalization(self) -> float:
        return float(np.sum(np.abs(self.template_fft) ** 2 / self.psd))

    def estimate_amplitude(self, trace: np.ndarray) -> float:
        trace = np.asarray(trace, dtype=float)
        if trace.shape != self.template.shape:
            raise ValueError("Trace and template must have the same shape.")
        return float(np.real(np.sum(self.kernel * np.fft.rfft(trace))) / self.normalization)


def build_optimal_filter(
    *,
    n_samples: int = DEFAULT_SAMPLES,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
    pulse_fit: PulseFit | None = None,
    pretrigger_fraction: float = 0.1,
    fit_file: str | Path = FIT_FILE,
    psd_file: str | Path | None = None,
) -> OptimalFilter:
    """Build matching PSD and pulse grids for Phase 6.

    The checked-in Phase 2 artifact is used automatically on the default
    1 MHz/131072 grid. Other grids use the analytic model unless ``psd_file``
    is supplied explicitly.
    """
    if not 0.0 <= pretrigger_fraction < 1.0:
        raise ValueError("pretrigger_fraction must lie in [0, 1).")
    fit = pulse_fit or fit_reference_pulse(t0=pretrigger_fraction * n_samples / sampling_frequency)
    time = np.arange(n_samples, dtype=float) / sampling_frequency
    template = fit.template(time)
    selected_psd = psd_file
    if (
        selected_psd is None
        and n_samples == DEFAULT_SAMPLES
        and sampling_frequency == DEFAULT_SAMPLING_FREQUENCY
    ):
        selected_psd = PSD_FILE
    _, psd = noise_generator(
        sampling_frequency=sampling_frequency,
        fit_file=fit_file,
        psd_file=selected_psd,
    ).build_psd_density(n_samples)
    # DC is deliberately absent for a variance process; exclude it from the
    # filter by assigning infinite noise rather than dividing by zero.
    psd = psd.copy()
    psd[psd <= 0.0] = np.inf
    return OptimalFilter(template, psd, sampling_frequency)


def validate_reference_noise(
    *,
    n_realizations: int = 32,
    n_samples: int = DEFAULT_SAMPLES,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
    seed: int = 1234,
) -> dict[str, Any]:
    """Welch-check a Phase 3 ensemble against its analytic target PSD."""
    generator = noise_generator(sampling_frequency=sampling_frequency, seed=seed)
    records = generator.generate_ensemble(n_realizations, n_samples)
    nperseg = min(16_384, n_samples)
    estimates = [
        welch(x, fs=sampling_frequency, nperseg=nperseg, detrend=False)
        for x in records
    ]
    frequency = estimates[0][0]
    estimated_psd = np.mean([item[1] for item in estimates], axis=0)
    model_psd, _ = load_composite().evaluate(
        frequency, sampling_frequency / nperseg, zero_dc=True
    )
    valid = (frequency > 0.0) & (model_psd > 0.0)
    ratio = np.sqrt(estimated_psd[valid] / model_psd[valid])
    return {
        "n_realizations": n_realizations,
        "median_asd_ratio": float(np.median(ratio)),
        "rms_fractional_asd_error": float(np.sqrt(np.mean((ratio - 1.0) ** 2))),
        "pulse_fit": asdict(fit_reference_pulse()),
    }
