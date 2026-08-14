"""Empirical calibration, held-out validation, and reproducible presets."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import csd, find_peaks, welch
from scipy.stats import kurtosis, skew

from .NoiseGenerator import NoiseGenerator
from .multichannel_noise import MultiChannelNoiseGenerator


CALIBRATION_SCHEMA_VERSION = 1


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class ReferenceDataset:
    """Standard empirical input with shape (records, channels, samples)."""

    data: np.ndarray
    sampling_frequency: float
    dataset_id: str
    units: str = "signal"
    channel_names: list[str] = field(default_factory=list)
    acquisition: dict[str, Any] = field(default_factory=dict)
    preprocessing: list[str] = field(default_factory=list)
    two_dimensional_layout: str | None = None

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data[None, None, :]
        elif self.data.ndim == 2:
            if self.two_dimensional_layout == "records_samples":
                self.data = self.data[:, None, :]
            elif self.two_dimensional_layout == "channels_samples":
                self.data = self.data[None, :, :]
            else:
                raise ValueError(
                    "two-dimensional data are ambiguous; set "
                    "two_dimensional_layout to 'records_samples' or 'channels_samples'."
                )
        elif self.two_dimensional_layout not in {None, "records_samples", "channels_samples"}:
            raise ValueError("Unsupported two_dimensional_layout.")
        if self.data.ndim != 3 or np.any(~np.isfinite(self.data)):
            raise ValueError("data must be finite with shape (R, C, N).")
        if self.sampling_frequency <= 0 or not self.dataset_id:
            raise ValueError("sampling_frequency and dataset_id are required.")
        if not self.channel_names:
            self.channel_names = [f"channel_{i}" for i in range(self.data.shape[1])]
        if len(self.channel_names) != self.data.shape[1]:
            raise ValueError("channel_names length must match channel count.")

    def save(self, path: str | Path) -> None:
        metadata = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "sampling_frequency": self.sampling_frequency,
            "dataset_id": self.dataset_id,
            "two_dimensional_layout": self.two_dimensional_layout,
            "units": self.units,
            "channel_names": self.channel_names,
            "acquisition": self.acquisition,
            "preprocessing": self.preprocessing,
        }
        np.savez_compressed(path, data=self.data, metadata=json.dumps(metadata))

    @classmethod
    def load(cls, path: str | Path):
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata"]))
        metadata.pop("schema_version", None)
        return cls(data=payload["data"], **metadata)


@dataclass
class CalibrationPreset:
    schema_version: int
    dataset_id: str
    sampling_frequency: float
    units: str
    channel_names: list[str]
    estimator: dict[str, Any]
    provenance: dict[str, Any]
    frequencies: np.ndarray
    psd: np.ndarray
    csd: np.ndarray
    statistics: dict[str, Any]
    uncertainty: dict[str, Any]
    claims: list[str]
    not_modeled: list[str]
    heldout_validation: dict[str, Any]

    def save(self, path: str | Path) -> None:
        payload = dict(self.__dict__)
        payload["csd"] = {"real": self.csd.real, "imag": self.csd.imag}
        Path(path).write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("frequencies", "psd"):
            data[key] = np.asarray(data[key], dtype=float)
        data["csd"] = np.asarray(data["csd"]["real"]) + 1j * np.asarray(data["csd"]["imag"]) \
            if isinstance(data["csd"], dict) else np.asarray(data["csd"], dtype=complex)
        return cls(**data)

    def sample_parameters(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Sample PSD density from bootstrap uncertainty in log space."""
        rng = np.random.default_rng(seed)
        lower = np.asarray(self.uncertainty["psd_lower"])
        upper = np.asarray(self.uncertainty["psd_upper"])
        lo = np.log(np.maximum(lower, 1e-30))
        hi = np.log(np.maximum(upper, 1e-30))
        return {"psd": np.exp(rng.uniform(lo, hi)), "frequencies": self.frequencies.copy()}

    def generate(self, N: int, seed: int | None = None) -> np.ndarray:
        """Regenerate a trace or channel array directly from calibrated PSD/CSD."""
        target_f = np.fft.rfftfreq(N, 1 / self.sampling_frequency)
        if self.csd.shape[1] > 1:
            generator = MultiChannelNoiseGenerator(
                {"noise_type": "white", "noise_power": 1, "sampling_frequency": self.sampling_frequency},
                seed=seed,
            )
            return generator.generate_from_csd(
                self.csd, N, target_frequencies=self.frequencies
            )
        density = np.interp(target_f, self.frequencies, self.psd[0])
        generator = NoiseGenerator(
            {"noise_type": "white", "noise_power": 1, "sampling_frequency": self.sampling_frequency},
            seed=seed,
        )
        power = generator.psd_density_to_rfft_power(density, self.sampling_frequency, N)
        return generator.sample_stationary_gaussian_from_rfft_power(power, N=N)


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def calibrate_dataset(
    dataset: ReferenceDataset,
    *,
    train_fraction: float = 0.75,
    nperseg: int = 256,
    overlap_fraction: float = 0.5,
    bootstrap_samples: int = 300,
    seed: int = 0,
) -> CalibrationPreset:
    """Fit PSD/CSD and empirical diagnostics using a train/held-out split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie in (0, 1).")
    if dataset.data.shape[0] < 2:
        raise ValueError("Calibration requires at least two records for a held-out split.")
    if bootstrap_samples < 20:
        raise ValueError("bootstrap_samples must be at least 20.")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must lie in [0, 1).")
    rng = np.random.default_rng(seed)
    order = rng.permutation(dataset.data.shape[0])
    split = max(1, min(len(order) - 1, int(round(train_fraction * len(order)))))
    train, heldout = dataset.data[order[:split]], dataset.data[order[split:]]
    nperseg = min(nperseg, train.shape[-1])
    noverlap = int(overlap_fraction * nperseg)
    psd_records = []
    for record in train:
        channel_psd = []
        for channel in record:
            f, p = welch(
                channel, fs=dataset.sampling_frequency, nperseg=nperseg,
                noverlap=noverlap, detrend="constant", scaling="density",
            )
            channel_psd.append(p)
        psd_records.append(channel_psd)
    psd_records = np.asarray(psd_records)
    mean_psd = psd_records.mean(axis=0)
    C = train.shape[1]
    mean_csd = np.zeros((len(f), C, C), dtype=complex)
    for i in range(C):
        for j in range(C):
            estimates = [
                csd(record[i], record[j], fs=dataset.sampling_frequency,
                    nperseg=nperseg, noverlap=noverlap, detrend="constant",
                    scaling="density")[1]
                for record in train
            ]
            mean_csd[:, i, j] = np.mean(estimates, axis=0)

    alpha = 0.05
    bootstrap_psd = np.asarray([
        psd_records[rng.integers(0, len(psd_records), len(psd_records))].mean(axis=0)
        for _ in range(bootstrap_samples)
    ])
    psd_lower = np.quantile(bootstrap_psd, alpha / 2, axis=0)
    psd_upper = np.quantile(bootstrap_psd, 1 - alpha / 2, axis=0)
    flat = train.reshape(-1, train.shape[-1])
    segment_power = np.var(flat, axis=1)
    standardized = (flat - flat.mean(axis=1, keepdims=True)) / np.maximum(
        flat.std(axis=1, keepdims=True), 1e-12
    )
    mean_spectrum = mean_psd.mean(axis=0)
    peaks, properties = find_peaks(
        mean_spectrum, prominence=max(np.median(mean_spectrum) * 5, 1e-30)
    )
    threshold = 5.0
    exceedance = np.abs(standardized) > threshold
    # Count the start of each exceedance run. Boolean ``diff`` marks both
    # rising and falling transitions and therefore approximately doubles the
    # inferred event rate.
    transitions = np.diff(
        exceedance.astype(np.int8),
        axis=1,
        prepend=np.zeros((exceedance.shape[0], 1), dtype=np.int8),
    )
    event_counts = np.sum(transitions == 1, axis=1)
    cumulative = np.cumsum(mean_spectrum)
    drift_index = int(np.searchsorted(cumulative, 0.1 * cumulative[-1]))
    stats = {
        "segment_power": segment_power,
        "segment_power_interval": np.quantile(segment_power, [0.025, 0.975]),
        "line_frequencies_hz": f[peaks],
        "line_prominences": properties.get("prominences", np.array([])),
        "artifact_rate_per_second": float(
            np.mean(event_counts) / (train.shape[-1] / dataset.sampling_frequency)
        ),
        "tail_exceedance_5sigma": float(np.mean(exceedance)),
        "skewness": float(skew(standardized.ravel())),
        "excess_kurtosis": float(kurtosis(standardized.ravel())),
        "drift_band_edge_hz": float(f[min(drift_index, len(f) - 1)]),
    }
    heldout_power = np.var(heldout, axis=-1).ravel()
    interval = stats["segment_power_interval"]
    heldout_fraction = float(np.mean((heldout_power >= interval[0]) & (heldout_power <= interval[1])))
    required_coverage = 0.5  # predeclared minimum for small held-out record sets
    heldout_psd_records = []
    for record in heldout:
        heldout_psd_records.append([
            welch(
                channel,
                fs=dataset.sampling_frequency,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
            )[1]
            for channel in record
        ])
    heldout_mean_psd = np.mean(np.asarray(heldout_psd_records), axis=0)
    positive_psd = mean_psd > max(float(np.max(mean_psd)) * 1e-12, 1e-30)
    median_log_psd_error = float(
        np.median(
            np.abs(
                np.log(
                    np.maximum(heldout_mean_psd[positive_psd], 1e-30)
                    / mean_psd[positive_psd]
                )
            )
        )
    )
    psd_interval_coverage = float(
        np.mean((heldout_mean_psd >= psd_lower) & (heldout_mean_psd <= psd_upper))
    )
    heldout_flat = heldout.reshape(-1, heldout.shape[-1])
    heldout_standardized = (
        heldout_flat - heldout_flat.mean(axis=1, keepdims=True)
    ) / np.maximum(heldout_flat.std(axis=1, keepdims=True), 1e-12)
    heldout_tail = float(np.mean(np.abs(heldout_standardized) > threshold))
    tail_standard_error = np.sqrt(
        max(stats["tail_exceedance_5sigma"], 1.0 / standardized.size)
        / heldout_standardized.size
    )
    tail_tolerance = float(max(0.005, 3.0 * tail_standard_error))
    heldout_csd = np.zeros_like(mean_csd)
    for i in range(C):
        for j in range(C):
            heldout_csd[:, i, j] = np.mean([
                csd(
                    record[i],
                    record[j],
                    fs=dataset.sampling_frequency,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend="constant",
                    scaling="density",
                )[1]
                for record in heldout
            ], axis=0)
    csd_relative_frobenius_error = float(
        np.linalg.norm(heldout_csd - mean_csd)
        / max(np.linalg.norm(mean_csd), 1e-30)
    )
    gates = {
        "power_coverage": heldout_fraction >= required_coverage,
        "psd_median_log_ratio": median_log_psd_error <= np.log(2.0),
        "psd_interval_coverage": psd_interval_coverage >= 0.5,
        "tail_exceedance": (
            abs(heldout_tail - stats["tail_exceedance_5sigma"]) <= tail_tolerance
        ),
        "csd_relative_frobenius": csd_relative_frobenius_error <= 0.75,
    }
    validation = {
        "power_coverage_fraction": heldout_fraction,
        "required_coverage_fraction": required_coverage,
        "median_absolute_log_psd_ratio": median_log_psd_error,
        "psd_interval_coverage_fraction": psd_interval_coverage,
        "heldout_tail_exceedance_5sigma": heldout_tail,
        "tail_exceedance_tolerance": tail_tolerance,
        "csd_relative_frobenius_error": csd_relative_frobenius_error,
        "gates": gates,
        "passed": all(gates.values()),
        "n_train_records": int(len(train)),
        "n_heldout_records": int(len(heldout)),
    }
    return CalibrationPreset(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        dataset_id=dataset.dataset_id,
        sampling_frequency=dataset.sampling_frequency,
        units=dataset.units,
        channel_names=dataset.channel_names,
        estimator={
            "method": "Welch/CSD",
            "nperseg": nperseg,
            "noverlap": noverlap,
            "detrend": "constant",
            "scaling": "density",
            "bootstrap_samples": bootstrap_samples,
            "split_seed": seed,
        },
        provenance={
            "dataset_id": dataset.dataset_id,
            "acquisition": dataset.acquisition,
            "preprocessing": dataset.preprocessing,
            "code_revision": _git_revision(),
            "calibration_date_utc": datetime.now(timezone.utc).isoformat(),
        },
        frequencies=f,
        psd=mean_psd,
        csd=mean_csd,
        statistics=stats,
        uncertainty={"psd_lower": psd_lower, "psd_upper": psd_upper},
        claims=[
            "stationary PSD",
            "cross-spectral density",
            "segment power",
            "tail diagnostics",
            "held-out PSD/CSD/power/tail validation",
        ],
        not_modeled=["causal artifact morphology", "state-dependent nonstationarity"],
        heldout_validation=validation,
    )
