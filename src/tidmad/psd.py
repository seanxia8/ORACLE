"""Per-band noise PSD ``J(f)`` for the chi2 objective (PLAN_04 §3.2).

``J(f)`` is the *same object the paper cites*: Paper 1's Phase-4 estimator
(``tidmad._vendor.psd.welch_psd_from_windows``), median-averaged boxcar
periodograms over 512 calibration windows of 4096 samples, estimated from
**noise-only science files** — never from evaluation data. The calibration
window length equals ``n_fft`` (4096), so the PSD grid coincides with the STFT
band grid and ``J`` is taken at each kept band's bin directly.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import TidmadSTFTConfig

__all__ = ["estimate_band_psd", "tile_for_stacked_bands", "whitening_identity_error"]


def estimate_band_psd(
    noise_windows: Iterable[np.ndarray],
    cfg: TidmadSTFTConfig,
    *,
    sampling_frequency: float,
    window: str = "boxcar",
    average: str = "median",
) -> tuple[np.ndarray, object]:
    """Estimate the per-band noise PSD for the chi2 objective.

    ``j_band`` is the median-averaged periodogram power of the noise-only
    windows **in the data pipeline's own STFT units** (so the whitened residual
    ``|resid|^2 / J`` is measured in the loss's coordinate system). This is the
    same estimator methodology as Paper 1's Phase-4 median-periodogram PSD; the
    boxcar PSD density is also computed and returned as ``estimate`` for the
    archived record and the ``χ²/dof`` comparison.
    """
    from ._vendor.psd import welch_psd_from_windows
    from .data import stft

    windows = list(noise_windows)
    if not windows:
        raise ValueError("no noise-only windows supplied for the PSD estimate")
    powers = []
    for w in windows:
        X = stft(w, cfg)[: cfg.n_bands_used, :]
        powers.append((X.real**2 + X.imag**2).mean(axis=1))
    j_band = np.median(np.vstack(powers), axis=0)
    if j_band.size < cfg.n_bands_used:
        raise ValueError(
            f"PSD grid has {j_band.size} kept bins but n_bands_used={cfg.n_bands_used}; "
            "calibration window and n_fft must match"
        )
    if not np.all(j_band > 0):
        raise ValueError("J(f) must be strictly positive")

    estimate = welch_psd_from_windows(
        windows,
        float(sampling_frequency),
        window=window,
        detrend=True,
        average=average,
        source="tidmad_rung4_science",
    )
    return j_band, estimate


def tile_for_stacked_bands(j_band: np.ndarray, cfg: TidmadSTFTConfig) -> np.ndarray:
    """Repeat the per-bin PSD over the real/imaginary band stacking."""
    return np.tile(j_band, 2)


def per_band_whitened_power(noise_windows: Iterable[np.ndarray], j_band: np.ndarray, cfg: TidmadSTFTConfig) -> np.ndarray:
    """Median per-band periodogram power divided by ``J``, in data-STFT units.

    Uses the same STFT as the data pipeline so the whitened power is measured in
    the residual's coordinate system. The result is flat across bands when
    ``J`` matches the true per-band noise power.
    """
    from .data import stft

    j = np.maximum(np.asarray(j_band, dtype=np.float64), np.finfo(float).tiny)
    powers = []
    for w in noise_windows:
        X = stft(w, cfg)[: cfg.n_bands_used, :]
        powers.append((X.real**2 + X.imag**2).mean(axis=1))
    if not powers:
        raise ValueError("no noise windows supplied for the whitening check")
    median_power = np.median(np.vstack(powers), axis=0)
    return median_power / j


def whitening_identity_error(
    noise_windows: Iterable[np.ndarray], j_band: np.ndarray, cfg: TidmadSTFTConfig
) -> float:
    """Max per-band deviation of the whitened power from its mean (scale-free).

    A correct ``J`` flattens the per-band noise power: after dividing each
    band's periodogram power by ``J[band]``, the result is constant across
    bands. The global periodogram-normalization constant is divided out, so this
    is valid even though ``J`` is a PSD density and the STFT is unnormalised.
    """
    whitened = per_band_whitened_power(noise_windows, j_band, cfg)
    # The DC band is degenerate for zero-mean calibration noise (both numerator
    # and J are ~0); exclude it from the flatness check over bands 1..end.
    whitened = whitened[1:]
    mean = float(np.mean(whitened))
    if mean <= 0:
        raise ValueError("whitened power has non-positive mean")
    return float(np.max(np.abs(whitened / mean - 1.0)))
