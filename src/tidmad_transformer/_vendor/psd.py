"""PSD estimation and stationarity selection for TIDMAD stretches.

Two things distinguish this from the generic estimator in
``noise_geometry.noise.psd``:

1. **Scale.** TIDMAD traces are ~2e9 samples per file. The PSD must be built by
   streaming Welch averages over windows rather than materializing the trace.
2. **Stationarity is not assumed, it is measured.** Paper 1's GWOSC stage failed
   precisely because a global stationary PSD does not describe real
   interferometer noise. Rather than repeat that assumption silently, this
   module measures block-to-block PSD drift and exposes an explicit,
   predeclarable selection rule. Whichever way the measurement comes out is
   reportable: good stationarity supports the Phase 2 comparison, poor
   stationarity *is* the Phase 4 boundary result.

Unit convention follows the repository throughout: one-sided PSD density on the
rFFT grid of the analysis window, and OF weights built by
``noise_geometry.noise.inverse_psd_weights`` so that DC/Nyquist handling lives
in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

__all__ = [
    "PSDEstimate",
    "StationarityReport",
    "welch_psd_from_windows",
    "block_psd_series",
    "assess_stationarity",
    "select_stationary_blocks",
]


@dataclass(frozen=True)
class PSDEstimate:
    """One-sided PSD density with the provenance needed to audit it."""

    frequencies: np.ndarray
    psd: np.ndarray
    n_windows: int
    window_samples: int
    sampling_frequency: float
    window: str
    source: str = "unspecified"
    metadata: dict = field(default_factory=dict)

    def weights(self) -> np.ndarray:
        """OF weights for this PSD, via the single canonical convention."""
        from .noise import inverse_psd_weights

        return inverse_psd_weights(self.psd, self.window_samples)


@dataclass(frozen=True)
class StationarityReport:
    """Block-to-block PSD variability, in dB, per frequency band."""

    block_frequencies: np.ndarray
    block_psds: np.ndarray  # (n_blocks, n_rfft)
    median_psd: np.ndarray
    max_abs_deviation_db: np.ndarray  # per block: worst band deviation
    band_edges_hz: tuple[float, ...]
    passing_blocks: np.ndarray  # boolean mask

    @property
    def fraction_passing(self) -> float:
        return float(np.mean(self.passing_blocks)) if self.passing_blocks.size else 0.0


def _window_function(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(n)
    if name == "boxcar":
        return np.ones(n, dtype=np.float64)
    raise ValueError(f"unsupported window: {name!r} (use 'hann' or 'boxcar')")


def welch_psd_from_windows(
    windows: Iterable[np.ndarray],
    sampling_frequency: float,
    *,
    window: str = "hann",
    detrend: bool = True,
    average: str = "median",
    source: str = "unspecified",
) -> PSDEstimate:
    """Average periodograms over an iterable of equal-length windows.

    ``average='median'`` is the default deliberately. TIDMAD noise contains
    glitches and non-stationary excursions; a mean-Welch estimate absorbs them
    into the PSD and then *under*-weights the affected bands during whitening,
    which silently degrades the very estimator under study. The median is bias-
    corrected to the mean of an exponential periodogram by the usual ``ln 2``
    factor so it remains an estimate of the same quantity.
    """
    if average not in {"mean", "median"}:
        raise ValueError("average must be 'mean' or 'median'")

    accum: list[np.ndarray] = []
    n_samples = None
    win = None
    for w in windows:
        x = np.asarray(w, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("each window must be one-dimensional")
        if n_samples is None:
            n_samples = x.size
            win = _window_function(window, n_samples)
        elif x.size != n_samples:
            raise ValueError("all windows must have the same length")
        if detrend:
            x = x - x.mean()
        spec = np.fft.rfft(x * win)
        accum.append(np.abs(spec) ** 2)

    if n_samples is None:
        raise ValueError("no windows supplied")

    power = np.vstack(accum)
    if average == "mean":
        avg = power.mean(axis=0)
    else:
        # Bias-correct the median of an exponential distribution to its mean.
        avg = np.median(power, axis=0) / np.log(2.0)

    fs = float(sampling_frequency)
    window_power = float(np.mean(np.asarray(win) ** 2))
    psd = avg / (fs * n_samples * window_power)
    if n_samples > 2:
        stop = n_samples // 2 + 1 - (n_samples + 1) % 2
        psd[1:stop] *= 2.0

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    return PSDEstimate(
        frequencies=freqs,
        psd=psd,
        n_windows=power.shape[0],
        window_samples=int(n_samples),
        sampling_frequency=fs,
        window=window,
        source=source,
        metadata={"average": average, "detrend": bool(detrend)},
    )


def block_psd_series(
    blocks: Iterable[Iterable[np.ndarray]],
    sampling_frequency: float,
    *,
    window: str = "hann",
    average: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies, psds)`` with one PSD row per block of windows."""
    rows = []
    freqs = None
    for block in blocks:
        est = welch_psd_from_windows(
            block, sampling_frequency, window=window, average=average, source="block"
        )
        if freqs is None:
            freqs = est.frequencies
        rows.append(est.psd)
    if freqs is None:
        raise ValueError("no blocks supplied")
    return freqs, np.vstack(rows)


def assess_stationarity(
    frequencies: np.ndarray,
    block_psds: np.ndarray,
    *,
    band_edges_hz: tuple[float, ...] = (1e2, 1e3, 1e4, 1e5, 1e6),
    tolerance_db: float = 3.0,
) -> StationarityReport:
    """Measure block-to-block PSD drift and flag blocks that exceed a tolerance.

    The comparison is made on band-averaged power, in dB, against the median
    block. Bands rather than bins because single-bin fluctuation is dominated by
    estimator variance, while a genuine non-stationarity moves a whole band.

    ``tolerance_db`` must be predeclared. It is a *selection rule*, and a
    selection rule chosen after seeing the downstream result is not a selection
    rule -- it is tuning.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    psds = np.atleast_2d(np.asarray(block_psds, dtype=np.float64))
    if psds.shape[1] != freqs.size:
        raise ValueError("block_psds columns must match frequencies")

    median_psd = np.median(psds, axis=0)
    edges = tuple(float(e) for e in band_edges_hz)

    band_masks = []
    lo = 0.0
    for hi in edges:
        mask = (freqs > lo) & (freqs <= hi)
        if mask.any():
            band_masks.append(mask)
        lo = hi
    tail = freqs > edges[-1]
    if tail.any():
        band_masks.append(tail)
    if not band_masks:
        raise ValueError("band_edges_hz produced no populated bands")

    deviations = np.zeros(psds.shape[0], dtype=np.float64)
    floor = np.finfo(float).tiny
    for i in range(psds.shape[0]):
        worst = 0.0
        for mask in band_masks:
            num = float(np.mean(psds[i, mask]))
            den = float(np.mean(median_psd[mask]))
            ratio_db = 10.0 * np.log10(max(num, floor) / max(den, floor))
            worst = max(worst, abs(ratio_db))
        deviations[i] = worst

    return StationarityReport(
        block_frequencies=freqs,
        block_psds=psds,
        median_psd=median_psd,
        max_abs_deviation_db=deviations,
        band_edges_hz=edges,
        passing_blocks=deviations <= float(tolerance_db),
    )


def select_stationary_blocks(report: StationarityReport) -> np.ndarray:
    """Indices of blocks passing the predeclared stationarity tolerance."""
    return np.flatnonzero(report.passing_blocks)
