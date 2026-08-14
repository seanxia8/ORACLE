"""PSD and leakage tests (PLAN_04 §3.6 items 4–5)."""

from __future__ import annotations

import numpy as np

from tidmad.config import FROZEN, TidmadSTFTConfig
from tidmad.psd import estimate_band_psd, tile_for_stacked_bands, whitening_identity_error
from tidmad._vendor.splits import chronological_window_split, leakage_audit


def _coloured_noise_windows(cfg: TidmadSTFTConfig, n_windows: int, seed: int) -> list[np.ndarray]:
    """Windows with a 1/f-ish power profile (normalized so std ~ 1)."""
    rng = np.random.default_rng(seed)
    n = cfg.win_length
    freqs = np.fft.rfftfreq(n, d=1.0 / cfg.sampling_frequency_hz)
    f = np.maximum(freqs[1:-1], freqs[1])  # exclude DC and Nyquist
    colour = (f[0] / f)  # 1/f, normalized at the first interior bin
    spec = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) * np.sqrt(colour)
    x = np.fft.irfft(spec, n=n)
    return [x / x.std() for _ in range(n_windows)]


def test_psd_positive_and_whitening_identity():
    cfg = FROZEN.stft
    windows = _coloured_noise_windows(cfg, 64, seed=11)
    cal, val = windows[:32], windows[32:]
    j_band, estimate = estimate_band_psd(
        cal, cfg, sampling_frequency=cfg.sampling_frequency_hz, window="boxcar", average="median"
    )
    assert j_band.shape == (cfg.n_bands_used,)
    assert np.all(j_band > 0)
    err = whitening_identity_error(val, j_band, cfg)
    assert err < 0.25, f"whitening identity error {err:.3f}"


def test_psd_matches_noise_power_profile():
    """A coloured PSD must reflect the injected colour, not be flat."""
    cfg = FROZEN.stft
    windows = _coloured_noise_windows(cfg, 32, seed=13)
    j_band, _ = estimate_band_psd(
        windows, cfg, sampling_frequency=cfg.sampling_frequency_hz, window="boxcar", average="median"
    )
    ratio = j_band[1] / j_band[-1]  # skip DC bin (zero-mean signal)
    assert ratio > 100, f"coloured noise PSD must be non-flat (ratio {ratio:.1f})"


def test_tile_for_stacked_bands():
    cfg = FROZEN.stft
    j = np.arange(1, cfg.n_bands_used + 1, dtype=np.float64)
    stacked = tile_for_stacked_bands(j, cfg)
    assert stacked.shape == (cfg.n_bands_stacked,)
    assert np.allclose(stacked[: cfg.n_bands_used], j)
    assert np.allclose(stacked[cfg.n_bands_used :], j)


def test_chronological_split_no_leakage():
    n_windows, fit_frac, guard = 100, 0.7, 4
    split = chronological_window_split(n_windows, fit_fraction=fit_frac, guard_windows=guard)
    rng = np.random.default_rng(17)
    fit = rng.standard_normal((split.fit_indices.size, 32))
    ev = rng.standard_normal((split.evaluation_indices.size, 32))
    audit = leakage_audit(
        split,
        fit_windows=fit,
        evaluation_windows=ev,
        window_samples=4096,
        stride=2048,
        sampling_frequency=10_000_000.0,
    )
    assert audit["index_overlap_count"] == 0
    assert audit["hash_overlap_count"] == 0
    assert audit["guard_gap_windows_realized"] == guard
