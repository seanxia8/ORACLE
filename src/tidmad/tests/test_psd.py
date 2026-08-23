"""PSD and leakage tests (PLAN_04 §3.6 items 4–5)."""

from __future__ import annotations

import numpy as np

from tidmad.config import FROZEN, TidmadSTFTConfig
from tidmad.psd import estimate_band_psd, tile_for_stacked_bands, whitening_identity_error
from tidmad._vendor.splits import chronological_window_split, leakage_audit


def _coloured_noise_windows(cfg: TidmadSTFTConfig, n_windows: int, seed: int) -> list[np.ndarray]:
    """Independent windows with a shared 1/f-ish power profile.

    Each window is an INDEPENDENT realisation (audit N4/C22: the previous
    version drew ``x`` once and returned the identical array ``n_windows``
    times, which made the calibration and validation splits the same waveform
    and the whitening check self-fulfilling).
    """
    rng = np.random.default_rng(seed)
    n = cfg.win_length
    freqs = np.fft.rfftfreq(n, d=1.0 / cfg.sampling_frequency_hz)
    f = np.maximum(freqs[1:-1], freqs[1])  # exclude DC and Nyquist
    colour = (f[0] / f)  # 1/f, normalized at the first interior bin
    windows = []
    for _ in range(n_windows):
        spec = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) * np.sqrt(colour)
        x = np.fft.irfft(spec, n=n)
        windows.append(x / x.std())
    return windows


def test_psd_positive_and_whitening_identity():
    """J estimated on one half of independent draws must whiten the other half.

    The threshold reflects real estimator statistics: each band's median-of-256
    periodograms has ~1.44/sqrt(256) = 9% relative sd; the check divides two
    independent medians and takes a MAX over 327 bands, so the null max
    deviation sits near 3.3 sigma * sqrt(2) * 9% ~ 0.42. 0.6 passes a correct
    J with margin and still fails a mis-estimated J (a wrong colour slope
    produces O(1)-to-O(100) deviations, see the non-flatness test below).
    """
    cfg = FROZEN.stft
    windows = _coloured_noise_windows(cfg, 512, seed=11)
    cal, val = windows[:256], windows[256:]
    j_band, estimate = estimate_band_psd(
        cal, cfg, sampling_frequency=cfg.sampling_frequency_hz, window="boxcar", average="median"
    )
    assert j_band.shape == (cfg.n_bands_used,)
    assert np.all(j_band > 0)
    err = whitening_identity_error(val, j_band, cfg)
    assert err < 0.6, f"whitening identity error {err:.3f}"
    # A deliberately wrong (flat) J must fail by a wide margin: this pins the
    # check's sensitivity, which the old self-fulfilling test never exercised.
    flat = np.full_like(j_band, float(np.median(j_band)))
    err_flat = whitening_identity_error(val, flat, cfg)
    assert err_flat > 2.0, f"flat J should not whiten 1/f noise (err {err_flat:.3f})"


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
