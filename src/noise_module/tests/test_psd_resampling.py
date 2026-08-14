from __future__ import annotations

import numpy as np

from noise_module.psd_resampling import (
    alias_fold_psd_density,
    inband_resample_psd_density,
    load_psd_density,
    save_psd_density,
    synthetic_resample_psd_density,
)


def test_inband_resample_psd_density_uses_target_rfft_grid() -> None:
    source_f = np.linspace(0.0, 2_000_000.0, 2001)
    source_p = 1.0 + source_f / source_f[-1]

    target_f, target_p, metadata = inband_resample_psd_density(
        source_f,
        source_p,
        target_sampling_frequency=1_000_000.0,
        target_samples=1000,
    )

    assert target_f.shape == (501,)
    assert target_p.shape == target_f.shape
    assert np.isclose(target_f[-1], 500_000.0)
    assert np.allclose(target_p, 1.0 + target_f / source_f[-1])
    assert metadata["method"] == "inband_interpolation"


def test_alias_fold_psd_density_adds_available_alias_bands() -> None:
    source_f = np.linspace(0.0, 2_000_000.0, 2001)
    source_p = np.ones_like(source_f)

    target_f, target_p, metadata = alias_fold_psd_density(
        source_f,
        source_p,
        target_sampling_frequency=1_000_000.0,
        target_samples=10,
    )

    assert np.allclose(target_f, np.linspace(0.0, 500_000.0, 6))
    # Interior 100-400 kHz bins receive f, 1 MHz - f, 1 MHz + f, 2 MHz - f.
    assert np.allclose(target_p[1:-1], 4.0)
    # Boundary bins use unique symmetric aliases to avoid one-sided double count.
    assert np.isclose(target_p[0], 3.0)
    assert np.isclose(target_p[-1], 2.0)
    assert metadata["method"] == "alias_fold"


def test_synthetic_resample_psd_density_returns_valid_target_psd() -> None:
    source_f = np.linspace(0.0, 2000.0, 257)
    source_p = np.full_like(source_f, 2.0e-3)

    target_f, target_p, metadata = synthetic_resample_psd_density(
        source_f,
        source_p,
        source_sampling_frequency=4000.0,
        target_sampling_frequency=1000.0,
        target_samples=128,
        n_traces=4,
        seed=123,
    )

    assert target_f.shape == (65,)
    assert target_p.shape == target_f.shape
    assert np.all(np.isfinite(target_p))
    assert np.all(target_p >= 0.0)
    assert metadata["resample_up"] == 1
    assert metadata["resample_down"] == 4


def test_save_and_load_psd_density_round_trip(tmp_path) -> None:
    path = tmp_path / "psd.npy"
    frequencies = np.array([0.0, 1.0, 2.0])
    psd = np.array([3.0, 4.0, 5.0])

    save_psd_density(path, frequencies, psd)
    loaded_f, loaded_p = load_psd_density(path)

    assert np.allclose(loaded_f, frequencies)
    assert np.allclose(loaded_p, psd)
