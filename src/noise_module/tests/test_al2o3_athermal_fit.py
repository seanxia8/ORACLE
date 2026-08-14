from __future__ import annotations

import numpy as np

from noise_module.fit_al2o3_athermal import (
    build_composite,
    build_psd_artifact,
    fit_all_components,
    load_all_components,
    load_asd,
    native_grid_psd,
    validate_against_total,
)


def test_fitted_composite_reproduces_total_noise_dat() -> None:
    components = load_all_components()
    fit_results = fit_all_components(components)
    composite = build_composite(fit_results)

    total_freq, total_asd = load_asd("total_noise.dat")
    diagnostics = validate_against_total(total_freq, total_asd, composite)

    assert diagnostics["max_frac_dev"] < 1e-6
    assert diagnostics["rms_frac_dev"] < 1e-6


def test_native_grid_sum_matches_total_noise_dat() -> None:
    components = load_all_components()
    freq, native_psd = native_grid_psd(components)

    total_freq, total_asd = load_asd("total_noise.dat")
    assert np.array_equal(freq, total_freq)
    assert np.allclose(np.sqrt(native_psd), total_asd, rtol=1e-9)


def test_psd_artifact_is_finite_on_target_grid(tmp_path) -> None:
    components = load_all_components()
    freq, native_psd = native_grid_psd(components)

    out_path = tmp_path / "artifact.npy"
    meta = build_psd_artifact(
        freq,
        native_psd,
        sampling_frequency=2000.0,
        n_samples=1024,
        out_path=out_path,
        alias_fold=True,
    )

    saved = np.load(out_path)
    assert saved.shape == (2, 513)
    assert np.all(np.isfinite(saved))
    assert np.all(saved[1] >= 0.0)
    assert meta["method"] == "alias_fold"
