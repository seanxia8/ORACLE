"""The vectorised direction_pca must equal the loop it replaced.

An optimisation that changes the numbers is a bug, not a speed-up, so the loop
survives as `direction_pca_loop` purely to be the oracle here. Ragged event
lengths are the interesting case: the vectorised path uses reduceat over a
sorted flat array, which is exactly where an off-by-one would hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prometheus_simulation.recon import direction_pca, direction_pca_loop


def _tracks(n_events: int, seed: int, ragged: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        nh = int(rng.integers(4, 40)) if ragged else 20
        zen, azi = rng.uniform(0.2, np.pi - 0.2), rng.uniform(0, 2 * np.pi)
        d = np.array([np.sin(zen) * np.cos(azi), np.sin(zen) * np.sin(azi), np.cos(zen)])
        s = np.sort(rng.uniform(0, 120, nh))
        P = rng.normal(0, 40, 3) + s[:, None] * d + rng.normal(0, 3, (nh, 3))
        rows.append(pd.DataFrame({
            "arm": "arm_a" if e % 2 else "arm_b", "event_id": e,
            "x": P[:, 0], "y": P[:, 1], "z": P[:, 2],
            "t": s / 0.3 + rng.normal(0, 1.0, nh)}))
    return pd.concat(rows, ignore_index=True)


def _assert_same(hits, min_hits=4):
    got = direction_pca(hits, min_hits).sort_index()
    ref = direction_pca_loop(hits, min_hits).sort_index()
    assert list(got.index) == list(ref.index)
    for col in ("elongation", "extent_m"):
        np.testing.assert_allclose(got[col], ref[col], rtol=1e-9, atol=1e-9,
                                   err_msg=col)
    # direction is a unit vector; compare it as one, sign included
    g = got[["reco_dx", "reco_dy", "reco_dz"]].to_numpy()
    r = ref[["reco_dx", "reco_dy", "reco_dz"]].to_numpy()
    np.testing.assert_allclose(np.abs((g * r).sum(1)), 1.0, atol=1e-9)
    np.testing.assert_allclose((g * r).sum(1), 1.0, atol=1e-9,
                               err_msg="sign flipped relative to the loop")


def test_matches_loop_equal_length_events():
    _assert_same(_tracks(60, seed=1, ragged=False))


def test_matches_loop_ragged_events():
    _assert_same(_tracks(120, seed=2, ragged=True))


def test_respects_min_hits():
    hits = _tracks(40, seed=3, ragged=True)
    for mh in (4, 10, 25):
        _assert_same(hits, min_hits=mh)


def test_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=["arm", "event_id", "x", "y", "z", "t"])
    out = direction_pca(empty)
    assert len(out) == 0
    assert list(out.columns) == ["reco_dx", "reco_dy", "reco_dz", "elongation", "extent_m"]
