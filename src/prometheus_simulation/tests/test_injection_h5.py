"""Tests for the h5 vertex surgery — the one code path that can silently
corrupt the pairing.

These build a synthetic LeptonInjector file with the structure the Prometheus
source actually reads (`injection/injection/LI_injection.py` and
`injection/lepton_injector_utils.py:apply_detector_offset`): one top-level
injector group holding compound datasets `initial`, `final_1`, `final_2` with a
`Position` field, plus `properties` with scalar x/y/z.

Getting this wrong is not loud. A wrong key silently leaves a final state
behind while the vertex moves, and the run still produces plausible-looking
output with the interaction pulled apart. So it is tested against the
structure rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from prometheus_simulation.simulate import (  # noqa: E402
    read_vertices, recentre_injection, set_vertices, _shift_h5,
)

N = 7
PARTICLE_DT = np.dtype([
    ("initial_type", "i4"),
    ("Position", "f8", (3,)),
    ("Direction", "f8", (2,)),
    ("Energy", "f8"),
])
PROPERTIES_DT = np.dtype([
    ("x", "f8"), ("y", "f8"), ("z", "f8"),
    ("totalEnergy", "f8"), ("zenith", "f8"), ("azimuth", "f8"),
])


def make_li_file(path, rng=None):
    """A minimal file with the real LI layout."""
    rng = rng or np.random.default_rng(0)
    pos = rng.normal(0, 50, (N, 3))
    with h5py.File(path, "w") as f:
        grp = f.create_group("VolumeInjector0")
        for key, offset in (("initial", 0.0), ("final_1", 0.0), ("final_2", 0.0)):
            arr = np.zeros(N, dtype=PARTICLE_DT)
            arr["Position"] = pos + offset
            arr["Direction"] = rng.uniform(0, np.pi, (N, 2))
            arr["Energy"] = 1e4
            grp.create_dataset(key, data=arr)
        props = np.zeros(N, dtype=PROPERTIES_DT)
        props["x"], props["y"], props["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
        props["totalEnergy"] = 1e4
        grp.create_dataset("properties", data=props)
    return pos


def read_all(path):
    with h5py.File(path, "r") as f:
        grp = f[list(f.keys())[0]]
        return (
            {k: np.asarray(grp[k]["Position"]) for k in ("initial", "final_1", "final_2")},
            np.stack([np.asarray(grp["properties"][a]) for a in "xyz"], axis=1),
        )


def test_shift_scalar_moves_every_group(tmp_path):
    p = tmp_path / "inj.h5"
    pos = make_li_file(p)
    delta = np.array([1.0, -2.0, 3.0])
    _shift_h5(p, delta)
    groups, props = read_all(p)
    for key, arr in groups.items():
        np.testing.assert_allclose(arr, pos + delta, err_msg=f"group {key} not shifted")
    np.testing.assert_allclose(props, pos + delta)


def test_shift_per_event_is_row_wise(tmp_path):
    p = tmp_path / "inj.h5"
    pos = make_li_file(p)
    delta = np.arange(N * 3, dtype=float).reshape(N, 3)
    _shift_h5(p, delta)
    groups, props = read_all(p)
    for arr in groups.values():
        np.testing.assert_allclose(arr, pos + delta)
    np.testing.assert_allclose(props, pos + delta)


def test_set_vertices_lands_exactly_on_the_points(tmp_path):
    """The acceptance criterion the agent prompt gates on."""
    p = tmp_path / "inj.h5"
    make_li_file(p)
    points = {"centre": [0.0, 0.0, 0.0], "radial": [40.0, 0.0, 0.0],
              "vertical": [0.0, 0.0, 70.0]}
    names = list(points)
    assignment = [names[i % 3] for i in range(N)]
    ref_offset = np.array([0.65, 0.60, -3090.0])
    targets = np.array([ref_offset + np.asarray(points[n]) for n in assignment])

    set_vertices(p, targets)
    np.testing.assert_allclose(read_vertices(p), targets, atol=1e-9)

    groups, props = read_all(p)
    # every final state moved with the vertex: the interaction stays intact
    for key, arr in groups.items():
        np.testing.assert_allclose(arr, targets, atol=1e-9, err_msg=key)
    np.testing.assert_allclose(props, targets, atol=1e-9)

    residual = read_vertices(p) - ref_offset - np.array(
        [points[n] for n in assignment])
    assert np.abs(residual).max() < 1e-6


def test_recentre_composes_with_set_vertices(tmp_path):
    """set_vertices puts the event on the point in the REFERENCE frame;
    recentre_injection then carries it into another detector's frame. The two
    must compose so the point sits at the same detector-relative place."""
    p = tmp_path / "inj.h5"
    make_li_file(p)
    ref_offset = np.array([0.65, 0.60, -3090.0])
    geom_offset = np.array([5.9, -2.5, -1972.0])
    point = np.array([40.0, 0.0, 0.0])

    set_vertices(p, np.tile(ref_offset + point, (N, 1)))
    q = tmp_path / "inj_arm.h5"
    recentre_injection(p, q, geom_offset - ref_offset)

    np.testing.assert_allclose(read_vertices(q) - geom_offset,
                               np.tile(point, (N, 1)), atol=1e-9)
    # the reference file is the pairing key: it must be untouched
    np.testing.assert_allclose(read_vertices(p) - ref_offset,
                               np.tile(point, (N, 1)), atol=1e-9)


def test_set_vertices_rejects_shape_mismatch(tmp_path):
    p = tmp_path / "inj.h5"
    make_li_file(p)
    with pytest.raises(ValueError):
        set_vertices(p, np.zeros((N + 1, 3)))
