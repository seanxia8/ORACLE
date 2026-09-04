"""End-to-end test of readout -> pairing check -> reconstruction, on a
synthetic event set written in the real Prometheus output schema.

Prometheus writes one parquet per run with two record fields
(`prometheus/prometheus.py:construct_output`):

    mc_truth  interaction, initial_state_energy, initial_state_type,
              initial_state_zenith, initial_state_azimuth, initial_state_x/y/z
    photons   sensor_pos_x/y/z, string_id, sensor_id, t, id_idx  (jagged)

The fixture below builds exactly that, for several arms sharing one set of
events, so the whole analysis path is exercised without a Prometheus install.
Hits are generated along the true track from the true vertex, so the baseline
estimators have a real signal to recover -- a test that only checked shapes
would pass on noise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

ak = pytest.importorskip("awkward")
pytest.importorskip("pyarrow")

from prometheus_simulation.readout import check_pairing, load_event_set  # noqa: E402
from prometheus_simulation import recon  # noqa: E402

POINTS = {"centre": [0.0, 0.0, 0.0], "radial": [40.0, 0.0, 0.0],
          "vertical": [0.0, 0.0, 70.0]}
PER_POINT = 8
N = PER_POINT * len(POINTS)
REF_OFFSET = np.array([0.65, 0.60, -3090.0])
ARMS = {                     # arm -> (offset, hits per event)
    "flower_xl": (REF_OFFSET, 40),
    "hexagon": (np.array([5.9, -2.5, -1972.0]), 25),
    "triangle": (np.array([0.0, 0.0, 0.0]), 6),
}
NULL_ARM = "flower_xl__seed2"


def _truth(rng):
    names = list(POINTS)
    assignment = [n for n in names for _ in range(PER_POINT)]
    zen = rng.uniform(0.2, np.pi - 0.2, N)
    azi = rng.uniform(0, 2 * np.pi, N)
    rel = np.array([POINTS[n] for n in assignment])
    return assignment, zen, azi, rel


def _write_arm(path, offset, n_hits, zen, azi, rel, rng, jitter=0.0):
    """Hits strung along the true track from the true vertex."""
    vert = rel + offset
    direction = np.stack([np.sin(zen) * np.cos(azi),
                          np.sin(zen) * np.sin(azi), np.cos(zen)], axis=1)
    xs, ys, zs, ts, sid, oid = [], [], [], [], [], []
    for i in range(len(vert)):
        s = np.sort(rng.uniform(0, 120, n_hits))          # metres along track
        p = vert[i] + s[:, None] * direction[i]
        p = p + rng.normal(0, 3.0 + jitter, p.shape)
        xs.append(p[:, 0]); ys.append(p[:, 1]); zs.append(p[:, 2])
        ts.append(s / 0.3 + rng.normal(0, 1.0, n_hits))   # ns, causal ordering
        sid.append(np.arange(n_hits) // 5)
        oid.append(np.arange(n_hits))
    arr = ak.Array({
        "mc_truth": ak.Array({
            "interaction": np.ones(len(vert), dtype=np.int64),
            "initial_state_energy": np.full(len(vert), 1e4),
            "initial_state_type": np.full(len(vert), 14, dtype=np.int64),
            "initial_state_zenith": zen,
            "initial_state_azimuth": azi,
            # ascontiguousarray: a column slice of `vert` is strided, and
            # ak.to_parquet rejects non-contiguous buffers.
            "initial_state_x": np.ascontiguousarray(vert[:, 0]),
            "initial_state_y": np.ascontiguousarray(vert[:, 1]),
            "initial_state_z": np.ascontiguousarray(vert[:, 2]),
        }),
        "photons": ak.Array({
            "sensor_pos_x": xs, "sensor_pos_y": ys, "sensor_pos_z": zs,
            "t": ts, "string_id": sid, "sensor_id": oid,
            "id_idx": [list(range(n_hits))] * len(vert),
        }),
    })
    path.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(arr, path / "out.parquet")


@pytest.fixture
def run_dir(tmp_path):
    rng = np.random.default_rng(7)
    assignment, zen, azi, rel = _truth(rng)
    arms_meta = []
    for arm, (off, nh) in ARMS.items():
        _write_arm(tmp_path / arm, off, nh, zen, azi, rel, rng)
        arms_meta.append({
            "arm": arm, "role": "geometry", "medium": "water",
            "recentre_delta_m": (off - REF_OFFSET).tolist(),
            "n_modules": 100, "n_strings": 10,
        })
    # photon null: same events, same geometry, different photon realisation
    _write_arm(tmp_path / NULL_ARM, REF_OFFSET, 40, zen, azi, rel,
               np.random.default_rng(99), jitter=0.5)
    arms_meta.append({"arm": NULL_ARM, "role": "photon_null", "medium": "water",
                      "recentre_delta_m": [0.0, 0.0, 0.0],
                      "n_modules": 100, "n_strings": 10})

    (tmp_path / "plan.json").write_text(json.dumps({
        "reference_geometry": "flower_xl",
        "reference_offset_m": REF_OFFSET.tolist(),
        "injection_points": POINTS,
        "event_point_assignment": assignment,
        "events_per_point": PER_POINT,
        "n_events": N,
        "energy_gev": 1e4,
        "arms": arms_meta,
    }))
    return tmp_path


def test_load_event_set_shapes_and_point_labels(run_dir):
    truth, hits, plan = load_event_set(run_dir)
    assert truth.arm.nunique() == len(ARMS) + 1
    assert len(truth) == N * (len(ARMS) + 1)
    for arm, (_, nh) in ARMS.items():
        sub = truth[truth.arm == arm]
        assert (sub.n_hits == nh).all()
        assert sub.point.value_counts().to_dict() == {k: PER_POINT for k in POINTS}
    assert set(hits.point.unique()) == set(POINTS)


def test_check_pairing_passes_on_a_correctly_paired_set(run_dir):
    truth, _, plan = load_event_set(run_dir)
    table = check_pairing(truth, plan)
    assert table.paired.all(), table


def test_check_pairing_catches_a_broken_pairing(run_dir):
    """The gate has to actually fail when the arms are not the same events."""
    truth, _, plan = load_event_set(run_dir)
    bad = truth.copy()
    m = bad.arm == "hexagon"
    bad.loc[m, "vertex_x"] = bad.loc[m, "vertex_x"] + 5.0
    table = check_pairing(bad, plan)
    assert not table.paired.all()
    assert not table.set_index("arm").loc["hexagon", "paired"]


def test_reconstruction_recovers_the_truth(run_dir):
    truth, hits, plan = load_event_set(run_dir)
    df = recon.reconstruct(truth, hits)
    # the early-hit vertex must beat the full centroid, which trails the track
    assert df["vertex_error_m_early"].median() < df["vertex_error_m"].median()
    # PCA direction on a straight track must be far better than random (90 deg)
    assert df["angular_error_deg"].median() < 20.0
    assert df["elongation"].median() > 0.8


def test_summarise_by_arm_and_by_point(run_dir):
    truth, hits, plan = load_event_set(run_dir)
    df = recon.reconstruct(truth, hits)
    s = recon.summarise(df)
    assert set(s.index) == set(ARMS) | {NULL_ARM}
    assert (s.trigger_efficiency == 1.0).all()

    sp = recon.summarise(df, by=["arm", "point"])
    assert len(sp) == (len(ARMS) + 1) * len(POINTS)
    assert sp.n_events.eq(PER_POINT).all()


def test_light_yield_reports_point_spread(run_dir):
    truth, hits, plan = load_event_set(run_dir)
    df = recon.reconstruct(truth, hits)
    ly = recon.light_yield(df)
    assert set(ly.columns) >= {"arm", "point", "median", "q16", "q84",
                               "point_spread_ratio"}
    # the fixture gives every point the same yield within an arm
    assert np.allclose(ly.point_spread_ratio.to_numpy(), 1.0)
    # and the arms differ from each other, which is the geometry effect
    assert ly.groupby("arm")["median"].first().nunique() > 1
