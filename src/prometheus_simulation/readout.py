"""Read a produced event set into tidy frames.

Prometheus writes one parquet per run with two record fields (see
``prometheus/prometheus.py:construct_output``):

    mc_truth   interaction, initial_state_energy, initial_state_type,
               initial_state_zenith, initial_state_azimuth,
               initial_state_x/y/z, final_state_* ...
    photons    sensor_pos_x, sensor_pos_y, sensor_pos_z, string_id,
               sensor_id, t, id_idx        (jagged: one list per event)

`load_event_set` returns
    truth  : DataFrame, one row per event per arm, indexed by (arm, event_id)
    hits   : DataFrame, one row per hit per event per arm

Event identity across arms is positional: arm A's event i and arm B's event i
are the same injected event, because every arm replayed the same injection
file. `check_pairing` verifies that rather than assuming it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parquet_files(arm_dir: Path) -> list[Path]:
    return sorted(p for p in arm_dir.glob("*.parquet"))


def load_arm(arm_dir: Path, arm: str, photon_field: str = "photons"
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    import awkward as ak   # noqa: PLC0415

    files = _parquet_files(arm_dir)
    if not files:
        raise FileNotFoundError(f"no parquet in {arm_dir}")
    arr = ak.concatenate([ak.from_parquet(f) for f in files])

    mc = arr["mc_truth"]
    truth = pd.DataFrame({
        "arm": arm,
        "event_id": np.arange(len(arr)),
        "energy_gev": ak.to_numpy(mc["initial_state_energy"]),
        "zenith_rad": ak.to_numpy(mc["initial_state_zenith"]),
        "azimuth_rad": ak.to_numpy(mc["initial_state_azimuth"]),
        "vertex_x": ak.to_numpy(mc["initial_state_x"]),
        "vertex_y": ak.to_numpy(mc["initial_state_y"]),
        "vertex_z": ak.to_numpy(mc["initial_state_z"]),
        "pdg": ak.to_numpy(mc["initial_state_type"]),
        "interaction": ak.to_numpy(mc["interaction"]),
    })

    if photon_field not in arr.fields:
        # Prometheus writes mc_truth only when NO event produced a hit.
        truth["n_hits"] = 0
        return truth, pd.DataFrame(
            columns=["arm", "event_id", "x", "y", "z", "t", "string_id", "sensor_id"])

    ph = arr[photon_field]
    counts = ak.to_numpy(ak.num(ph["t"]))
    truth["n_hits"] = counts
    event_id = np.repeat(np.arange(len(arr)), counts)
    hits = pd.DataFrame({
        "arm": arm,
        "event_id": event_id,
        "x": ak.to_numpy(ak.flatten(ph["sensor_pos_x"])),
        "y": ak.to_numpy(ak.flatten(ph["sensor_pos_y"])),
        "z": ak.to_numpy(ak.flatten(ph["sensor_pos_z"])),
        "t": ak.to_numpy(ak.flatten(ph["t"])),
        "string_id": ak.to_numpy(ak.flatten(ph["string_id"])),
        "sensor_id": ak.to_numpy(ak.flatten(ph["sensor_id"])),
    })
    return truth, hits


def load_event_set(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "plan.json").read_text())
    assignment = plan.get("event_point_assignment", [])
    truths, hits = [], []
    for arm in plan["arms"]:
        d = run_dir / arm["arm"]
        if not d.exists():
            continue
        t, h = load_arm(d, arm["arm"])
        t["role"] = arm["role"]
        t["point"] = [assignment[i] if i < len(assignment) else "?"
                      for i in t["event_id"]]
        t["medium"] = arm["medium"]
        t["n_modules"] = arm["n_modules"]
        t["n_strings"] = arm["n_strings"]
        h["point"] = [assignment[i] if i < len(assignment) else "?"
                      for i in h["event_id"]] if len(h) else []
        truths.append(t)
        hits.append(h)
    if not truths:
        raise FileNotFoundError(f"no arm output found under {run_dir}")
    return (pd.concat(truths, ignore_index=True),
            pd.concat(hits, ignore_index=True), plan)


def check_pairing(truth: pd.DataFrame, plan: dict, atol: float = 1e-6) -> pd.DataFrame:
    """Assert arm-to-arm event identity up to the recentring delta.

    If this fails, the arms are NOT the same events and nothing downstream is
    interpretable -- do not proceed to reconstruction.
    """
    deltas = {a["arm"]: np.asarray(a["recentre_delta_m"]) for a in plan["arms"]}
    ref = plan["reference_geometry"]
    base = truth[truth.arm == ref].set_index("event_id").sort_index()
    rows = []
    for arm, grp in truth.groupby("arm"):
        g = grp.set_index("event_id").sort_index()
        common = base.index.intersection(g.index)
        d = deltas.get(arm, np.zeros(3))
        dv = np.stack([
            (g.loc[common, f"vertex_{ax}"].to_numpy()
             - base.loc[common, f"vertex_{ax}"].to_numpy()) - d[i]
            for i, ax in enumerate("xyz")], axis=1)
        rows.append({
            "arm": arm,
            "n_events": len(g),
            "n_common": len(common),
            "max_vertex_residual_m": float(np.abs(dv).max()) if len(common) else np.nan,
            "max_energy_rel_residual": float(np.abs(
                g.loc[common, "energy_gev"].to_numpy()
                / base.loc[common, "energy_gev"].to_numpy() - 1).max()) if len(common) else np.nan,
            "max_zenith_residual_rad": float(np.abs(
                g.loc[common, "zenith_rad"].to_numpy()
                - base.loc[common, "zenith_rad"].to_numpy()).max()) if len(common) else np.nan,
            "mean_n_hits": float(g["n_hits"].mean()),
            "trigger_rate_ge4": float((g["n_hits"] >= 4).mean()),
            "n_points": int(g["point"].nunique()) if "point" in g else 0,
        })
    out = pd.DataFrame(rows)
    out["paired"] = (out.max_vertex_residual_m < atol) & \
                    (out.max_energy_rel_residual < atol) & \
                    (out.max_zenith_residual_rad < atol)
    return out
