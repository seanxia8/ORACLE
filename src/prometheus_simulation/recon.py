"""Deliberately simple, geometry-free baseline reconstructions.

The point is not accuracy. It is that the SAME estimator, applied to the SAME
events, gives different answers in different geometries -- and that difference
is the geometry effect, measured without a trained model anywhere in the loop.
A learned reconstruction can only be interpreted against this baseline.

    vertex     charge-weighted centroid of hit modules
    direction  principal axis of the hit cloud, signed by the time ordering
    energy     log of the total hit count, calibrated per geometry

Every estimator here is a linear or eigenvector operation on the hit cloud, so
none of them can memorise a geometry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _per_event(hits: pd.DataFrame):
    return hits.groupby(["arm", "event_id"], sort=True)


def vertex_centroid(hits: pd.DataFrame) -> pd.DataFrame:
    """Hit-count-weighted centroid. The crudest possible vertex estimate."""
    g = _per_event(hits)[["x", "y", "z"]].mean()
    return g.rename(columns={"x": "reco_x", "y": "reco_y", "z": "reco_z"})


def vertex_earliest(hits: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Centroid of the k earliest hits -- much closer to the true vertex than
    the full centroid, because late hits trail along the muon track."""
    h = hits.sort_values(["arm", "event_id", "t"])
    g = h.groupby(["arm", "event_id"], sort=True).head(k)
    out = g.groupby(["arm", "event_id"])[["x", "y", "z"]].mean()
    return out.rename(columns={"x": "reco_x_early", "y": "reco_y_early",
                               "z": "reco_z_early"})


def direction_pca(hits: pd.DataFrame, min_hits: int = 4) -> pd.DataFrame:
    """Principal axis of the hit cloud, oriented by the time ordering.

    For a track-like event the hits lie along the muon path, so the leading
    eigenvector of the hit covariance is the direction up to a sign; the sign
    comes from correlating the projection with the hit time.

    Fully vectorised over events. The per-event loop this replaces was 98% of
    reconstruction time (6.0 s of 6.1 s at 20k events) because it called
    ``np.cov`` once per event. Ragged event lengths are handled with
    ``np.add.reduceat`` over a sorted flat array, and the 3x3 eigenproblems are
    solved as one stacked ``np.linalg.eigh``. Results are identical to the loop
    version; ``tests/test_recon_vectorised.py`` pins that.
    """
    if len(hits) == 0:
        return pd.DataFrame(
            columns=["reco_dx", "reco_dy", "reco_dz", "elongation", "extent_m"],
            index=pd.MultiIndex.from_arrays([[], []], names=["arm", "event_id"]))

    h = hits.sort_values(["arm", "event_id"], kind="stable")
    keys = pd.MultiIndex.from_arrays([h["arm"].to_numpy(), h["event_id"].to_numpy()])
    codes, uniques = pd.factorize(keys, sort=False)
    counts = np.bincount(codes)
    keep = counts >= min_hits
    if not keep.any():
        return pd.DataFrame(
            columns=["reco_dx", "reco_dy", "reco_dz", "elongation", "extent_m"],
            index=pd.MultiIndex.from_arrays([[], []], names=["arm", "event_id"]))

    P = h[["x", "y", "z"]].to_numpy(dtype=float)
    tt = h["t"].to_numpy(dtype=float)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    # per-event means, broadcast back to hits
    sums = np.add.reduceat(P, starts, axis=0)
    n = counts[:, None].astype(float)
    means = sums / n
    Pc = P - np.repeat(means, counts, axis=0)
    tsum = np.add.reduceat(tt, starts)
    tc = tt - np.repeat(tsum / counts, counts)

    # covariance per event: the six unique products, ddof=1 to match np.cov
    prod = np.stack([Pc[:, i] * Pc[:, j]
                     for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))], axis=1)
    S = np.add.reduceat(prod, starts, axis=0) / np.maximum(counts - 1, 1)[:, None]
    C = np.empty((len(counts), 3, 3))
    C[:, 0, 0], C[:, 0, 1], C[:, 0, 2] = S[:, 0], S[:, 1], S[:, 2]
    C[:, 1, 0], C[:, 1, 1], C[:, 1, 2] = S[:, 1], S[:, 3], S[:, 4]
    C[:, 2, 0], C[:, 2, 1], C[:, 2, 2] = S[:, 2], S[:, 4], S[:, 5]

    w, V = np.linalg.eigh(C)                  # stacked 3x3, ascending eigenvalues
    axis = V[:, :, -1]

    # sign: the projection must increase with time (Pearson r > 0)
    proj = np.einsum("ij,ij->i", Pc, np.repeat(axis, counts, axis=0))
    cov_pt = np.add.reduceat(proj * tc, starts)
    var_p = np.add.reduceat(proj * proj, starts)
    var_t = np.add.reduceat(tc * tc, starts)
    denom = np.sqrt(var_p * var_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, cov_pt / denom, 0.0)
    axis = np.where((r < 0)[:, None], -axis, axis)

    # extent along the (possibly flipped) axis
    proj = np.einsum("ij,ij->i", Pc, np.repeat(axis, counts, axis=0))
    order = np.argsort(codes, kind="stable")   # already sorted; kept for clarity
    del order
    pmax = np.maximum.reduceat(proj, starts)
    pmin = np.minimum.reduceat(proj, starts)

    wsum = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        elong = np.where(wsum > 0, w[:, -1] / wsum, np.nan)

    idx = pd.MultiIndex.from_tuples(
        [uniques[i] for i in np.flatnonzero(keep)], names=["arm", "event_id"])
    return pd.DataFrame({
        "reco_dx": axis[keep, 0], "reco_dy": axis[keep, 1], "reco_dz": axis[keep, 2],
        "elongation": elong[keep], "extent_m": (pmax - pmin)[keep],
    }, index=idx)


def direction_pca_loop(hits: pd.DataFrame, min_hits: int = 4) -> pd.DataFrame:
    """Reference implementation kept only as the oracle for the vectorised one."""
    rows = []
    for (arm, eid), g in hits.groupby(["arm", "event_id"], sort=True):
        if len(g) < min_hits:
            continue
        P = g[["x", "y", "z"]].to_numpy()
        t = g["t"].to_numpy()
        Pc = P - P.mean(0)
        w, V = np.linalg.eigh(np.cov(Pc.T) if len(g) > 1 else np.eye(3))
        axis = V[:, -1]
        proj = Pc @ axis
        if np.corrcoef(proj, t)[0, 1] < 0:
            axis = -axis
            proj = -proj
        elongation = float(w[-1] / w.sum()) if w.sum() > 0 else np.nan
        rows.append({"arm": arm, "event_id": eid,
                     "reco_dx": axis[0], "reco_dy": axis[1], "reco_dz": axis[2],
                     "elongation": elongation,
                     "extent_m": float(proj.max() - proj.min())})
    return pd.DataFrame(rows).set_index(["arm", "event_id"])


def energy_proxy(truth: pd.DataFrame) -> pd.DataFrame:
    """log10(n_hits) as the raw energy observable, before any calibration."""
    out = truth.set_index(["arm", "event_id"])[["n_hits"]].copy()
    out["log_nhits"] = np.log10(out["n_hits"].clip(lower=1))
    return out


def reconstruct(truth: pd.DataFrame, hits: pd.DataFrame,
                min_hits: int = 4) -> pd.DataFrame:
    """Join truth, the three estimators, and their per-event errors."""
    t = truth.set_index(["arm", "event_id"])
    df = (t.join(vertex_centroid(hits))
            .join(vertex_earliest(hits))
            .join(direction_pca(hits, min_hits))
            .join(energy_proxy(truth)[["log_nhits"]]))

    # true direction unit vector from (zenith, azimuth)
    th, ph = df["zenith_rad"].to_numpy(), df["azimuth_rad"].to_numpy()
    true_dir = np.stack([np.sin(th) * np.cos(ph),
                         np.sin(th) * np.sin(ph),
                         np.cos(th)], axis=1)
    reco_dir = df[["reco_dx", "reco_dy", "reco_dz"]].to_numpy()
    with np.errstate(invalid="ignore"):
        cos = np.clip((true_dir * reco_dir).sum(1), -1, 1)
    df["angular_error_deg"] = np.degrees(np.arccos(cos))

    for tag, cols in (("", ["reco_x", "reco_y", "reco_z"]),
                      ("_early", ["reco_x_early", "reco_y_early", "reco_z_early"])):
        d = df[cols].to_numpy() - df[["vertex_x", "vertex_y", "vertex_z"]].to_numpy()
        df[f"vertex_error_m{tag}"] = np.linalg.norm(d, axis=1)
    return df


def light_yield(df: pd.DataFrame, min_hits: int = 4) -> pd.DataFrame:
    """Hit multiplicity per (arm, point) at the single fixed energy.

    With energy fixed there is no energy resolution to fit -- the observable is
    the light yield itself, and how it varies with vertex position within a
    geometry. The spread across the three points measures how position-
    dependent that geometry's response is; a detector whose yield barely
    changes between `centre` and `radial` is uniform over the common region,
    one whose yield halves is not.
    """
    g = df[df["n_hits"] >= min_hits]
    out = g.groupby(["arm", "point"])["n_hits"].agg(
        median="median", q16=lambda v: v.quantile(0.16),
        q84=lambda v: v.quantile(0.84), n="size")
    med = out["median"].unstack("point")
    out = out.reset_index()
    spread = (med.max(axis=1) / med.min(axis=1)).rename("point_spread_ratio")
    return out.merge(spread, left_on="arm", right_index=True, how="left")


def calibrate_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm linear fit of log10(E) on log10(n_hits), then the residual.

    Only meaningful for a SPECTRUM. With the fixed-energy design there is no
    energy variation to fit, so this returns an empty frame -- use
    `light_yield` instead. Kept for the spectrum-sampled configuration.
    """
    out = []
    for arm, g in df.groupby("arm"):
        m = np.isfinite(g["log_nhits"]) & (g["n_hits"] >= 4)
        if m.sum() < 10:
            continue
        y = np.log10(g.loc[m, "energy_gev"].to_numpy())
        x = g.loc[m, "log_nhits"].to_numpy()
        a, b = np.polyfit(x, y, 1)
        r = y - (a * x + b)
        out.append({
            "arm": arm, "slope": float(a), "intercept": float(b),
            "n_used": int(m.sum()),
            "energy_resolution_dex": float(np.std(r)),
            "r2": float(1 - np.var(r) / np.var(y)) if np.var(y) > 0 else np.nan,
        })
    return pd.DataFrame(out)


def summarise(df: pd.DataFrame, min_hits: int = 4,
              by: "list[str] | None" = None) -> pd.DataFrame:
    """The headline comparison table.

    ``by=["arm"]`` (default) gives one row per geometry. ``by=["arm","point"]``
    breaks it down by injection point, which is the comparison the fixed-point
    design was built for: the SAME three vertices at the SAME energy, so a
    difference between arms within a point is geometry and nothing else.
    """
    by = by or ["arm"]
    g = df[df["n_hits"] >= min_hits]
    s = g.groupby(by).agg(
        n_triggered=("n_hits", "size"),
        median_n_hits=("n_hits", "median"),
        median_vertex_error_m=("vertex_error_m_early", "median"),
        median_angular_error_deg=("angular_error_deg", "median"),
        median_elongation=("elongation", "median"),
    )
    total = df.groupby(by).size().rename("n_events")
    s = s.join(total)
    s["trigger_efficiency"] = s["n_triggered"] / s["n_events"]
    return s
