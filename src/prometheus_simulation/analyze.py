"""Headless analysis of a produced event set.

Everything the notebook does in Parts 2-5, as one command, so a remote agent
can run the experiment end to end and hand back files rather than a screenshot.

    python -m src.prometheus_simulation.analyze --run runs/pilot

Writes into <run>/analysis/:
    REPORT.md                 the summary a human reads first
    pairing.csv               the gate: every arm must be paired
    summary_by_arm.csv        one row per geometry
    summary_by_point.csv      one row per geometry x injection point
    light_yield.csv           median hits and the point-spread ratio
    null_separation.csv       geometry effect vs the photon-seed floor
    fig_*.png                 the same figures as the notebook

Exit code is non-zero if a gate fails, so it can be used in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .readout import check_pairing, load_event_set
from . import recon

PALETTE = ["#008B7A", "#C2410C", "#6D28A8", "#2E6FB0", "#A8325A", "#4D7C0F"]
INK, INK2, GRID, SURFACE = "#1c1c1a", "#4a4a46", "#e4e4df", "#fcfcfb"


def _style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "legend.frameon": False,
        "figure.dpi": 120, "lines.linewidth": 2,
    })


def null_separation(df: pd.DataFrame, plan: dict, arms: list[str]) -> pd.DataFrame:
    """Geometry effect measured in units of the photon-stochasticity floor.

    The null arm is the reference geometry re-run with the same events and a
    different photon seed. Its spread is the scale; a cross-geometry difference
    only means something as a multiple of it.
    """
    null_arm = next((a["arm"] for a in plan["arms"] if a["role"] == "photon_null"), None)
    ref = plan["reference_geometry"]
    lv = df.index.get_level_values("arm")
    if null_arm is None or null_arm not in lv:
        return pd.DataFrame()
    base = df.xs(ref, level="arm")["n_hits"]
    null = df.xs(null_arm, level="arm")["n_hits"]
    c = base.index.intersection(null.index)
    d_null = (np.abs(null.loc[c] - base.loc[c]) / np.maximum(base.loc[c], 1)).to_numpy()
    rows = []
    for k in arms:
        if k == ref or k not in lv:
            continue
        other = df.xs(k, level="arm")["n_hits"]
        c2 = base.index.intersection(other.index)
        a = (np.abs(other.loc[c2] - base.loc[c2]) / np.maximum(base.loc[c2], 1)).to_numpy()
        p95 = float(np.percentile(d_null, 95))
        rows.append({
            "arm": k,
            "median_cross": float(np.median(a)),
            "null_p95": p95,
            "ratio_to_null_p95": float(np.median(a) / p95) if p95 > 0 else np.inf,
            "auc_vs_null": float((a[:, None] > d_null[None, :]).mean()),
        })
    out = pd.DataFrame(rows).set_index("arm")
    out["separated"] = out.auc_vs_null > 0.75
    return out


def _figures(df, ly, effect, arms, points, out_dir, energy):
    import matplotlib.pyplot as plt
    color = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(arms)}

    # errors by injection point
    fig, axes = plt.subplots(2, len(points), figsize=(4 * len(points), 7), squeeze=False)
    for j, pt in enumerate(points):
        for i, (col, label, logx) in enumerate(
                [("vertex_error_m_early", "vertex error [m]", True),
                 ("angular_error_deg", "angular error [deg]", False)]):
            ax = axes[i][j]
            for k in arms:
                sel = df[(df.index.get_level_values("arm") == k) & (df["point"] == pt)]
                v = sel[col].dropna()
                v = v[v > 0] if logx else v
                if len(v) < 5:
                    continue
                bins = (np.logspace(np.log10(v.min()), np.log10(v.max()), 24)
                        if logx else np.linspace(0, 180, 24))
                ax.hist(v, bins=bins, histtype="step", lw=2, color=color[k], label=k)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(label); ax.set_ylabel("events")
            if i == 0:
                ax.set_title(f"point: {pt}", fontsize=10)
    axes[0][0].legend(fontsize=8, ncol=2)
    fig.suptitle("Same events, same estimator — geometry across columns of fixed vertex",
                 y=1.01, color=INK)
    fig.tight_layout(); fig.savefig(out_dir / "fig_errors_by_point.png", bbox_inches="tight")
    plt.close(fig)

    # light yield
    if not ly.empty:
        piv = ly.pivot(index="arm", columns="point", values="median").reindex(arms)
        lo = ly.pivot(index="arm", columns="point", values="q16").reindex(arms)
        hi = ly.pivot(index="arm", columns="point", values="q84").reindex(arms)
        x = np.arange(len(piv)); w = 0.8 / max(len(points), 1)
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        for j, pt in enumerate(points):
            if pt not in piv:
                continue
            off = (j - (len(points) - 1) / 2) * w
            vals = piv[pt].to_numpy(dtype=float)
            err = np.abs(np.vstack([vals - lo[pt].to_numpy(dtype=float),
                                    hi[pt].to_numpy(dtype=float) - vals]))
            ax.bar(x + off, vals, width=w * 0.9, color=PALETTE[j % len(PALETTE)],
                   edgecolor=SURFACE, linewidth=2, label=pt)
            ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor=INK2,
                        elinewidth=1.2, capsize=2)
        ax.set_yscale("log"); ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=15)
        ax.set_ylabel("median hits per event  (16–84%)")
        ax.set_title(f"Light yield at {energy:,.0f} GeV — only vertex and geometry change")
        ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / "fig_light_yield.png", bbox_inches="tight")
        plt.close(fig)

    # null separation
    if not effect.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        y = np.arange(len(effect))
        ax.barh(y, effect.auc_vs_null, height=0.55,
                color=[color.get(k, PALETTE[0]) for k in effect.index],
                edgecolor=SURFACE, linewidth=2)
        ax.axvline(0.75, ls="--", lw=1.6, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(effect.index); ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUC: geometry change vs photon seed alone")
        ax.set_title("Is the geometry effect above the photon-stochasticity floor?")
        fig.tight_layout(); fig.savefig(out_dir / "fig_null_separation.png",
                                        bbox_inches="tight")
        plt.close(fig)


def evaluate_gates(pairing: pd.DataFrame, plan: dict) -> "tuple[dict, list[str]]":
    """The two gates, and any messages explaining a failure.

    A gate that passes when the evidence is ABSENT is not a gate. An earlier
    parallel run path omitted ``vertex_residual_max_m`` from plan.json and this
    treated the omission as a pass, so the check was vacuous exactly where it
    was least likely to be noticed. Missing evidence now fails.
    """
    vres = plan.get("vertex_residual_max_m")
    gates = {
        "all_arms_paired": bool(pairing.paired.all()),
        "vertices_on_points": (vres is not None) and (float(vres) < 1e-6),
    }
    msgs = []
    if vres is None:
        msgs.append(
            "GATE vertices_on_points: FAILED - plan.json has no "
            "vertex_residual_max_m, so it is unknown whether the vertices "
            "landed on the injection points. Regenerate with a version that "
            "records it (simulate.prepare_injection always does).")
    elif not gates["vertices_on_points"]:
        msgs.append(f"GATE vertices_on_points: FAILED - residual {vres} m")
    if not gates["all_arms_paired"]:
        bad = pairing.loc[~pairing.paired, "arm"].tolist()
        msgs.append(f"GATE all_arms_paired: FAILED - arms {bad}")
    return gates, msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--min-hits", type=int, default=4, help="NuBench trigger cut")
    a = ap.parse_args()

    out_dir = a.run / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    truth, hits, plan = load_event_set(a.run)
    arms = [x["arm"] for x in plan["arms"] if x["role"] == "geometry"]
    points = list(plan.get("injection_points", {})) or sorted(truth["point"].unique())

    pairing = check_pairing(truth, plan)
    pairing.to_csv(out_dir / "pairing.csv", index=False)
    vres = plan.get("vertex_residual_max_m")
    gates, gate_msgs = evaluate_gates(pairing, plan)
    for m in gate_msgs:
        print(m, file=sys.stderr)

    df = recon.reconstruct(truth, hits, min_hits=a.min_hits)
    by_arm = recon.summarise(df, min_hits=a.min_hits)
    by_point = recon.summarise(df, min_hits=a.min_hits, by=["arm", "point"])
    ly = recon.light_yield(df, min_hits=a.min_hits)
    effect = null_separation(df, plan, arms)

    by_arm.to_csv(out_dir / "summary_by_arm.csv")
    by_point.to_csv(out_dir / "summary_by_point.csv")
    ly.to_csv(out_dir / "light_yield.csv", index=False)
    if not effect.empty:
        effect.to_csv(out_dir / "null_separation.csv")

    _figures(df, ly, effect, arms, points, out_dir, plan.get("energy_gev", np.nan))

    rec_path = a.run / "physics_record.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    lines = [
        "# Cross-geometry event set — analysis report", "",
        f"- run: `{a.run}`",
        f"- design: {plan.get('design', 'n/a')}",
        f"- events: {plan.get('n_events')} "
        f"({plan.get('events_per_point')} x {len(points)} points) "
        f"at {plan.get('energy_gev')} GeV",
        f"- reference geometry: {plan.get('reference_geometry')}",
        f"- fingerprint: {rec.get('fingerprint', 'n/a')}",
        f"- prometheus commit: {rec.get('environment', {}).get('prometheus_commit', 'n/a')}",
        f"- host / libc: {rec.get('environment', {}).get('toolchain', {}).get('hostname', 'n/a')}"
        f" / {rec.get('environment', {}).get('toolchain', {}).get('libc', 'n/a')}"
        + (f" (container: {rec['environment']['toolchain']['container']})"
           if rec.get('environment', {}).get('toolchain', {}).get('container') else ""),
        f"- compiler: {rec.get('environment', {}).get('toolchain', {}).get('cxx', 'n/a')}",
        f"- jax: {(rec.get('environment', {}).get('jax') or {}).get('default_backend', 'not installed')}"
        f"  devices={(rec.get('environment', {}).get('jax') or {}).get('devices', [])}",
        "", "## Gates", "",
        f"- all arms paired: **{gates['all_arms_paired']}**",
        f"- vertices on the injection points: **{gates['vertices_on_points']}** "
        + (f"(max residual {vres})" if vres is not None
           else "(NOT RECORDED — gate cannot be satisfied)"),
        "", "## Per geometry", "", by_arm.round(3).to_markdown(),
        "", "## Per geometry x injection point", "", by_point.round(3).to_markdown(),
        "", "## Light yield", "", ly.round(3).to_markdown(index=False),
    ]
    if not effect.empty:
        lines += ["", "## Geometry effect vs the photon-stochasticity floor", "",
                  effect.round(3).to_markdown(),
                  "", "AUC > 0.75 means the geometry change is separated from photon "
                  "noise at this statistics. Values near 0.5 mean it is not, and the "
                  "arm needs more events before anything is claimed from it."]
    if rec.get("unresolved_parameters"):
        lines += ["", "## Declared deviations (unpublished in NuBench)", ""]
        lines += [f"- `{k}` = {rec['physics'].get(k)}" for k in rec["unresolved_parameters"]]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {out_dir}")
    print(json.dumps(gates, indent=2))
    if not all(gates.values()):
        print("GATE FAILED — do not interpret the results above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
