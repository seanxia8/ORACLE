#!/usr/bin/env python3
"""Restore a NuBench checkpoint and run a module-dropout development pilot.

This script is intended to run inside the official GraphNeT 1.8.0 CPU
container. It uses the trusted released pickled checkpoint and compatibility
aliases for GraphNeT module paths that changed after the checkpoint was saved.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
import types
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.loader import DataLoader


def install_graphnet_compatibility_aliases() -> None:
    """Alias three GraphNeT module paths used by the released pickle."""
    import graphnet.models.data_representation.graphs.edges.edges as edges
    import graphnet.models.data_representation.graphs.graph_definition as graph_def
    import graphnet.models.data_representation.graphs.nodes.nodes as nodes

    sys.modules["graphnet.models.graphs.graph_definition"] = graph_def
    sys.modules["graphnet.models.graphs.edges.edges"] = edges
    sys.modules["graphnet.models.graphs.nodes.nodes"] = nodes


install_graphnet_compatibility_aliases()

from graphnet.data.dataset import SQLiteDataset  # noqa: E402
from graphnet.models import Model  # noqa: E402


FEATURES = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "charge", "t"]
TRUTH = [
    "interaction",
    "initial_state_energy",
    "initial_state_type",
    "initial_state_zenith",
    "initial_state_azimuth",
]


def patch_released_detector(detector: Any) -> None:
    """Replace the released detector's five preprocessing methods wholesale.

    CAVEAT (audit C17): this substitutes hand-written re-implementations of the
    NuBench feature transforms for whatever the released pickle carried, and it
    is therefore itself a candidate cause of any re-inference disagreement with
    the released predictions — it must be listed alongside k-NN backend and
    version-skew hypotheses, not treated as neutral plumbing. The substitution
    (and the two guessed GraphDefinition defaults installed in ``main``) is
    recorded in ``smoke_test.json`` so no downstream reader can miss it.

    Decisive test for the open 0.944-degree parity blocker: run the identical
    re-inference twice, once with all tensors on CPU and once on GPU, same
    checkpoint and preprocessing. If CPU-vs-GPU alone reproduces the offset it
    is the torch_cluster k-NN backend; if the two agree with each other but
    both differ from the released parquet, the cause is preprocessing (this
    function), alignment (GraphNeT issue #880), or version skew.
    """

    def sensor_pos_xy(_self: Any, values: torch.Tensor) -> torch.Tensor:
        return values / 100.0

    def sensor_pos_z(_self: Any, values: torch.Tensor) -> torch.Tensor:
        return values / 1000.0

    def sensor_time(_self: Any, values: torch.Tensor) -> torch.Tensor:
        return values / 1_000_000.0

    def sensor_charge(_self: Any, values: torch.Tensor) -> torch.Tensor:
        return torch.log10(1.0 + torch.clamp(values, min=1e-2))

    def feature_map(self: Any) -> dict[str, Any]:
        return {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
            "charge": self._charge,
        }

    detector._sensor_pos_xy = types.MethodType(sensor_pos_xy, detector)
    detector._sensor_pos_z = types.MethodType(sensor_pos_z, detector)
    detector._t = types.MethodType(sensor_time, detector)
    detector._charge = types.MethodType(sensor_charge, detector)
    detector.feature_map = types.MethodType(feature_map, detector)
    detector._replace_with_identity = None


class ModuleDropoutSQLiteDataset(SQLiteDataset):
    """SQLiteDataset that removes complete observed sensors before graph creation."""

    def __init__(
        self,
        *args: Any,
        module_dropout_fraction: float,
        transform_seed: int,
        minimum_modules: int = 9,
        **kwargs: Any,
    ) -> None:
        self._module_dropout_fraction = module_dropout_fraction
        self._transform_seed = transform_seed
        self._minimum_modules = minimum_modules
        super().__init__(*args, **kwargs)

    def _query(self, sequential_index: int):  # type: ignore[no-untyped-def]
        features, truth, node_truth, loss_weight = super()._query(sequential_index)
        if self._module_dropout_fraction <= 0.0 or len(features) == 0:
            return features, truth, node_truth, loss_weight

        sensor_positions, inverse = np.unique(
            features[:, :3], axis=0, return_inverse=True
        )
        n_modules = len(sensor_positions)
        requested = int(round(self._module_dropout_fraction * n_modules))
        n_drop = min(requested, max(0, n_modules - self._minimum_modules))
        if n_drop == 0:
            return features, truth, node_truth, loss_weight

        event_no = int(np.asarray(truth).reshape(-1)[0])
        # Nested severities (audit C16): the permutation is seeded by
        # (transform_seed, event_no) ONLY — not by the severity — so the
        # dropped sets are prefixes of one fixed per-event module order.
        # A module lost at 10% dropout is therefore also lost at 25% and 50%,
        # removing perturbation-identity noise from the severity axis.
        rng = np.random.default_rng(
            np.random.SeedSequence([self._transform_seed, event_no])
        )
        permutation = rng.permutation(n_modules)
        dropped = permutation[:n_drop]
        keep = ~np.isin(inverse, dropped)
        return features[keep], truth, node_truth, loss_weight


def select_balanced_events(
    predictions: Path,
    database: Path,
    n_events: int,
    minimum_modules: int,
    seed: int,
) -> pd.DataFrame:
    """Select a balanced development sample with enough modules to perturb.

    Audit fix (C12): the previous version collected candidates from the HEAD of
    the parquet in file order and randomised only within that prefix, so the
    pilot was one contiguous slice of the test set. This version reads the full
    label column and samples the candidate pool uniformly at random over ALL
    rows before the module-count filter. The >=minimum_modules enrichment (and
    its interaction with the severity axis) remains and is disclosed in the
    output; see RESULT.md.
    """
    required = [
        "event_no",
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "is_track",
    ]
    target_each = n_events // 2
    candidate_each = max(1_000, target_each * 50)
    candidates_all = pq.read_table(predictions, columns=required).to_pandas()
    candidates_all["event_no"] = candidates_all["event_no"].astype(int)
    pools: list[pd.DataFrame] = []
    for offset, label in enumerate((True, False)):
        pool = candidates_all[candidates_all["is_track"].astype(bool) == label]
        take = min(candidate_each, len(pool))
        pools.append(pool.sample(n=take, random_state=seed + 100 + offset, replace=False))
    candidates = pd.concat(pools, ignore_index=True)

    module_counts: list[tuple[int, int]] = []
    ids = candidates["event_no"].astype(int).tolist()
    with sqlite3.connect(database) as connection:
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT event_no,
                       COUNT(DISTINCT printf('%.9g|%.9g|%.9g',
                                             sensor_pos_x,
                                             sensor_pos_y,
                                             sensor_pos_z)) AS n_modules
                FROM pulses_no_noise
                WHERE event_no IN ({placeholders})
                GROUP BY event_no
            """
            module_counts.extend(connection.execute(query, chunk).fetchall())

    counts_frame = pd.DataFrame(
        module_counts, columns=["event_no", "observed_modules"]
    )
    candidates = candidates.merge(counts_frame, on="event_no", how="inner")
    candidates = candidates[candidates["observed_modules"] >= minimum_modules]
    selected_frames: list[pd.DataFrame] = []
    for offset, label in enumerate((True, False)):
        pool = candidates[candidates["is_track"].astype(bool) == label]
        if len(pool) < target_each:
            raise RuntimeError(
                f"Only {len(pool)} label={label} events have at least "
                f"{minimum_modules} observed modules; need {target_each}"
            )
        selected_frames.append(
            pool.sample(n=target_each, random_state=seed + offset, replace=False)
        )
    result = pd.concat(selected_frames, ignore_index=True)
    return result.sort_values("event_no").reset_index(drop=True)


def make_dataset(
    database: Path,
    data_representation: Any,
    event_ids: list[int],
    dropout: float,
    seed: int,
    minimum_modules: int,
) -> ModuleDropoutSQLiteDataset:
    """Construct the paired clean or module-dropout dataset.

    ``minimum_modules`` is the DATASET-side floor below which dropout is capped
    (kept >= 9 so the 8-NN graph stays defined). It is now forwarded explicitly
    (audit C13) instead of silently defaulting while the selection-side
    ``--minimum-observed-modules`` threshold said something else. Note the two
    thresholds have different purposes: the floor guarantees graph validity for
    ANY event; the selection threshold (>= 20 by default) additionally keeps
    the realised severity equal to the nominal severity for the whole sample.
    """
    return ModuleDropoutSQLiteDataset(
        path=str(database),
        pulsemaps="pulses_no_noise",
        features=FEATURES,
        truth=TRUTH,
        truth_table="mc_truth",
        data_representation=data_representation,
        selection=event_ids,
        module_dropout_fraction=dropout,
        transform_seed=seed,
        minimum_modules=minimum_modules,
    )


def infer(
    model: Model,
    dataset: SQLiteDataset,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    """Run deterministic CPU inference and capture the backbone embedding."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    event_ids: list[torch.Tensor] = []
    zenith: list[torch.Tensor] = []
    azimuth: list[torch.Tensor] = []

    def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        embeddings.append(output.detach().cpu())

    handle = model.backbone.register_forward_hook(capture)
    model.inference()
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch)[0]
            predictions.append(output.detach().cpu())
            event_ids.append(batch.event_no.detach().cpu().reshape(-1))
            zenith.append(batch.initial_state_zenith.detach().cpu().reshape(-1))
            azimuth.append(batch.initial_state_azimuth.detach().cpu().reshape(-1))
    elapsed = time.perf_counter() - started
    handle.remove()
    return {
        "prediction": torch.cat(predictions).numpy(),
        "embedding": torch.cat(embeddings).numpy(),
        "event_no": torch.cat(event_ids).numpy().astype(int),
        "zenith": torch.cat(zenith).numpy(),
        "azimuth": torch.cat(azimuth).numpy(),
        "elapsed_seconds": elapsed,
    }


def direction_vectors(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Convert spherical truth to Cartesian unit vectors."""
    return np.column_stack(
        (
            np.sin(zenith) * np.cos(azimuth),
            np.sin(zenith) * np.sin(azimuth),
            np.cos(zenith),
        )
    )


def angle_between_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Calculate row-wise opening angles in degrees."""
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    cosine = np.clip(np.einsum("ij,ij->i", first, second), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def bootstrap_median_ci(
    values: np.ndarray, indices: np.ndarray
) -> tuple[float, float]:
    """Percentile bootstrap CI for a median, over PRE-DRAWN event indices.

    Audit fix (C15): the resample indices are drawn once in ``main`` and shared
    across all severities, so the design is a genuinely paired bootstrap — the
    same resampled event sets are scored at every severity — instead of four
    independent marginal resamples whose side-by-side intervals invite a paired
    inference they do not license.
    """
    medians = np.median(values[indices], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_paired_delta_median_ci(
    values: np.ndarray, reference: np.ndarray, indices: np.ndarray
) -> tuple[float, float, float]:
    """(delta, low, high) for median(values) - median(reference), paired.

    This is the interval that actually supports a "severity X degrades the
    metric" claim in the paired design; the marginal CIs do not.
    """
    deltas = np.median(values[indices], axis=1) - np.median(reference[indices], axis=1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = float(np.median(values) - np.median(reference))
    return point, float(low), float(high)


def bootstrap_mean_ci(
    values: np.ndarray, indices: np.ndarray
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean, over the shared event indices."""
    means = np.mean(values[indices], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def knn_retention_per_event(
    clean: np.ndarray, shifted: np.ndarray, k: int = 10
) -> np.ndarray:
    """Per-event fraction of clean k-nearest neighbors retained after perturbation.

    Returns the per-event array (audit C15: the mean was previously reported as
    a bare scalar with no uncertainty while its table neighbours carried CIs).
    The bootstrap CI computed from this array treats the neighbour graph as
    fixed and resamples events — an approximation, since resampling would also
    change the graph; it is labelled as such in the output.
    """
    effective_k = min(k + 1, len(clean))
    clean_neighbors = NearestNeighbors(n_neighbors=effective_k).fit(clean)
    shifted_neighbors = NearestNeighbors(n_neighbors=effective_k).fit(shifted)
    clean_idx = clean_neighbors.kneighbors(clean, return_distance=False)[:, 1:]
    shifted_idx = shifted_neighbors.kneighbors(shifted, return_distance=False)[:, 1:]
    retained = [
        len(set(a.tolist()).intersection(b.tolist())) / max(1, len(a))
        for a, b in zip(clean_idx, shifted_idx)
    ]
    return np.asarray(retained, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--released-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-events", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--minimum-observed-modules", type=int, default=20)
    parser.add_argument(
        "--dataset-minimum-modules",
        type=int,
        default=9,
        help="dataset-side dropout floor keeping the 8-NN graph defined; "
        "distinct from --minimum-observed-modules (selection-side, keeps the "
        "realised severity equal to nominal for the whole sample) — audit C13",
    )
    parser.add_argument(
        "--dropout-fractions", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5]
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    released = select_balanced_events(
        args.released_predictions,
        args.database,
        args.n_events,
        args.minimum_observed_modules,
        args.seed,
    )
    event_ids = released["event_no"].astype(int).tolist()

    model = Model.load(str(args.checkpoint))
    data_representation = model._graph_definition
    patch_released_detector(data_representation._detector)
    if not hasattr(data_representation, "_repeat_labels"):
        data_representation._repeat_labels = False
    if not hasattr(data_representation, "_add_static_features"):
        data_representation._add_static_features = True
    # The released config contains training-time charge/time jitter. Published
    # test predictions use pulses_no_noise without this augmentation.
    data_representation._perturbation_dict = None

    results: dict[float, dict[str, np.ndarray | float]] = {}
    for dropout in args.dropout_fractions:
        print(f"Running module dropout={dropout:.3f}", flush=True)
        dataset = make_dataset(
            args.database,
            data_representation,
            event_ids,
            dropout,
            args.seed,
            args.dataset_minimum_modules,
        )
        results[dropout] = infer(model, dataset, args.batch_size)

    clean = results[0.0]
    # Alignment guard (audit C14): every severity's row order must match the
    # clean run exactly — the displacement and retention metrics below are
    # strictly row-paired subtractions, and a loader that reorders or drops an
    # event under `selection=` would otherwise corrupt them with no symptom.
    clean_event_order = np.asarray(clean["event_no"])
    for dropout, result in results.items():
        if not np.array_equal(np.asarray(result["event_no"]), clean_event_order):
            raise RuntimeError(
                f"event order at dropout={dropout} does not match the clean run; "
                "paired metrics would silently compare different events "
                "(cf. GraphNeT issue #880)"
            )
    clean_order = pd.DataFrame({"event_no": clean["event_no"]}).reset_index()
    release_order = clean_order.merge(released, on="event_no", how="left")
    released_vector = release_order[
        ["dir_x_pred", "dir_y_pred", "dir_z_pred"]
    ].to_numpy()
    clean_prediction = np.asarray(clean["prediction"])[:, :3]
    truth_vector = direction_vectors(
        np.asarray(clean["zenith"]), np.asarray(clean["azimuth"])
    )
    checkpoint_release_angle = angle_between_deg(clean_prediction, released_vector)

    clean_embedding = np.asarray(clean["embedding"])
    embedding_scale = np.std(clean_embedding, axis=0, ddof=1)
    embedding_scale = np.where(embedding_scale > 1e-8, embedding_scale, 1.0)
    clean_standardized = (clean_embedding - clean_embedding.mean(axis=0)) / embedding_scale
    rng = np.random.default_rng(args.seed)

    # Shared bootstrap indices: ONE set of event resamples scores every
    # severity, making all CIs and the paired-difference CIs below genuinely
    # paired (audit C15).
    n_events_run = len(clean_event_order)
    replicates = 1_000
    boot_indices = rng.integers(0, n_events_run, size=(replicates, n_events_run))
    clean_angle = angle_between_deg(
        truth_vector, np.asarray(clean["prediction"])[:, :3]
    )

    summary_rows: list[dict[str, float | int | list[float]]] = []
    for dropout in args.dropout_fractions:
        result = results[dropout]
        prediction = np.asarray(result["prediction"])[:, :3]
        embedding = np.asarray(result["embedding"])
        angle = angle_between_deg(truth_vector, prediction)
        prediction_drift = angle_between_deg(clean_prediction, prediction)
        displacement = np.linalg.norm(
            (embedding - clean_embedding) / embedding_scale, axis=1
        ) / math.sqrt(clean_embedding.shape[1])
        shifted_standardized = (
            embedding - clean_embedding.mean(axis=0)
        ) / embedding_scale
        angle_ci = bootstrap_median_ci(angle, boot_indices)
        displacement_ci = bootstrap_median_ci(displacement, boot_indices)
        delta, delta_low, delta_high = bootstrap_paired_delta_median_ci(
            angle, clean_angle, boot_indices
        )
        retention = knn_retention_per_event(
            clean_standardized, shifted_standardized, k=10
        )
        retention_ci = bootstrap_mean_ci(retention, boot_indices)
        summary_rows.append(
            {
                "module_dropout_fraction": float(dropout),
                "n_events": int(len(angle)),
                "embedding_dimension": int(clean_embedding.shape[1]),
                "median_angular_error_deg": float(np.median(angle)),
                "angular_error_ci95_low": angle_ci[0],
                "angular_error_ci95_high": angle_ci[1],
                # The interval that supports a degradation claim in this paired
                # design: median angular error minus clean, same resamples.
                "paired_delta_median_angular_error_deg": delta,
                "paired_delta_ci95_low": delta_low,
                "paired_delta_ci95_high": delta_high,
                "median_prediction_drift_deg": float(np.median(prediction_drift)),
                "median_standardized_embedding_displacement": float(
                    np.median(displacement)
                ),
                "embedding_displacement_ci95_low": displacement_ci[0],
                "embedding_displacement_ci95_high": displacement_ci[1],
                "knn10_retention": float(np.mean(retention)),
                # Fixed-graph approximation: events are resampled, the
                # neighbour graph is not recomputed per resample.
                "knn10_retention_ci95_low": retention_ci[0],
                "knn10_retention_ci95_high": retention_ci[1],
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "events_per_second": float(
                    len(angle) / float(result["elapsed_seconds"])
                ),
            }
        )

    frame = pd.DataFrame(summary_rows)
    frame.to_csv(args.output_dir / "pilot_metrics.csv", index=False)

    smoke = {
        "n_events": int(len(clean_prediction)),
        "balanced_track_cascade": True,
        "checkpoint_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "embedding_hook": "StandardModel.backbone output before DirectionReconstructionWithKappa",
        "embedding_dimension": int(clean_embedding.shape[1]),
        # Full disclosure of every deviation from the released pipeline
        # (audit C17): these are candidate causes of the open parity offset.
        "pipeline_substitutions": {
            "detector_feature_map_replaced": [
                "_sensor_pos_xy",
                "_sensor_pos_z",
                "_t",
                "_charge",
                "feature_map",
            ],
            "guessed_graphdefinition_defaults": {
                "_repeat_labels": False,
                "_add_static_features": True,
                "_perturbation_dict": None,
            },
            "module_path_aliases": [
                "graphnet.models.graphs.graph_definition",
                "graphnet.models.graphs.edges.edges",
                "graphnet.models.graphs.nodes.nodes",
            ],
        },
        "sampling": {
            "candidate_pool": "uniform random over full parquet (audit C12)",
            "selection_minimum_observed_modules": int(args.minimum_observed_modules),
            "dataset_minimum_modules_floor": int(args.dataset_minimum_modules),
            "dropout_sets_nested_across_severities": True,
        },
        "checkpoint_vs_released_prediction": {
            "median_direction_disagreement_deg": float(
                np.median(checkpoint_release_angle)
            ),
            "p95_direction_disagreement_deg": float(
                np.percentile(checkpoint_release_angle, 95)
            ),
            "max_direction_disagreement_deg": float(
                np.max(checkpoint_release_angle)
            ),
        },
        "clean_subset_metric": {
            "checkpoint_median_angular_error_deg": float(
                np.median(angle_between_deg(truth_vector, clean_prediction))
            ),
            "released_median_angular_error_deg": float(
                np.median(angle_between_deg(truth_vector, released_vector))
            ),
        },
        "event_ids": event_ids,
    }
    (args.output_dir / "smoke_test.json").write_text(
        json.dumps(smoke, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    x = frame["module_dropout_fraction"].to_numpy()
    axes[0].errorbar(
        x,
        frame["median_angular_error_deg"],
        yerr=np.vstack(
            (
                frame["median_angular_error_deg"] - frame["angular_error_ci95_low"],
                frame["angular_error_ci95_high"] - frame["median_angular_error_deg"],
            )
        ),
        marker="o",
        capsize=3,
    )
    axes[0].set_ylabel("Median angular error [deg]")
    axes[1].errorbar(
        x,
        frame["median_standardized_embedding_displacement"],
        yerr=np.vstack(
            (
                frame["median_standardized_embedding_displacement"]
                - frame["embedding_displacement_ci95_low"],
                frame["embedding_displacement_ci95_high"]
                - frame["median_standardized_embedding_displacement"],
            )
        ),
        marker="o",
        capsize=3,
        color="tab:orange",
    )
    axes[1].set_ylabel("Median latent displacement (standardized)")
    axes[2].errorbar(
        x,
        frame["knn10_retention"],
        yerr=np.vstack(
            (
                frame["knn10_retention"] - frame["knn10_retention_ci95_low"],
                frame["knn10_retention_ci95_high"] - frame["knn10_retention"],
            )
        ),
        marker="o",
        capsize=3,
        color="tab:green",
    )
    axes[2].set_ylabel("10-NN retention")
    axes[2].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.set_xlabel("Dropped observed-module fraction")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "module_dropout_pilot.png", dpi=180)
    fig.savefig(args.output_dir / "module_dropout_pilot.pdf")
    plt.close(fig)

    print(json.dumps(smoke, indent=2), flush=True)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
