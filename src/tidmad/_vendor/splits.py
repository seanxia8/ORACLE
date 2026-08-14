"""Window splits and the leakage audit for the TIDMAD stage.

Two properties matter and neither is automatic:

1. **Chronological, not random.** Consecutive windows of a continuous stream are
   strongly dependent. A random split would put near-duplicate windows on both
   sides of the fit/evaluation boundary and inflate every effect. The CRESST
   stage split chronologically within each channel for the same reason; this is
   the streaming analogue.
2. **A gap between fit and evaluation.** Even a chronological split leaks if the
   two sides are adjacent, because noise correlation time can exceed the window
   length. A predeclared guard interval removes that.

The audit function returns evidence, not a verdict: index disjointness, the
realized time gap, and window-hash overlap, so the run record can carry proof
rather than an assurance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = ["WindowSplit", "chronological_window_split", "leakage_audit", "window_hashes"]


@dataclass(frozen=True)
class WindowSplit:
    """Indices of a chronological fit/evaluation split with a guard gap."""

    fit_indices: np.ndarray
    evaluation_indices: np.ndarray
    guard_windows: int
    fit_fraction: float

    def as_dict(self) -> dict:
        return {
            "n_fit": int(self.fit_indices.size),
            "n_evaluation": int(self.evaluation_indices.size),
            "guard_windows": int(self.guard_windows),
            "fit_fraction": float(self.fit_fraction),
            "fit_index_range": [int(self.fit_indices.min()), int(self.fit_indices.max())]
            if self.fit_indices.size
            else [],
            "evaluation_index_range": [
                int(self.evaluation_indices.min()),
                int(self.evaluation_indices.max()),
            ]
            if self.evaluation_indices.size
            else [],
        }


def chronological_window_split(
    n_windows: int,
    *,
    fit_fraction: float = 0.7,
    guard_windows: int = 0,
) -> WindowSplit:
    """Split window indices chronologically with a guard gap between the sides."""
    n = int(n_windows)
    if n <= 0:
        raise ValueError("n_windows must be positive")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie strictly between 0 and 1")
    guard = int(guard_windows)
    if guard < 0:
        raise ValueError("guard_windows must be non-negative")

    n_fit = int(round(fit_fraction * n))
    n_fit = min(max(n_fit, 1), n - 1)
    eval_start = min(n_fit + guard, n)
    if eval_start >= n:
        raise ValueError(
            f"guard_windows={guard} leaves no evaluation windows "
            f"(n_windows={n}, n_fit={n_fit})"
        )

    return WindowSplit(
        fit_indices=np.arange(n_fit, dtype=np.int64),
        evaluation_indices=np.arange(eval_start, n, dtype=np.int64),
        guard_windows=guard,
        fit_fraction=float(fit_fraction),
    )


def window_hashes(windows: np.ndarray) -> list[str]:
    """SHA-256 of each window's bytes, for duplicate detection."""
    X = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    return [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in X]


def leakage_audit(
    split: WindowSplit,
    *,
    fit_windows: np.ndarray | None = None,
    evaluation_windows: np.ndarray | None = None,
    window_samples: int | None = None,
    stride: int | None = None,
    sampling_frequency: float | None = None,
) -> dict:
    """Return machine-readable evidence that fit and evaluation are disjoint."""
    fit_set = set(int(i) for i in split.fit_indices)
    eval_set = set(int(i) for i in split.evaluation_indices)
    index_overlap = sorted(fit_set & eval_set)

    audit: dict = {
        "index_overlap_count": len(index_overlap),
        "index_overlap_examples": index_overlap[:10],
        "guard_windows": int(split.guard_windows),
    }

    if stride is not None and sampling_frequency is not None and split.evaluation_indices.size:
        gap_windows = int(split.evaluation_indices.min() - split.fit_indices.max() - 1)
        audit["guard_gap_windows_realized"] = gap_windows
        audit["guard_gap_seconds"] = float(gap_windows * int(stride) / float(sampling_frequency))

    if window_samples is not None and stride is not None:
        audit["windows_overlap_in_samples"] = bool(int(stride) < int(window_samples))

    if fit_windows is not None and evaluation_windows is not None:
        fit_hashes = set(window_hashes(fit_windows))
        eval_hashes = window_hashes(evaluation_windows)
        collisions = [h for h in eval_hashes if h in fit_hashes]
        audit["hash_overlap_count"] = len(collisions)
        audit["hash_overlap_examples"] = collisions[:5]

    return audit
