# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Content matching: the audit P1.1 fix.

Every N-intervened event gets a *clean twin* with the same observed content —
nearest clean neighbour in (n_pulses, total charge, time spread), matched
greedily without replacement under a standardized-distance caliper. A
positive attribution result on matched cells cannot be fingerprint
recognition of the corruption operator, because the matched clean events look
the same in exactly the marginals the corruption changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import EventPulses

FEATURE_NAMES = ("n_pulses", "total_charge_pe", "time_spread_ns")


def content_features(events: list[EventPulses]) -> np.ndarray:
    """(n_event, 3) observed-content features used for matching."""
    return np.array(
        [[e.n_pulses, e.total_charge_pe, e.time_spread_ns] for e in events],
        dtype=float,
    )


@dataclass(frozen=True)
class MatchResult:
    treated_index: np.ndarray   # indices into the treated list, matched only
    control_index: np.ndarray   # indices into the clean pool, same order
    distance: np.ndarray        # standardized distances of accepted pairs
    unmatched: np.ndarray       # treated indices with no control in caliper

    @property
    def n_matched(self) -> int:
        return self.treated_index.size


def match_clean_controls(
    treated: list[EventPulses] | np.ndarray,
    clean_pool: list[EventPulses] | np.ndarray,
    caliper: float = 1.0,
) -> MatchResult:
    """Greedy 1:1 nearest-neighbour matching without replacement.

    Features are standardized by the clean pool's mean and std. caliper
    is the maximum accepted standardized Euclidean distance. Treated events
    are processed in order of their best available distance, so scarce
    controls go to the pairs that need them most.
    """
    X_t = treated if isinstance(treated, np.ndarray) else content_features(treated)
    X_c = clean_pool if isinstance(clean_pool, np.ndarray) else content_features(clean_pool)
    if X_c.shape[0] == 0:
        raise ValueError("Empty clean pool.")
    mu = X_c.mean(axis=0)
    sd = X_c.std(axis=0)
    sd[sd == 0.0] = 1.0
    Zt = (X_t - mu) / sd
    Zc = (X_c - mu) / sd

    # full distance matrix; productions run this in chunks per stratum, and
    # cell sizes (1e4-ish) keep it comfortably in memory.
    d = np.linalg.norm(Zt[:, None, :] - Zc[None, :, :], axis=2)

    n_t = Zt.shape[0]
    order = np.argsort(d.min(axis=1), kind="stable")
    used = np.zeros(Zc.shape[0], dtype=bool)
    t_idx, c_idx, dist = [], [], []
    unmatched = []
    for i in order:
        row = np.where(used, np.inf, d[i])
        j = int(np.argmin(row))
        if np.isfinite(row[j]) and row[j] <= caliper:
            used[j] = True
            t_idx.append(int(i)); c_idx.append(j); dist.append(float(row[j]))
        else:
            unmatched.append(int(i))
    return MatchResult(
        np.asarray(t_idx, dtype=int),
        np.asarray(c_idx, dtype=int),
        np.asarray(dist, dtype=float),
        np.asarray(unmatched, dtype=int),
    )
