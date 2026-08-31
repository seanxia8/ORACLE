# oracle_paired — ORACLE-Paired dataset production

Copyright (c) 2026 Dowling Wong. MIT (see repo `src/noise_module/LICENSE`
terms; package headers carry SPDX tags). Please cite via `CITATION.cff`.

Produces the paired multi-geometry point-cloud dataset described in
`docs/EXPERIMENT_DESIGN.md`: one seeded Prometheus injection replayed through
ORCA and ARCA, an in-project NuBench-style detector response (the stage where
every acquisition knob lives), declared N/S/U strata, content-matched clean
controls, and provenance-carrying Parquet export readable by GraphNeT.

| Module | Role |
|---|---|
| `config.py` | Prometheus run plans: one injecting config + one replaying config per further geometry (`inject=False`, same `injection_file`, different `geo_file`) — the pairing mechanism; U-family injection plans |
| `response.py` | NuBench §3.2 emulation: QE, ≥5 µs trigger window, uniform noise photons, TTS merge, 1 ns / 0.25 pe smearing, pulse-count cuts. The TTS window length is not stated in the NuBench paper — our default is a declared assumption |
| `interventions.py` | N1–N5: module loss (incl. by-string), hit thinning, timing jitter, per-string gain drift, noise-rate scale — each seeded and serialised into provenance |
| `strata.py` | S selectors (low-E, edge vertex, horizontal) and U builders (pile-up overlay, correlated module noise) |
| `matching.py` | Content-matched clean twins (audit P1.1): greedy 1:1 nearest neighbour in (n_pulses, charge, time spread), standardized, calipered |
| `export.py` | Pulse + truth Parquet with `injection_id` (the pairing key), stratum, exact intervention JSON, seeds |
| `toy.py` | Synthetic track/cascade photon events so everything downstream of Prometheus is testable without it |

Prometheus itself is never imported: the cluster runs it from the emitted
JSON configs, and this package consumes its photon output.

Tutorial: `notebooks/oracle_paired_dataset_tutorial.ipynb`.
Tests: `pytest src/oracle_paired/tests`.
