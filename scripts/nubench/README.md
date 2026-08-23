# NuBench feasibility scripts

Migrated from the external `paper3/scripts/` folder on 17 August 2026, with the
audit fixes applied (see `docs/PAPER3_AUDIT.md` items C11–C17 and the amended
`results/nubench_hexagon_ice_le_dynedge/RESULT.md`).

Both scripts target the official GraphNeT 1.8.0 CPU environment and need the
external NuBench artifacts (SQLite database, released prediction parquet,
released checkpoint) — none of which are committed here. Because GraphNeT's
loader deserializes the released model, only use a checkpoint from a trusted
source.

## `nubench_reference_metrics.py`

Recomputes Table 6 metrics from the released predictions. Post-audit it
computes BOTH the `is_track` (topology) and `interaction` (CC/NC) groupings
and evaluates the 0.01-unit identity gate against each — the earlier version's
grouping mislabel is a candidate explanation for the previously unexplained
0.055-degree discrepancy.

```bash
python scripts/nubench/nubench_reference_metrics.py \
  --predictions /path/to/DynEdge_predictions.parquet \
  --output /path/to/reference_metrics.json
```

## `nubench_smoke_pilot.py`

Paired module-dropout development pilot on the Hexagon Ice LE DynEdge
checkpoint. Post-audit: uniform-random candidate sampling over the full
parquet, nested dropout sets across severities, a hard event-order alignment
assert between clean and perturbed runs, shared-resample (paired) bootstrap
CIs including the paired-difference interval, a CI on 10-NN retention, and
full disclosure of every pipeline substitution in `smoke_test.json`.

```bash
python scripts/nubench/nubench_smoke_pilot.py \
  --database /path/to/hexagon_ice_le.db \
  --checkpoint /path/to/DynEdge_checkpoint.pth \
  --released-predictions /path/to/DynEdge_predictions.parquet \
  --output-dir /path/to/pilot-output
```

The open V0-B parity blocker (0.944-degree median re-inference offset) and its
decisive CPU-vs-GPU test are documented in `docs/OPEN_DECISIONS.md` (D5).
