# Open decisions — researched and resolved (17 August 2026)

This document answers the open technical decisions from the revision plan and
the audit (`docs/PAPER3_AUDIT.md`) with primary-source research. Each entry
states the finding, the evidence, and the recommended decision. Items marked
**DECIDED (proposed)** need Junjie's sign-off at the next meeting; items marked
**FIXED** are implemented in this repository.

---

## D1. NuBench paired geometries — revision-plan open question #9

**Finding: NuBench does NOT provide common latent events across geometries, but
Prometheus supports generating them, and all six NuBench geometry files ship
inside Prometheus.**

- The seven NuBench datasets are independent per-geometry productions:
  different event counts (23.1M / 22.9M / 20.5M / 24.0M / 10.1M / 20.5M /
  8.6M), two different energy ranges, no event-ID correspondence, no seed
  record. The paired-geometry pilot as previously agreed cannot be built from
  the released artifacts.
- The Prometheus paper (arXiv:2304.14526, §5.4.2) names the use case
  explicitly: injection files can be re-used "for simulating the same events in
  different detectors in order to compare detector performance". Mechanism,
  verified in source: run LeptonInjector once, then loop geometries with
  `config.injection.lepton_injector.inject = False` and
  `paths.injection_file = <existing .h5>`, setting `config.detector.geo_file`
  per run. `config.run.random_state_seed` seeds injection, PROPOSAL, and the
  olympus photon stage.
- The six NuBench geometries are shipped in Prometheus `resources/geofiles/`
  (verified by string/module count): Triangle=`pone_triangle.geo`,
  Cluster=`gvd.geo`, Flower S=`orca.geo`, Flower L=`arca.geo`,
  Flower XL=`trident.geo`, Hexagon=`icecube.geo`.
- Caveats: (a) water/olympus runs are exactly reproducible; ice/PPC is
  seedable only to Poisson-level photon variation (documented in the
  Prometheus paper) — event pairing still holds, photon realisations differ.
  (b) The injection volume is auto-sized per detector; generate the injection
  against the largest geometry or set the volume manually so all detectors are
  contained, and put all compared geometries at a common site/depth.
  (c) NuBench's simplified detector response (§3.2: 30 cm OMs, 20% QE, photon
  merging, 1 ns/0.25 pe smearing, <4 or >1e6 pulse cuts) is fully specified in
  the paper but NOT released as code — regenerating NuBench-equivalent data
  means reimplementing that section.
- Cost: ~1–10 CPU-s/event/geometry at 1–100 TeV (repo profiling data);
  a 100k-event paired set over two geometries is a few hundred CPU-hours,
  embarrassingly parallel. Licenses: Prometheus LGPL-2.1; NuBench artifacts
  carry NO license — email the GraphNeT team before redistributing anything
  derived from them.

**DECIDED (proposed):** Phase B becomes a two-geometry Prometheus re-simulation
(true event pairing, our own seeds and provenance), starting with one pair at
a common depth. The Hexagon-water vs Hexagon-Ice-LE released contrast is
usable earlier but only as an explicitly distribution-level medium comparison,
never as a causal layout claim. A paired multi-geometry event set is itself a
releasable dataset contribution.

## D2. TIDMAD consequence metric K — audit P2.2

**Finding: the denoising score cannot be the "scientific consequence" for the
paper's C4 claim; a bounded relative exclusion-limit comparison is feasible.**

- The TIDMAD paper says Benchmark 1 (denoising score, log_5.27 Lambda) "lacks
  direct relevance to fundamental science" and introduces Benchmark 2 (the
  axion exclusion limit) for that reason. Science files carry no injected
  ground truth (CH1 only), so score-K and physics-K live on disjoint data.
- Benchmark 2 pipeline (verified from repo): denoise science files →
  `brazilband.py <files> --level coarse` (chunked FFT → averaged PSD →
  iMinuit/MINOS unbinned ML fits over ~1,000 mass points; `fine` = 11.1M) →
  optional AxionLimits plotting. All 208 files ≈ 560 GB; the dominant cost is
  our own inference over ~834 Gsamples.
- A SUBSET is statistically meaningful for a *relative* comparison: e.g. 20 of
  208 files (~54 GB) raises the noise floor ~sqrt(208/20) ≈ 3.2x, but both
  arms see identical raw data, so the limit ratio between denoisers is
  preserved to first order. Not a publishable limit; a valid arm-vs-arm K.

**DECIDED (proposed):** three-tier K for the waveform arm. K_dev = upstream
denoising score (fast iteration, disclosed as non-scientific); K_rel =
coarse-level Brazil-band limit on a frozen 20-file science subset, identical
file list for both arms (the C4 consequence variable on TIDMAD); K_full = the
controlled simulator, where assumed and realized covariance are both known —
the primary C4 testbed moves there. The 328-band truncation and the
`--in-band-only` file selection (files 0016–0019 carry out-of-band injections
per the P1 survey) must be declared wherever K is reported.

## D3. LArTPC ordering — Panda before SPINE (audit P2.3)

- Panda: pip-installable, four public HF checkpoints (`panda.load("base")` …),
  PILArNet-M (~168 GB) auto-download. SPINE: ~25 GB container, ROOT + LArCV2 +
  MinkowskiEngine from source, `pip install spine` explicitly not for
  training/full-chain inference, reconstructed HDF5 + physics datasets marked
  `TODO` in the 2026 workshop repo. Panda benchmarks against SPINE.

**DECIDED (proposed):** Panda is the scale-transfer testbed; SPINE stays a
future domain-specific extension. This reverses the earlier plan ordering and
reduces Junjie-side infrastructure load.

## D4. NuBench layerwise hooks — audit P1.3

**Finding (verified in GraphNeT source, main @ 2026-07):** DynEdge exposes four
conv layers as `model.backbone._conv_layers[0..3]` (`torch.nn.ModuleList`),
plus `_post_processing` and `_readout` `nn.Sequential` modules — all hookable
with `register_forward_hook`, no model edits. `StandardModel` stores the GNN as
`model.backbone` (the `gnn=` kwarg is deprecated). Global pooling is a method,
not a module: hook `_post_processing`/`_readout` around it.

**FIXED (design):** the layerwise arm on NuBench is now defined as six hook
points (4 conv + post-processing + readout), not one. On the TIDMAD arm,
`TidmadTransformer.pooled_representation` was added as the third stage anchor
(temporal, band, pooled).

## D5. Re-inference parity (0.944° blocker) — audit C17

Candidate causes, ranked (research findings):

1. Our own five-method detector substitution + two guessed GraphDefinition
   defaults (now disclosed in `smoke_test.json`);
2. Standardisation constants / node definition — read them from the released
   "Model Artifacts" config bundle instead of reconstructing by hand;
3. NuBench's train/test asymmetry: test partitions have noise pulses removed;
4. GraphNeT issue #880 (open, 2026-04-30): `predict_as_dataframe` iterates the
   dataloader twice and the passes can silently misalign — "downstream analysis
   just produces wrong physics";
5. torch_cluster kNN backend divergence (CPU nanoflann KD-tree vs CUDA
   brute-force with strict-`>` lowest-index tie-breaking; can return fewer
   edges — pyg#4906). Real at source level, amplified by DynEdge's per-layer
   latent-space re-graphing, but a *median* shift of 0.9° wants a coherent
   cause, which points at 1–4 first. No GraphNeT issue documents kNN
   nondeterminism — do not cite one.

**Decisive test (cheap):** identical re-inference twice, CPU vs GPU, same
checkpoint and preprocessing. CPU≠GPU ≈ 0.9° → backend; CPU=GPU≠parquet →
preprocessing/alignment/version skew.

**Also FIXED:** `nubench_reference_metrics.py` grouped Table 6 by the
`is_track` topology flag while labelling the output CC/NC (nu_e-CC cascades
pooled into "NC"). It now computes both groupings and gates each — re-run this
BEFORE emailing a NuBench author about the 0.055° V0-A discrepancy.

## D6. Power analysis for C1 — audit P1.4

The 80% detection / 5-point non-inferiority / 1% false-alert numbers are
currently unpowered. Before freezing: simulate the null on the controlled
waveform testbed to get the alarm-statistic variance per window count, then
size windows-per-cell for 80% power at the chosen margins. The pilot's
overlapping CIs at n=256 (15.28° [12.83–18.71] vs 17.95° [13.89–22.60]) show
n=256 is far too small for the low-severity cells. UNRESOLVED — needs the
simulator run.

## D7. Repository consolidation debt (audit C23, N2)

`models/current_compact.py` vs `model.py` (6-line fork), two `muon.py` copies,
`tidmad/registry.py` fork of `models/registry.py`, and registry entries
pointing at absent modules. UNRESOLVED — mechanical, do in one sweep after the
first TIDMAD result so imports don't churn mid-experiment.

## D8. What was NOT wrong (for the record)

- The member `pyproject.toml` files DO pin torch==2.5.1+cu124 and carry real
  dependency lists; audit item N7 arose from a partial file staging and is
  withdrawn as stated (the root workspace file legitimately has an empty
  dependency list).
- The upstream benchmark invocation is genuinely external and unmodified
  (subprocess to upstream `benchmark.py`, no fallback scorer) — the K function
  itself was never in question, only its scientific meaning (D2).

---

### Sources

Prometheus: arXiv:2304.14526; github.com/Harvard-Neutrino/prometheus (source:
`prometheus/prometheus.py`, `config_types.py`, `utils/config_mims.py`,
`resources/geofiles/`, `paper_plots/timing/`). NuBench: arXiv:2511.13111;
JINST 21 (2026) T05001, DOI 10.1088/1748-0221/21/05/T05001;
github.com/graphnet-team/NuBench. TIDMAD: arXiv:2406.04378 (NeurIPS 2025
D&B); github.com/jessicafry/TIDMAD (`brazilband.py`, README);
huggingface.co/datasets/jessicafry/TIDMAD. Panda: arXiv:2512.01324;
github.com/DeepLearnPhysics/Panda. SPINE: github.com/DeepLearnPhysics/spine;
spine.readthedocs.io. GraphNeT: github.com/graphnet-team/graphnet
(`src/graphnet/models/gnn/dynedge.py`, `standard_model.py`), issue #880;
pytorch_geometric issue #4906; rusty1s/pytorch_cluster (`knn_cpu.cpp`,
`knn_cuda.cu`).
