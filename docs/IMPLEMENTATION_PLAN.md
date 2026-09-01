# ORACLE implementation plan

_31 August 2026. The buildable version of `EXPERIMENT_DESIGN.md`. That document
says what the study is; this one says what gets written, in what order, with
what interfaces, and what "done" means for each piece. Review it with
`docs/REVIEW_PROMPTS.md` §A before any code is written, and §B after._

---

## 0. Ground rules — what "real, not toy" means here

Every rule below is enforced by something a reviewer can check, not by intent.

1. **Protocol before data.** Endpoints, thresholds (false-alert budget, κ_m,
   equivalence margins, power targets) and the family lists live in
   `docs/PREREGISTRATION.md` and are hash-frozen *before* any confirmatory run.
   Confirmatory runners refuse to start unless the config carries the frozen
   hash (WP7).
2. **Two modes, never mixed.** `mode: dev` may touch anything and produces
   nothing citable. `mode: confirmatory` consumes frozen configs, held-out
   seeds, held-out severities and held-out families, and writes results only
   into `experiments/<id>/results/confirmatory/`. The paper cites only the
   latter.
3. **Every metric has an analytic unit test and a positive control.** A
   monitor that cannot be shown to recover a known answer on synthetic data
   with a closed-form solution (WP2 tests) does not enter the frozen set. A
   metric is promoted from descriptive to confirmatory only if it responds
   monotonically to the positive-control severity ladder.
4. **Baselines get the same budget.** Identical windows, identical
   false-alert calibration on the same clean null, identical splits. A
   baseline that is not calibrated to 1 % FAR on held-out clean data is not a
   baseline.
5. **Resampling units are the real units.** Intervals resample *event groups*
   and *perturbation seeds* separately; never pulses, never windows that share
   an event. Enforced in `oracle_diag.report`.
6. **Seeds and provenance in every artifact.** Each result table carries the
   config hash, the git commit, the data manifest hash, and the seed list.
   Anything without them is dev output.
7. **Toy generators never touch confirmatory paths.** `oracle_paired.toy` and
   any `demo_*` helper are import-forbidden under `experiments/` — a test
   greps for it (WP7). The tutorial notebook is documentation, not evidence.
8. **Negative results are reportable.** Every comparison declares an
   equivalence margin in advance, so "no benefit" is a finding, not a failure.
9. **Costs are measured, not asserted.** C3 latency and storage numbers come
   from `oracle_diag.hooks.benchmark` on the real models with p50/p95 over
   ≥ 30 repetitions, wall-clock and bytes, on a named machine.
10. **Nothing is "validated" by adaptation gains.** Any localization claim is
    preceded by activation patching (WP6); LoRA repair is secondary.

---

## 1. Target repository layout

```
src/
  noise_module/          exists  — Tier-1 generator, Σ̂/Σ instrument
  oracle_paired/         exists  — Tier-2 production (needs prometheus_io, WP9)
  tidmad_transformer/    exists  — Tier-1/3 subject model + training
  reconstruction_model/  exists  — DELight transformer (Tier-1 subject variant)
  oracle_diag/           NEW     — the diagnostics layer (WP1–WP7)
    hooks/               representation capture, sketches, cost benchmark
    metrics/             displacement family, angles, kNN, layer profile
    perturb/             output-null / aligned / random generators
    baselines/           alibi-detect wrappers + embedding/output/uncertainty
    abstain/             split conformal, risk–coverage
    consequence/         K per tier, C4 matrix, conditional AUROC
    protocol/            config schema, freeze/hash, mode enforcement, registry
    report/              bootstrap CIs (correct units), RESULT.md writer
    tests/
experiments/
  cov_B_dissociation/    each: config.yaml, run.py, analysis.py,
  cov_A_kappa_sweep/           PREREGISTRATION.md (per-experiment section),
  cov_C_nsu_waveform/          results/{dev,confirmatory}/ (gitignored),
  cov_D_power/                 RESULT.md (committed, generated)
  paired_E1_detection/ … paired_E5_geometry/
  tidmad_T1_sigma_contrast/
docs/
  PREREGISTRATION.md     the frozen protocol (WP0); hash recorded in configs
  IMPLEMENTATION_PLAN.md this file
  REVIEW_PROMPTS.md      reviewer instructions, before and after
```

---

## 2. Shared interfaces (write these first; everything plugs into them)

```python
# oracle_diag/protocol/types.py
@dataclass(frozen=True)
class Family:            # a declared or undeclared perturbation family
    name: str            # "N1_module_loss", "S4_cascade", "U2_throughgoing", "null", "aligned", ...
    contract: str        # "N" | "S" | "U" | "artifact" | "clean"
    declared: bool       # False => abstention target, never in training/calibration
    severity: float | None

@dataclass
class Window:            # the monitoring unit; the resampling unit is event_group
    ids: np.ndarray      # event ids in the window
    event_group: int     # groups that share latent events (paired geometries share a group)
    family: Family
    seed: int

@dataclass
class RepresentationBatch:
    stage: str           # named hook point
    Z: np.ndarray        # (n, d) or sketch
    outputs: np.ndarray  # model outputs for the same events
    meta: dict

class Metric(Protocol):
    name: str
    def fit(self, clean: list[RepresentationBatch]) -> "Metric": ...      # clean reference only
    def score(self, batch: RepresentationBatch) -> float: ...             # alarm magnitude A
    def needs_covariance(self) -> bool: ...                               # True only for whitened

class Perturbation(Protocol):
    family: Family
    def apply(self, batch: RepresentationBatch, rng) -> RepresentationBatch: ...

class Consequence(Protocol):
    name: str
    def score(self, clean_outputs, perturbed_outputs, truth) -> float: ...  # K
```

Rules: `Metric.fit` sees clean data only; `score` never sees labels; every
`Perturbation` records its exact parameters into `RepresentationBatch.meta`.

---

## 3. Work packages

Each WP lists purpose, deliverables, interfaces, acceptance criteria (AC — all
must pass), and dependencies. Effort is in focused engineering days.

### WP0 — Protocol freeze (`docs/PREREGISTRATION.md`) · 2 d · no deps

Deliverables: the frozen protocol — for each claim the exact endpoint,
threshold and analysis; the declared N/S family lists per tier; the undeclared
U list; κ_m per tier with justification; false-alert budget 1 %; equivalence
margins (ΔF1 ±0.05; power 5 points); resampling units; the severity ladders;
the positive-control definitions; the *prediction table* skeleton to be filled
after Tier 1 (§5). A `freeze` command writes the SHA-256 of the file into
`protocol/frozen.json`.
AC: every claim C1–C5 maps to ≥ 1 experiment and ≥ 1 numeric endpoint; every
experiment maps back to a claim; a reviewer running `REVIEW_PROMPTS.md §A`
finds no unresolved blocker.

### WP1 — `oracle_diag.hooks` · 3 d · deps: none (models exist)

Deliverables: `capture(model, batch, stages) -> list[RepresentationBatch]` for
the compact transformer (stages: input embedding, each encoder block, pooled,
head) and for DynEdge (the six D4 points + pooled); `Sketch` (10 % event
sampling; fixed-size random-projection + moment sketch); `benchmark()` for C3.
AC: hooked tensors equal a manual forward to 1e-6; stage names are stable and
listed in one registry; sketch power loss measured (C3 endpoint: ≤ 5 points
F1/power loss vs full capture, tested on Tier-1 dev data); benchmark reports
p50/p95 latency and bytes/event on a named machine over ≥ 30 runs.

### WP2 — `oracle_diag.metrics` · 4 d · deps: WP1

Deliverables: `StandardizedDisplacement` (normalized by clean within-stratum
dispersion), `WhitenedDisplacement(Σ)` (Σ⁻¹ metric; `needs_covariance=True`),
`JacobianProjectedDisplacement` (∂output/∂Z at the stage, via autograd;
projects displacement onto the leading right-singular subspace), `PrincipalAngles`,
`KNNRetention`, `LayerProfile` (vector of per-stage scores).
AC (analytic): on Gaussian synthetic data with known Σ, whitened displacement
equals the Mahalanobis distance to 1e-9; on a *linear* model
`y = W Z`, Jacobian-projected displacement of a null-space perturbation is
exactly 0 and of an aligned one equals its norm times the singular value; kNN
retention is 1.0 on identity and → 1/k on random permutation. AC (control):
each metric is monotone across the positive-control ladder on Tier-1 dev data.

### WP3 — `oracle_diag.perturb` · 3 d · deps: WP1

Deliverables: `OutputNull(k)`, `OutputAligned(k)`, `NormMatchedRandom(k)` —
representation-level perturbations at stage k built from the local Jacobian
SVD, norm-matched to a declared displacement magnitude; input-level adapters
that wrap `noise_module` N families (waveforms) and `oracle_paired`
interventions (point clouds) behind the same `Perturbation` protocol.
AC: on the linear model, `OutputNull` changes outputs by < 1e-6 relative and
`OutputAligned` by the predicted amount; norms match to 1e-6; on the real
compact transformer, `OutputNull` changes the downstream consequence K by
< 5 % of what `OutputAligned` does at the same norm (this *is* the Cov-B
pre-check and is recorded, not asserted).

### WP4 — `oracle_diag.baselines` · 2 d · deps: WP1

Deliverables: wrappers with the `Metric` interface around `alibi-detect`
(RBF-MMD, classifier two-sample test, corrected univariate KS on inputs),
embedding mean/covariance distance, output-distribution and
uncertainty/entropy tests; a common `calibrate(clean_null, far=0.01)`.
AC: on held-out clean windows every baseline's realized FAR is within the
binomial 95 % CI of 1 %; the strong baseline arm ("all generic") is committed
to in `PREREGISTRATION.md` before any N/S result is looked at.

### WP5 — `oracle_diag.abstain` · 2 d · deps: WP2

Deliverables: split-conformal nonconformity over the declared-family
signature; unseen-family AUROC; selective risk and macro-F1 vs retained
coverage; risk–coverage AUC; macro-F1 at 80 % coverage.
AC: marginal coverage guarantee holds on exchangeable synthetic data (within
the finite-sample bound); U families are never present in calibration
(enforced by `Family.declared` in the loader).

### WP6 — `oracle_diag.consequence` · 3 d · deps: WP1, subject models

Deliverables: `AngularError` (Tier 2), `BrazilBandRel` wrapping TIDMAD's
upstream code on the frozen 20-file subset (Tier 3, D2), `WeightedResidual(Σ)`
— the likelihood-weighted reconstruction residual under the *true* Σ (Tier 1);
the C4 2×2 matrix with κ_m; conditional-on-alarm AUROC/AUPRC; within-severity
stratification; `ActivationPatch(stage_k)` — substitute the clean stage-k
representation into the perturbed forward pass and measure K recovery.
AC: K functions are pure and unit-tested against hand-computed values; the
conditional AUROC refuses to run pooled across severities (raises unless a
stratum is given); activation patching recovers ≥ 95 % of K on a perturbation
applied *at* stage k (self-consistency control).

### WP7 — `oracle_diag.protocol` + `report` · 3 d · deps: WP0

Deliverables: config schema (dataclasses, validated), `freeze` and hash check,
`mode` enforcement (confirmatory refuses without frozen hash and without
held-out seed lists), experiment registry, data-manifest hashing, bootstrap
over (event_group, perturbation_seed) with the correct nesting, `RESULT.md`
generator (endpoint, estimate, CI, n, config hash, commit), and the
**toy-import guard**: a test that fails if anything under `experiments/`
imports `oracle_paired.toy` or any `demo_*` symbol.
AC: a confirmatory run with a stale hash exits non-zero with a clear message;
a synthetic experiment with known truth gets correct CI coverage (≈ 95 %) under
the nested bootstrap; the toy guard fails on a deliberately planted import.

### WP8 — Tier-1 experiments (ORACLE-Cov) · 6 d · deps: WP1–WP7

Order matters: **B first** (unblocked, a day), then D (sizes everything),
then A and C.

- **cov_B_dissociation.** Subject: compact transformer trained under Σ̂
  (inverse-PSD objective) on `noise_module` data. Arms at matched displacement
  norm at the pooled stage and one mid stage: `null`, `aligned`, `random`,
  `clean`. Monitors: all WP2 + WP4. Endpoint: per-arm K, per-arm A for each
  monitor; the pre-registered sign prediction (Euclidean-magnitude monitors
  rank `null` ≥ `aligned`; Jacobian-projected and whitened rank `aligned` ≫
  `null`). Seeds: 20 perturbation seeds × 5 model seeds. Output: the first
  citable figure.
- **cov_D_power.** Simulate the null (clean vs clean windows) for every
  endpoint at candidate n; size windows-per-cell for 80 % power at 1 % FAR
  against the smallest severity in the ladder; write the numbers into
  `PREREGISTRATION.md` — this is what freezes WP0's counts and ORACLE-Paired's
  120k.
- **cov_A_kappa_sweep.** Train under Σ̂; deploy under Σ with
  κ(Σ̂⁻¹Σ) ∈ {1, 3, 10, 30, 100, 300, 1000}, constructed by
  `MultiChannelNoiseGenerator` with declared correlation structure (implied and
  realized covariance both saved). Endpoint: Spearman rank agreement between
  Euclidean displacement and K, and between whitened displacement and K, as a
  function of κ, with CIs. Prediction: the former degrades with κ, the latter
  does not (control, not monotonicity — do not over-commit).
- **cov_C_nsu_waveform.** N = `noise_module` families (PSD change, drift,
  lines, glitches, non-Gaussian tails) at the ladder; S = clean windows from
  declared support tails (amplitude, rise time, pile-up spacing); U = held-out
  families. Endpoints: C1 (power at 1 % FAR, delay in windows), C2 (macro-F1,
  ΔF1 vs the all-generic arm with CI and the three-way reading), C3 cost, C4
  observational. Every N cell has a content-matched S cell.

AC: each experiment has `PREREGISTRATION.md` section, `RESULT.md` with CIs,
`mode: confirmatory` artifacts, and passes the toy guard; cov_B's sign
prediction is stated in the pre-registration *before* the confirmatory run.

### WP9 — Tier-2 production and experiments (ORACLE-Paired) · 8 d + cluster · deps: WP1–WP7, D5 for released-number claims only

Deliverables: `oracle_paired.prometheus_io` (Prometheus parquet →
`EventPhotons`, with schema test against a real Prometheus output file);
cluster submission from `prometheus_config_pairs` (one injecting run, one
replaying run, ORCA and ARCA at common site); 1k-event pilot to re-profile
CPU cost *before* the full queue; the full production at the WP8-D counts;
strata + matching + export; GraphNeT dataset conversion; DynEdge inference
with WP1 hooks; experiments E1–E5 with the *pre-registered* Tier-1 predictions
(§5) tested once.
AC: pilot reproduces exactly on rerun (water/olympus); provenance columns
complete for 100 % of events; matched cells achieve ≥ 90 % match rate inside
the caliper (else overproduction factor is raised and recorded); E5 reports,
per (monitor, family), the pre-registered prediction and the outcome, with no
post-hoc additions to the prediction table.

### WP10 — Tier-3 (TIDMAD) · 3 d · deps: WP1–WP6

Deliverables: the two Σ̂ trainings (MSE, inverse-PSD) with identical seeds and
data; monitors on both; `K_dev` disclosed as non-scientific; `K_rel` on the
frozen 20-file subset, identical list both arms; 328-band truncation and
in-band file selection declared.
AC: the two arms differ *only* in the loss; `K_rel` file list is hash-frozen;
the monitor contrast between arms is reported with CIs.

### WP11 — Paper assembly · ongoing

Each figure/table in `latex/` names the experiment id and `RESULT.md` it comes
from; a script regenerates every figure from `results/confirmatory/`; the
proposal's E-matrix is replaced by the realized one.

---

## 4. Order and gates

| Milestone | Contents | Gate to pass |
|---|---|---|
| **M0** | WP0 draft, WP7 skeleton | §A review: no blockers |
| **M1** | WP1, WP2, WP3 with analytic tests | all AC green; positive-control ladders monotone |
| **M2** | cov_B (first citable result), WP4–WP6 | sign prediction pre-registered, then confirmed or refuted — either is reportable |
| **M3** | cov_D → freeze counts; cov_A, cov_C | `PREREGISTRATION.md` hash frozen; prediction table (§5) filled |
| **M4** | WP9 pilot + production, WP10 | pilot exact-replay check; D5 both halves before any released-NuBench number |
| **M5** | E1–E5 run **once** in confirmatory mode | §B review: no blockers |
| **M6** | WP11 | every figure traces to a confirmatory `RESULT.md` |

Nothing in M4–M5 starts before M3's hash is frozen. That ordering is the
scientific content of the design, not a project-management preference.

---

## 5. The pre-registration bridge (filled at M3, tested at M5)

For each monitor in the frozen set and each ORACLE-Paired family, one row,
written *before* any Tier-2 data is looked at:

| Monitor | Family | Predicted at 1 % FAR | Basis (Tier-1 result) |
|---|---|---|---|
| Euclidean displacement (pooled) | N1 module loss 25 % | fires | cov_C, N ladder |
| Euclidean displacement (pooled) | S1 low-energy (matched to N1) | fires — **false attribution** | cov_C: content confound |
| Jacobian-projected (pooled) | S1 low-energy | quiet | cov_B: output-null analogue |
| … | … | … | … |

The E5 result is the confusion between this table and the outcome, with no
rows added afterwards. A monitor that predicts out-of-sample failures in a
testbed it never saw is the referee-proof form of every claim in the paper.

---

## 6. Definition of done for the whole study

- Every claim C1–C5 has a confirmatory `RESULT.md` with CI, config hash and
  commit; every figure traces to one.
- The toy guard, the freeze check and the mode enforcement all pass in CI.
- §B review reports no blocker.
- The pre-registration table has a recorded outcome for every row.
- Negative or equivocal results are in the paper with their equivalence
  margins, not in a drawer.
