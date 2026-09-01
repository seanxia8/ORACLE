# ORACLE experiment design — the agreed solution

_31 August 2026. The canonical short version. **Buildable form:
`IMPLEMENTATION_PLAN.md` (work packages, interfaces, acceptance criteria,
gates); reviewer instructions: `REVIEW_PROMPTS.md`.** Full derivations,
sources and caveats: `docs/archive/DATASET_PRODUCTION_PLAN.md` (production detail),
`docs/archive/OPEN_DECISIONS.md` D1–D8, `docs/archive/NOVELTY_REVIEW.md`
(claim wording), `docs/archive/PAPER3_AUDIT.md`._

## The story in one paragraph

The paper's claim is that a monitoring alarm on a frozen reconstruction model
can be made to *rank scientific consequence* — and that whether it does is
governed by the metric it is computed in: an unweighted representation
displacement is the wrong-metric statistic, and the consequence-relevant
quantity is displacement in the Σ⁻¹-whitened metric pulled back through the
output Jacobian. No public dataset can test this, because none has a known
assumed-versus-realized covariance, and none is event-paired across the factors
being varied. So the study produces its own data — twice — and uses a public
real-data benchmark as the external check.

## The three tiers

**Tier 1 — ORACLE-Cov (mechanism).** Controlled waveforms from
`src/noise_module/`. The encoder is trained under an explicit assumed
covariance Σ̂ (the inverse-PSD-weighted objective makes the assumption
literal) and deployed under a realized Σ that `MultiChannelNoiseGenerator`
reports exactly. Strata: matched (Σ̂ = Σ); a κ(Σ̂⁻¹Σ) sweep; output-null
perturbations (large alarm, ≈zero consequence); norm-matched output-aligned
perturbations (same alarm, large consequence); random; clean. Everything
theoretical is proved here: the whitening result, the designed C4 dissociation
with its predicted sign, and the D6 power analysis that sizes every other arm.
Unblocked today.

**Tier 2 — ORACLE-Paired (realism).** A Prometheus production: one seeded
LeptonInjector run reused across ORCA and ARCA (both water → olympus →
exactly reproducible), our own implementation of NuBench §3.2 detector
response (which is where every acquisition knob lives), and per geometry:
`clean`; N1–N5 acquisition interventions (module loss, hit thinning, timing
jitter, gain drift, noise rate); S1–S5 matched rare-physics families; U1–U4
undeclared families for abstention (τ double-bang, through-going μ, pile-up,
correlated noise); plus content-matched clean twins for every N event
(nearest clean neighbour in observed multiplicity, total charge, time
spread — the audit P1.1 fix). `injection_id` is the pairing key no public set
has. No Σ̂ exists here — deliberately: this tier tests, out-of-sample, the
failure predictions made at Tier 1, and it is where the one monitor
computable *without* knowing Σ (the Jacobian-projected one) earns its
deployability claim, on a frozen model nobody in this project trained, with a
physics consequence variable (angular error).

**Tier 3 — TIDMAD (real data).** Already in hand: the same compact
transformer trained twice, MSE versus inverse-PSD-weighted — two different Σ̂
choices on identical real electronics noise. One paragraph of the paper; the
paragraph that blocks "it only works in your simulator". Consequence per D2's
three-tier K (K_rel as the confirmatory variable).

**The bridge that makes it one study, not three demos:** after Tier 1,
pre-register which monitor families will fail on which ORACLE-Paired
perturbation families — then run Tier 2 once. A mechanism that predicts
out-of-sample failures in a testbed it never saw is the referee-proof form of
every claim.

## Claims × tiers

One claim ladder, the proposal's (`IMPLEMENTATION_PLAN.md` WP0):

| Claim | Tier 1 ORACLE-Cov | Tier 2 ORACLE-Paired | Tier 3 TIDMAD |
|---|---|---|---|
| C1 detection @1% FAR | controlled families; power sizing (cov_D) | E1 on frozen DynEdge, matched cells | — |
| C2 attribution N vs matched-clean (primary), N vs S (secondary), **with abstention on U** | cov_C: full control; held-out U families | E2: matched cells; U1–U4; geometry-transfer of the layer profile (E5) | — |
| C3 cost (capture vs sample/sketch) | cov_C on the compact transformer | E3 on DynEdge | — |
| C4 alarm ranks consequence, conditional on alarm | **designed**: output-null vs output-aligned — K prediction registered, monitor ranking partly by construction; κ sweep (cov_A, cov_B) | **observational**: E4, conditional-on-alarm AUROC on angular error within severity strata | K_rel across the two Σ̂ trainings (T1) |
| C5 stage localization, by activation patching only | cov_E | replication on E2's consequential cells | — |

## Sequencing

1. Tier 1 now (includes the D6 null simulation — the only place it can run).
2. Tier 2 *production* in parallel — building the dataset is not gated by
   anything. The Tier 2 *confirmatory run* is gated by D5: because angular
   error is the consequence variable, the 0.944° re-inference offset is a
   confound on K until the CPU-vs-GPU test resolves it or it is shown stable
   across severity strata (`IMPLEMENTATION_PLAN.md` §7.2). Quoted NuBench
   numbers and any email to the authors wait for both halves of D5.
3. Everything else — Panda, LIGO, MicroBooNE, JUNO, FASER — is one sentence of
   future work each. Ten testbeds against an unwritten diagnostics layer is
   the project's largest schedule risk.

## Release posture

ORACLE-Paired goes to Hugging Face with generation config and seed manifest —
licence-clean (Prometheus LGPL-2.1, our seeds). **ORACLE-Cov does not go out
before the preprint**: publishing its dataset plus generation scripts is
publishing `src/noise_module/`, whose authorship position was only just
established. Release together with the arXiv submission.

## Models versus diagnostics — what exists, what we build

Two things the meeting note blurs. The **subject models** — the frozen
networks being monitored — all exist: the compact two-stage transformer in
`src/tidmad_transformer/` and `src/reconstruction_model/` for Tiers 1 and 3
(retrained per assumed covariance Σ̂; that retraining *is* the experiment),
and DynEdge, ParticleNeT, GRIT and DeepIce from GraphNeT for Tier 2. Nothing to
download or invent.

The **diagnostics are protocol code, not models**, and most of it is standard
machinery to be reused and cited. The three bold rows are small in code and are
the paper; build them first, on Tier 1, because the pre-registered predictions
for Tier 2 come from them.

| Component | Status | Tiers |
|---|---|---|
| Representation hooks | partly done — `pooled_representation` is in; DynEdge hook points settled (D4); finish the package | 1, 2, 3 |
| Standardized displacement, k-NN retention, principal angles | implement — standard linear algebra, ~200 lines; parts exist in `scripts/nubench/` | 1, 2, 3 |
| **Σ⁻¹-whitened displacement** | implement — trivial once Σ is known | **1 only** (needs Σ) |
| **Jacobian-projected displacement** | implement — autograd Jacobian of output w.r.t. representation, project; the one monitor computable *without* knowing Σ | 1, 2, 3 |
| **Output-null / output-aligned perturbation generators** | implement — SVD of the local output Jacobian; norm-matched | 1 (designed C4), 2 (replication) |
| Baselines: corrected univariate KS, RBF-MMD, classifier two-sample test, embedding mean/covariance distance, output and uncertainty tests | **reuse** — `alibi-detect` ships MMD, C2ST and KS drift detectors; cite, do not reimplement | all |
| Five-arm attribution classifier (input / output+uncertainty / final embedding / all-generic / full layerwise) | reuse — scikit-learn regularized logistic regression, identical splits | all |
| Conformal abstention, risk–coverage AUC | reuse (`MAPIE`) or ~50 lines of split conformal | 1, 2 |
| Content-matched clean cells (audit P1.1) | **done** — `oracle_paired.matching` | 2 |
| Consequence variable K | angular error (trivial); `K_rel` via TIDMAD's upstream Brazil-band code (D2); `K_full` on the controlled simulator | 2 / 3 / 1 |
| Activation-patching causal check (before any repair claim) | implement — substitute the clean stage-k representation into the perturbed forward pass; simple hook | 1, 2 |
| LoRA repair | reuse — `peft` | 1, 2 |
| D6 power analysis / null simulation | implement — scripts on ORACLE-Cov; the only place the null is exactly simulable | 1 (sizes every other arm) |

Rough size of the whole layer: 1.5–3k lines, of which about 80% is wiring
standard components. That is consistent with `docs/archive/NOVELTY_REVIEW.md`:
the contribution is the instantiation and the evaluation protocol, not new
monitors.

## Which datasets, finally

Of the candidates in the meeting note — TIDMAD, NuBench, MicroBooNE open data,
LIGO/Virgo/KAGRA — the paper uses **two, plus one that was not on the list**:

| | Role | Why |
|---|---|---|
| **Controlled simulator** (`src/noise_module/`, ORACLE-Cov) | Tier 1 — the headline | the only place Σ̂ and Σ are both known; unblocked today |
| **NuBench**, via our own Prometheus production (ORACLE-Paired), *not* the released tarballs | Tier 2 — realism | pairing, matched cells, frozen public model, physics consequence variable |
| **TIDMAD** | Tier 3 — real data | two Σ̂ trainings on identical real electronics noise; already in hand |
| MicroBooNE open data | future work, one sentence | LArTPC is already covered by the Panda-before-SPINE decision (D3) as the *later* domain extension; MicroBooNE adds entry cost without adding a claim |
| LIGO / Virgo / KAGRA | future work, one paragraph | not needed, but the strongest *optional* external check: the assumed covariance is literally public (the published ASD), glitches are Σ̂ ≠ Σ with real ground-truth labels (Gravity Spy), and MLGWSC-1 dataset 4 supplies real O3a noise plus public frozen detection models. If a second real-data arm is ever wanted, this is the one; budget two to three weeks of domain entry |

Ten candidate testbeds against a diagnostics layer that is still mostly
unwritten is the project's largest schedule risk. Three is the number.
