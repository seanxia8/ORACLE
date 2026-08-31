# ORACLE experiment design — the agreed solution

_31 August 2026. The canonical short version. Full derivations, sources and
caveats: `docs/archive/DATASET_PRODUCTION_PLAN.md` (production detail),
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

| Claim | Tier 1 ORACLE-Cov | Tier 2 ORACLE-Paired | Tier 3 TIDMAD |
|---|---|---|---|
| C1 detection @1% FAB | controlled families; power sizing (D6) | frozen DynEdge, matched cells | smoke check |
| C2 N/S attribution + localization | full control | matched cells; geometry-transfer of the layer profile | — |
| C3 abstention | held-out families | U1–U4 | — |
| C4 alarm ranks consequence | **designed**: output-null vs output-aligned, sign predicted; κ sweep | **observational**: conditional-on-alarm AUROC on angular error | K_rel across the two Σ̂ trainings |
| C5 repair | activation patching first, then LoRA | replication | — |

## Sequencing

1. Tier 1 now (includes the D6 null simulation — the only place it can run).
2. Tier 2 production in parallel — it is *not* gated by the 0.944° parity
   blocker, which concerns released NuBench checkpoints on released NuBench
   data; only claims quoting NuBench's own numbers are gated (D5, both
   halves, before any such claim or any email to the NuBench authors).
3. Everything else — Panda, LIGO, MicroBooNE, JUNO, FASER — is one sentence of
   future work each. Ten testbeds against an unwritten diagnostics layer is
   the project's largest schedule risk.

## Release posture

ORACLE-Paired goes to Hugging Face with generation config and seed manifest —
licence-clean (Prometheus LGPL-2.1, our seeds). **ORACLE-Cov does not go out
before the preprint**: publishing its dataset plus generation scripts is
publishing `src/noise_module/`, whose authorship position was only just
established. Release together with the arXiv submission.
