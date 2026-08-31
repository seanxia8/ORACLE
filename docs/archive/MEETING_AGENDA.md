# Paper 3 — meeting agenda

_28 August 2026. Supporting material: `docs/NOVELTY_REVIEW.md`,
`docs/OPEN_DECISIONS.md`, `docs/PAPER3_AUDIT.md`. Items 1–3 are the priority
decisions; items 4–8 can move to email if time runs short._

## 1. NuBench geometry pairing — decision required

Finding: the released NuBench datasets are independent per-geometry
productions and share no latent events (different event counts, energy ranges,
no event-ID correspondence). The Phase-1 paired-geometry pilot as previously
agreed is not executable from released artifacts.

Recovery path: Prometheus supports re-using one seeded injection across
detector geometries, and all six NuBench geometry files ship inside
Prometheus. A paired multi-geometry event set would be a citable dataset
contribution in its own right.

Options on the table (geometry validity falls under the physics-side
ownership):

- **(a) Re-simulate** a paired two-geometry set with Prometheus
  (~few hundred CPU-hours per pair, embarrassingly parallel; exactly
  reproducible in water, Poisson-level photon variation in ice).
- **(b) Interim released-data contrast:** Hexagon water vs Hexagon Ice LE,
  reported strictly as a distribution-level medium comparison, never a causal
  layout claim.
- **(c) Drop the geometry axis** and run NuBench purely as the
  cross-architecture replication of the acquisition study.

Details and sources: `OPEN_DECISIONS.md` D1.

## 2. Novelty positioning — agree the target claim wording

The prior-art scan (`NOVELTY_REVIEW.md`) found direct competitors for all
three headline claims: frozen-encoder shift-type identification exists
(Roschewitz et al., MICCAI 2025, arXiv:2411.07940); shift-type-determines-
which-layer-to-adapt is the headline of Surgical Fine-Tuning (ICLR 2023,
arXiv:2210.11466); harmful-vs-benign shift detection is an established line
(Podkopaev & Ramdas, ICLR 2022; Rabanser et al., NeurIPS 2019).

What survives, ranked: (1) consequence as a physics observable under a known
measurement covariance; (2) the conditional-on-alarm C4 endpoint with designed
alarm/consequence dissociations; (3) first monitoring result for learned
neutrino-telescope reconstruction; (4) the integrated pipeline.

Decision sought: adopt the target claim wording in `NOVELTY_REVIEW.md` §5
before further manuscript writing, so all subsequent text defends one agreed
sentence. This is protection, not shrinkage: every clause is chosen to be
undefeatable by the citations above.

Known internal weakness, flagged deliberately: the current N interventions
(module loss, hit thinning) manufacture low-multiplicity events — members of
the declared S strata — so an unmatched design partially reduces to
recognizing the corruption operator's fingerprint. Fix already in the revised
proposal: every N cell gets a content-matched S cell (same observed
multiplicity, charge, time spread).

## 3. A⊕B artifact class — proposed upgrade

The A⊕B contrast from the review comments (physics-supported modification vs
representation artifact with no science meaning, inside one representation)
stays the scientific core. Proposed refinement: instantiate the artifact class
as an **output-null perturbation** (latent displacement in the null space of
the local output Jacobian) rather than a random LoRA update.

Rationale: a random low-rank update is labelled non-physical; an output-null
perturbation is additionally *predictably inconsequential*, and its
norm-matched output-aligned counterpart is predictably consequential. The
contrast then carries a sign prediction — C4 becomes a designed test rather
than a correlation, and every displacement-magnitude-only baseline fails it by
construction. LoRA remains available as a secondary artifact generator.

## 4. First result: the dissociation experiment is unblocked

The output-null / output-aligned / norm-matched-random / clean design runs on
the controlled simulator today: no NuBench parity, no geometry gate, no
external data. It supplies the collaboration's first citable result while the
joint infrastructure is built, and it is the hedge against the main scooping
vector (a medical-imaging extension of arXiv:2411.07940 adding a performance
link — see `NOVELTY_REVIEW.md` §7).

## 5. TIDMAD consequence metric — sign-off requested

The TIDMAD authors state the denoising score "lacks direct relevance to
fundamental science" (it exists alongside a second, physics benchmark for that
reason), and the science files carry no injected ground truth. Proposal —
three-tier K:

- **K_dev** — upstream denoising score: fast iteration, disclosed as
  non-scientific.
- **K_rel** — coarse-level exclusion-limit comparison on a frozen 20-file
  science subset, identical file list for both arms (the TIDMAD C4 variable).
- **K_full** — the controlled simulator, where assumed and realized covariance
  are both known: the primary C4 testbed.

The 328-band truncation and the in-band file selection (files 0016–0019 carry
out-of-band injections) are declared wherever K is reported. Details:
`OPEN_DECISIONS.md` D2.

## 6. LArTPC ordering — Panda before SPINE

Proposed reversal of the earlier ordering on infrastructure cost: Panda is
pip-installable with four public HF checkpoints and an auto-downloading
dataset; SPINE requires a ~25 GB container with ROOT + LArCV2 +
MinkowskiEngine built from source, and its reconstructed/physics datasets are
still marked TODO. Panda benchmarks against SPINE, so a Panda result already
speaks to the SPINE question. SPINE remains the future domain-specific
extension. Details: `OPEN_DECISIONS.md` D3.

## 7. Repair validation — pre-registered caveat

Hase et al. (NeurIPS 2023, arXiv:2301.04213) showed causal-tracing
localization does not predict where model editing succeeds: adaptation gains
at a diagnosed stage are a weak validator of a localization claim. Proposal:
precede any LoRA repair claim with an **activation-patching** causal check
(substitute the clean representation at stage k into the perturbed forward
pass; measure consequence recovery). Zero training cost; pre-registering it
now is cheap, hearing it from a referee later is not.

## 8. Logistics

- Repository status: post-audit fixes are in; the tidmad test suite is green
  (26 passed, 1 skipped); a runnable day-one path exists
  (`docs/tidmad.md`, synthetic tests need no external data).
- Parity blocker plan: (i) re-run the released-metric rescoring with the fixed
  interaction/topology grouping — the earlier 0.055° discrepancy may dissolve;
  (ii) run the CPU-vs-GPU decisive test for the 0.944° re-inference offset.
  Agreement sought: no email to the NuBench authors until both are done.
- Cadence and ownership split; confirm fortnightly.
- Freeze list for this meeting: one representation endpoint per testbed, one
  downstream physics endpoint, and the consequence threshold κ_m.
