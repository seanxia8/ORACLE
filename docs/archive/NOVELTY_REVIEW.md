# Novelty review — Paper 3

_Prepared 28 August 2026 for the collaboration meeting. Every prior-art claim
below was verified against the cited arXiv record or source repository during
the 17 August audit (`docs/PAPER3_AUDIT.md` §3); nothing here is from memory._

## 1. The claims under review

As stated in the current proposal, the paper's novelty rests on three claims:

- **Claim A (attribution).** A frozen model's intermediate representations can
  distinguish corrupted acquisition (N) from clean but under-supported physics
  (S), with conformal abstention on undeclared failure families.
- **Claim B (consequence).** Monitor alarm magnitude predicts downstream
  scientific consequence — the 2×2 alarm/consequence matrix, with a
  conditional-on-alarm AUROC as the primary endpoint.
- **Claim C (localization).** The layerwise profile identifies *where* a
  consequential failure emerges, later validated by adapting the diagnosed
  stage versus a deliberately wrong stage.

## 2. Verdict in one paragraph

None of the three claims is novel as a *method*. Each has a directly competing
published paper, and two of the three are near re-derivations. What is
genuinely open is the *instantiation* and the *evaluation protocol*: no prior
work monitors a learned particle-reconstruction model's internals, none uses a
physics observable under a known measurement covariance as the consequence
variable, and no prior paper reports a conditional-on-alarm consequence AUROC
or a designed alarm/consequence dissociation. The paper survives review if and
only if it is repositioned around those three things and cites the competing
work up front. The revised proposal (17 August) already makes this move; this
note is the evidence base for it.

## 3. Claim-by-claim

### Claim A — shift-type attribution from frozen representations

**Closest prior art.**

- **Roschewitz, Mehta, Jones & Glocker, MICCAI 2025**
  ([arXiv:2411.07940](https://arxiv.org/abs/2411.07940)) — *the* direct
  competitor. Unsupervised, frozen-encoder framework that explicitly classifies
  **which type** of shift occurred (prevalence vs covariate vs mixed) via BBSD
  on task-model outputs + MMD on frozen self-supervised features, with a
  staged decision rule. Same goal, same architectural commitment (frozen
  encoder, no retraining).
- **Castro, Walker & Glocker, Nature Communications 2020**
  ([arXiv:1912.08142](https://arxiv.org/abs/1912.08142)) — establishes the
  taxonomy our N/S contract restates in detector language: *acquisition shift*
  (measurement-process change) vs *manifestation/population shift* (real but
  differently-distributed cases), with a causal-graph argument for why they
  demand different responses. Must be cited or the terminology reads as
  borrowed without attribution.
- **Yang, Zhou & Liu, IJCV 2023** ([arXiv:2204.05306](https://arxiv.org/abs/2204.05306))
  — full-spectrum OOD: covariate-shift OOD (tolerate) vs semantic-shift OOD
  (reject), motivated by the same intuition.
- Slice discovery (**Domino**, [arXiv:2203.14960](https://arxiv.org/abs/2203.14960);
  **Spotlight**, [arXiv:2107.00758](https://arxiv.org/abs/2107.00758)) already
  finds coherent rare underperforming subgroups in frozen representation
  space — the S branch alone is largely solved.
- Conformal abstention is applied machinery, not a contribution: Bates et al.,
  Ann. Statist. 2023 ([arXiv:2104.08279](https://arxiv.org/abs/2104.08279));
  iDECODe, AAAI 2022 ([arXiv:2201.02331](https://arxiv.org/abs/2201.02331)).

**What Roschewitz et al. do NOT do:** layerwise evidence (final-layer only),
abstention on unseen families, and — decisively — any link from the identified
shift to performance degradation. Those gaps are exactly our room.

**Internal design caveat (audit P1.1):** our own N intervention (module
loss/hit thinning) *manufactures* low-multiplicity events, i.e. members of the
declared S strata. Without content-matched S cells, a positive Claim A result
is fingerprint recognition of our corruption operator, and a referee who sees
it will say so. The matched-cell design is now in the proposal; it must survive
into the frozen protocol.

**Verdict: not novel as a method.** Claim the physics instantiation, the
layerwise extension, and the N/S contract as a *closed, declared* family — not
shift-type identification as such.

### Claim B — alarm magnitude predicts scientific consequence

**Closest prior art.** This is the most crowded lane; "harmful shift
detection" is a named research line.

- **Podkopaev & Ramdas, ICLR 2022** ([arXiv:2110.06177](https://arxiv.org/abs/2110.06177))
  — "detect harmful shifts while ignoring benign ones" is the paper's thesis,
  four years before ours.
- **Amoukou et al., NeurIPS 2024** ([arXiv:2412.12910](https://arxiv.org/abs/2412.12910))
  — the label-free sequential version; closes the "no labels in deployment"
  escape.
- **Rabanser, Günnemann & Lipton, NeurIPS 2019**
  ([arXiv:1810.11953](https://arxiv.org/abs/1810.11953)) — *Failing Loudly*
  already pairs detection power with whether accuracy actually degraded
  (shift-malignancy analysis). The first paper any referee will name.
- **Zhang et al., ICML 2023** ([arXiv:2210.10769](https://arxiv.org/abs/2210.10769))
  — Shapley attribution of a *performance change* to specific shift
  mechanisms; already couples attribution to consequence, threatening Claims A
  and B jointly.
- **ATC** ([arXiv:2201.04234](https://arxiv.org/abs/2201.04234)) and **DoC**
  ([arXiv:2107.03315](https://arxiv.org/abs/2107.03315)) — confidence-derived
  scalars already predict OOD accuracy; "alarm magnitude predicts consequence"
  is largely answered affirmatively for classification.
- **Bayram, Ahmed & Kassler, KBS 2022** — an entire survey of
  performance-aware drift detectors. "Nobody asks whether the drift matters"
  is unsayable.

**What none of them have:** (i) a consequence variable that is a *physics
observable under a known measurement covariance* rather than classification
accuracy; (ii) the **conditional-on-alarm** AUROC — among fired alarms, does
magnitude rank consequence? — and the explicit 2×2 contingency reporting the
low-alarm/high-consequence blind-spot rate; (iii) any *designed dissociation*
between alarm and consequence. Our output-null / output-aligned pair (equal
latent displacement, opposite consequence by construction) makes C4 a test
with a predicted sign instead of a mechanically positive correlation — I found
no prior instance of this design anywhere.

**Verdict: the question is old; our measurement of it is new.** Present C4 as
an evaluation-protocol refinement of harmful-shift detection, in a domain
where "harm" has a likelihood-geometric definition (Σ⁻¹-weighted residual)
instead of a held-out label.

### Claim C — layerwise localization, validated by targeted adaptation

**Closest prior art.**

- **Lee et al., ICLR 2023, "Surgical Fine-Tuning"**
  ([arXiv:2210.11466](https://arxiv.org/abs/2210.11466)) — the most dangerous
  single paper for us. Headline result: *the type of distribution shift
  determines which layer block is best to adapt* (input corruptions → early
  layers; feature-level → middle; label shift → last), across seven datasets,
  with an automatic gradient-norm criterion and a theoretical account. Our
  diagnosed-stage-vs-wrong-stage LoRA experiment is structurally this, with
  LoRA substituted for block fine-tuning and a detector for CIFAR-C.
- **Hase et al., NeurIPS 2023** ([arXiv:2301.04213](https://arxiv.org/abs/2301.04213))
  — a methodological landmine: causal-tracing localization does *not* predict
  where model editing succeeds. Published evidence that "adaptation at the
  diagnosed stage worked better" is a weak validator of a localization claim —
  a positive result does not confirm the diagnosis and a negative one does not
  falsify it.
- **Lambert et al., MICCAI-UNSURE 2023**
  ([arXiv:2307.15647](https://arxiv.org/abs/2307.15647)) — single-layer OOD
  detectors behave inconsistently *depending on anomaly type*; that
  layer-of-best-detection depends on failure type is already published.
- The layerwise machinery itself is ~8 years old (Mahalanobis on all
  intermediate layers, Lee et al. NeurIPS 2018).

**What remains:** using the layerwise profile as a *diagnostic output* (where
did it emerge) rather than a detection ingredient, on stages with physical
meaning (within-waveform time structure vs cross-band structure; per-DynEdge-
conv graph refinement). And a defensible validation route: **activation
patching** — substitute the clean representation at stage k into the perturbed
forward pass and measure consequence recovery — is a direct causal test that
pre-empts the Hase critique and costs no training. LoRA arms then become a
secondary practicality result, not the load-bearing validation.

**Verdict: not novel as a finding; salvageable as mechanism-with-physics.**
Never use the word "novel" near this claim.

### Domain lane (the part with no competition)

- CMS ML data-quality monitoring ([arXiv:2501.13789](https://arxiv.org/abs/2501.13789),
  ECAL autoencoders [arXiv:2309.10157](https://arxiv.org/abs/2309.10157),
  [arXiv:2407.20278](https://arxiv.org/abs/2407.20278)) detects acquisition
  faults from *monitoring histograms*, not from a reconstruction model's
  internals, and never touches N-vs-S or consequence.
- No published work monitors learned neutrino-telescope reconstruction at all
  (verified against the GraphNeT/NuBench ecosystem, August 2026).
- Panda ([arXiv:2512.01324](https://arxiv.org/abs/2512.01324)) establishes
  reusable LArTPC representations; it evaluates transfer quality, not
  post-training failure attribution — close prior work and a later testbed,
  not a subsumption.

## 4. What survives — the defensible contributions, ranked

1. **Consequence as a physics observable with known covariance.** The
   controlled simulator supplies both the assumed Σ̂ and the realized Σ, so
   "harm" is defined in the instrument's own likelihood geometry — no prior
   harmful-shift work has this, because no ML benchmark has a ground-truth
   noise model. This is the contribution only this group can write (it is also
   where Paper 1's whitening framework re-enters).
2. **The dissociation-designed C4 endpoint.** Conditional-on-alarm AUROC +
   2×2 blind-spot reporting + output-null/output-aligned arms with a predicted
   sign. Novel as an evaluation protocol; small but real, and it survives even
   if every detection result is null.
3. **First monitoring result for learned neutrino-telescope reconstruction**
   (and learned detector reconstruction generally, at the representation
   level). Domain-transfer novelty; referees discount it alone, which is why
   it rides on 1–2.
4. **The integrated pipeline** (declared N/S/E contract → abstain → localize →
   causally check). Integration novelty only; claim it last, if at all.

## 5. Recommended claim wording

**Do not write:** "We propose a novel framework for attributing and localizing
distribution shifts in learned representations."

**Write:** "We port harmful-shift detection and shift-type identification to
learned physics reconstruction, where — unlike any prior setting — the
consequence variable is a physics observable under a known measurement
covariance. We test, for the first time, whether alarm magnitude ranks
scientific consequence *conditional on the monitor firing*, using intervention
families designed to dissociate alarm from consequence; and we treat layerwise
localization as a hypothesis to be causally checked by activation patching,
not validated by adaptation gains."

Every clause of that sentence is defensible against the papers in §3; nothing
in it can be taken away by a citation we missed.

## 6. Prepared answers for the three inevitable objections

- **"Roschewitz et al. already did shift-type identification."** Yes — final
  layer only, no abstention on unseen families, and no consequence link; we
  cite them as the method baseline and differ on all three, in a domain where
  the shift types have a measurement-model definition rather than a clinical
  one.
- **"Why not just use DQM?"** Detector DQM watches histograms of raw
  quantities; it cannot see a failure that is invisible marginally but moves
  the model along task-relevant representation directions (the output-aligned
  arm demonstrates exactly this class), and it says nothing about which
  physics result degrades.
- **"Why not train the robustness in?"** (systematics-aware training,
  [arXiv:2105.08742](https://arxiv.org/abs/2105.08742)): orthogonal —
  mitigation at training time does not answer whether a *deployed, frozen*
  model's current data violates its acquisition contract, and frozen
  checkpoints are the operational reality for released benchmarks and
  collaboration-approved models.

## 7. Scooping risk assessment

Highest-risk vector: a medical-imaging group (the Glocker lab lineage)
extending arXiv:2411.07940 with a performance link — that would cover A+B in
their domain. Our moat is item 1 of §4: they have no measurement covariance
and no physics observable. Second vector: a HEP group wiring existing DQM
autoencoders to reconstruction models. Both argue for publishing the
simulator-based dissociation result (unblocked now, no external dependencies)
before the full two-testbed study.
