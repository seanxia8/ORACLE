# Review prompt — the 3 September 2026 theme adjustment of a two-paper programme

You are an independent reviewer. You have read access to three repositories on this machine
and to four external papers. You will not edit any file. Your deliverable is one report,
written as `docs/reviews/2026-09-0X_theme_review.md` for the experiment repository (return the
text; the author commits it). Two pages of findings first, verdicts last. Quote before you judge;
every finding cites a file and a section or line.

## 1. The material

Repositories (all on branch `main` unless stated):

- **Paper 1** — `~/Documents/arXiv-v1-polished-extended-preprint`. *Noise Covariance Defines
  Reconstruction Geometry: Optimal Filtering, Weighted Subspaces, and Tied Linear Autoencoders.*
  Build: `latexmk -pdf main.tex` (68 pages, zero errors at commit `be35292`). Sources are the
  `0*_*.tex` and `app*.tex` files; `07b_separable_evidence.tex` is drafted but **not** input into
  the build (see the rule in §5).
- **Paper 3 / ORACLE** — `~/Documents/ORACLE`, branch `dev`. `latex/paper3_proposal.tex`, a
  six-page collaboration concept note; build twice with `pdflatex` from `latex/` (commit
  `0760c37`).
- **Experiment repository** — `~/Documents/noise-weighted-subspace-reconstruction`. All code and
  results for both papers: `experiments/paper1/`, `experiments/oracle/`, `results/paper1/`,
  `results/oracle/`, plans in `docs/EXPERIMENT_PLAN_PAPER1.md`, `docs/EXPERIMENT_PLAN_PAPER3.md`,
  `docs/TODO.md` (commits `78f6557`, `a6a4173`).

External papers to read in full before you start:

- **Sam Young, César Jesús-Valls, Kazuhiro Terao — *Panda Diplomacy: Foundation Model
  Pre-training across Particle Imaging Detectors for High Energy and Nuclear Physics*,
  arXiv:2609.00611.** One point-cloud self-distillation recipe pre-trained independently on
  LArTPC, collider-TPC and water-Cherenkov data; frozen representations adapted with ~10³ labels
  per detector match or approach specialised baselines (70× to 1000× label reduction); linear
  probes find directions for particle causality and track curvature. Its authors list transfer
  across detectors and from simulation to real data as untested. A local copy is at
  `noise-weighted-subspace-reconstruction/paper_rps2026/sources/panda_diplomacy_2609.00611v1.pdf`.
- **Shah Rukh Qasim, Jan Kieseler, Yutaro Iiyama, Maurizio Pierini — *Learning representations
  of irregular particle-detector geometry with distance-weighted graph networks*
  (GravNet/GarNet), arXiv:1902.07987.** Every sensor is mapped to a learned coordinate space S
  and messages are weighted by a potential exp(−d²) in that space; all models ≈100k parameters;
  metrics reported inclusively and on the overlapping-shower subset; a resource section
  (inference time, memory vs input size).
- **Jan Kieseler — *Object condensation: one-stage grid-free multi-object reconstruction in
  physics detectors, graph and image data*, arXiv:2002.03605.** A potential-based loss whose
  minimiser places one representative per object; each vertex carries a learned weight β for
  whether it can represent its object; hyperparameters q_min, s_B, s_C and inference thresholds
  t_β, t_d in one table; proof-of-concept on images, then particle flow against a baseline PF
  with efficiency, fake rate, response and jet resolution; extrapolation beyond the training
  multiplicity.

The author's own reading of the two Kieseler papers, and of what to copy from them
methodologically, is in `noise-weighted-subspace-reconstruction/docs/background/KIESELER_METHOD_NOTES_2026-09.md`;
the prior-art search is in `docs/NOVELTY_CHECK_2026-09-03.md` (copies in both manuscript repos).

## 2. Why the adjustment was needed — the comment the author received

After reading Panda Diplomacy the author asked whether it overlapped his own work and whether
his was weaker. The assessment he received (an external comment, 3 September 2026) is the
trigger for everything you are reviewing. Its substance, condensed and partly quoted:

> There is substantial thematic overlap at the research-programme level — both say *don't treat
> representation learning for detectors as generic ML applied to arrays; the structure of the
> measurement process should determine what representation we learn.* But the independent
> variables are almost orthogonal. Sam holds the learning recipe approximately fixed and varies
> the detector: **How detector-general can the representation-learning recipe be?** You hold the
> measurement problem controlled and vary the representation geometry and class,
> (Σ⁻¹) × (F): **What determines which scientific representation gets learned?** That is
> considerably more mechanistic. Sam has not done your paper before you.
>
> Why his nevertheless feels stronger: Sam closes the loop *representation → scientific
> information → useful downstream reconstruction*. Your strongest result closes *measurement
> geometry → representation → known latent structure*. **You have a mechanism. Sam has a
> consequence.** A strong paper has both. The experimental physicist asks: *so what can I
> reconstruct now that I couldn't before?* That box isn't convincingly filled yet.
>
> Panda tells the story you were trying to tell with the "four architectures" — but better.
> Generality across architectures is less scientifically convincing than generality across
> physical regimes. Don't add architecture #5 and #6; show the same covariance-as-geometry
> principle across detector/noise regimes A, B, C.
>
> Learn from his physical latent variables: don't stop at "NFPA has a smaller projector angle";
> show z₁ ↔ E, z₂ ↔ t₀, z₃ ↔ τ_rise, and ask which representation — MSE or Σ⁻¹ — organises its
> latent space according to the actual physical degrees of freedom. Your appendix's amplitude
> RMSE 0.198 → 0.143 and decoded-amplitude correlation 0.13 → 0.70 is much closer to Panda's
> latent-space story than the main paper makes apparent. Develop that aggressively.
>
> The stronger eventual story: (1) measurement theory — the detector covariance defines
> distinguishability; (2) representation theory — (Σ⁻¹, F); (3) controlled evidence — wrong
> geometry learns the wrong latent structure; (4) semantic evidence — latent coordinates align
> with physical variables; (5) real detectors — the same principle predicts when covariance-aware
> learning helps; (6) scientific consequence. Chain:
> detector → noise covariance → information geometry → representation → physical latent
> variables → better reconstruction. That is not Panda's paper, and it is conceptually more
> interesting — but only if you stop expanding horizontally across methods and close the chain.

The author accepted this and took one further decision that you must review as a decision, not
re-litigate: **Paper 3 becomes the main paper of the programme, and Paper 1 becomes the leading
service paper** (先行服务型) — it supplies the theory, the instruments and the controlled
evidence that Paper 3 uses, and changes to serve Paper 3.

## 3. How the adjustment was made

The design note is `THEME_ADJUSTMENT_2026-09-03.md` (identical copies in all three repos:
Paper 1 `plan/`, ORACLE `docs/`, experiment repo `docs/`). Read it first. In short, the
determinants of a learned representation for a calibrated readout are stated as three plus one:

| determinant | fixed by | Paper 1 | Paper 3 |
|---|---|---|---|
| Σ⁻¹ — the metric | the instrument (calibration) | theory + controlled evidence | inherited |
| F — the representable class | the architecture | theory + controlled evidence | inherited; frozen subjects are instances |
| S — the excited support | the training data | new Prop. 7.3 | the S-family lever |
| Σ̂ vs Σ — assumed vs realised covariance | deployment | out of scope | the N-family lever |

Paper 1 closes `detector → Σ⁻¹ → geometry → representation → physical latent variables` on
controlled data; Paper 3 closes `→ physical latent variables → consequence / knowing when not
to trust it` on frozen models and real detectors.

**Paper 1 changes (commit `be35292`; earlier the same day `a9cc004` added the Kieseler-standard
material — a related-work paragraph "Learned geometry versus measured geometry" on GravNet,
object condensation and Panda; a "Parameter count and cost" paragraph in §6; the (Σ⁻¹, F)
pipeline figure in §2.4):**

1. `01_intro.tex` — a paragraph at the end of the theoretical contribution: the question, the
   three determinants, "theory and controlled-evidence half of a two-part programme", what is
   handed to the companion study.
2. `06_loss_geometry.tex` — new §7.4 *From observation geometry to representation geometry*
   with Prop. 7.2 (pullback metric: I(z) = J_gᵀΣ⁻¹J_g, and J_hᵀΣ⁻¹J_h on any representation)
   and Prop. 7.3 (excited support: the population objective identifies the representation only
   on T_S = span{Σ^{-1/2}∂g/∂z_j : j ∈ S}; along T_S^⊥ the model is fixed by architecture and
   initialisation; rank I(z) < m ⇒ not locally identifiable). Proofs in
   `appE_symmetry_geometry.tex` §E.5 `app:representation-geometry`.
3. `08_discussion.tex` — the companion-study pointer now names Props. 7.2–7.3 as its predictions.
4. `07b_separable_evidence.tex` (not in the build) — reordered: "Which axes carry the physics"
   (probe R², axis participation; experiment P1-E9) first, then factor recovery, then metric
   reversal; the figure placeholder is now the paper's intended Figure 1 with panel (a) = latent
   axes × physical variables under MSE vs Σ⁻¹.
5. `TODO.md` — role statement; abstract rewrite pending; instrument table to become "three noise
   regimes, one principle" with a predictor column (P1-E10); attention ladder P1-E8c demoted.

**ORACLE changes (commit `0760c37`; earlier `ed01ca6` added the positioning paragraph, a
reporting standard, bridge metrics and the monitor-stack figure):**

1. New title *What a Frozen Detector Representation Resolves: Measurement Geometry, Training
   Support, and Consequence-Aware Diagnosis*; the former title kept as a subtitle.
2. Abstract opens with the question and the determinants; closes with the probes and the
   consequence in physical units on three measurement systems.
3. §1 new paragraph *The question* with Panda as the foil (what it shows, what it does not say).
4. §3 attributes its first two mechanism statements to Paper 1 Props. 7.2–7.3.
5. New claim **C0 — physical-variable organisation**: ridge probes from every hooked
   representation to the tier's physical variables, with the pattern predicted from the
   determinants; experiment row E0b.
6. §5 opens with "three physically different measurement systems under one principle;
   generality across regimes, not architectures".
7. Everything else — contracts N/S, C1–C4, dissociation design, tiers, monitoring comparison —
   unchanged.

**Experiments (commits `78f6557`, `a6a4173`):**

- `experiments/paper1/latent_probes.py` (P1-E9): ridge probes z → {amplitude, rise, decay,
  planted latent} for CW-PCA, NFPA, Iso-MPCA, IsoPCA at α = 0; probe R², max |corr|, axis
  participation; results in `results/paper1/P1-E9/` (four seeds; partial numbers in
  `docs/EXPERIMENT_PLAN_PAPER1.md` §1b: NFPA amplitude R² 0.44–0.49, rise 0.87; CW-PCA
  amplitude 0.13; both isotropic twins ≈ 0 on every variable).
- `experiments/oracle/run_e0b_probes.py` (P3-E0b / C0): frozen-probe R², cross-validated refit
  R², axis cosine, participation, RMSE ratio, on clean / N / S windows for `lin_cw` and
  `mlp_ae`; dev results and the reading against four pre-registered predictions in
  `docs/EXPERIMENT_PLAN_PAPER3.md` §5 (one prediction restated: R² is ill-defined on S windows
  that narrow a variable's range).
- `experiments/oracle/oracle_cov/families.py`: window seeds used Python's salted `hash()`; fixed
  with `zlib.crc32`. All earlier Tier-1 dev outputs are therefore not byte-reproducible.
- P1-E10 (gain predictor across the seven CRESST channels and three instruments) is specified
  as a cluster runbook, not run.

## 4. What to review

Answer these, in order, quoting the text you judge.

**A. Does the adjustment answer the comment?** For each of the comment's six boxes (measurement
theory, representation theory, controlled evidence, semantic evidence, real detectors with a
predictor, consequence), say which paper now owns it, whether that paper *states* it or merely
*promises* it, and whether the promise is backed by an implemented experiment, a runbook, or
nothing. Be exact about the consequence box: what physical-unit consequence does each paper
actually measure today?

**B. Is the split honest and stable?** Paper 1 is declared the service paper. Check that it
does not now read as a preface — does it still make claims a reader can evaluate without
Paper 3? Check that ORACLE does not depend on the determinants being *true* without saying so:
if Prop. 7.3 were false, which ORACLE claims survive? Is that dependence written down as a risk
anywhere? Check the proposition numbers ORACLE cites against Paper 1's build (`main.aux`).

**C. The two new propositions.** Prop. 7.2 is standard Fisher information; is its
"representation statement" (J_hᵀΣ⁻¹J_h on any representation) proved or asserted, and does it
belong in Paper 1 or is it Paper 3 content? Prop. 7.3 claims the objective's Hessian "restricted
to T_S^⊥ vanishes"; the proof shows the population objective is flat in the *orientation* of a
fixed-rank projector on T_S^⊥. Is the statement stronger than the proof? Does "for nonlinear g
it holds locally" need a hypothesis on the decoder class that is not stated? Name what would
make each proposition false.

**D. The probes as evidence.** Read `latent_probes.py` and `run_e0b_probes.py`. Are the probe
R², axis participation and RMSE-ratio endpoints well-defined and fairly applied across
methods (standardisation, ridge scaling, train/test split)? Is "the isotropic twin reads out
nothing" a fair statement or an artefact of how Iso-MPCA codes are formed? Is participation
comparable between a 6-dimensional code and a 64-dimensional hidden layer? Would Panda's
authors accept these as the same kind of probe they ran?

**E. Against the three external papers.** (i) Is the description of Panda in ORACLE §1 and in
Paper 1's related work accurate and fair — no claim attributed to Panda that it does not make,
including its own stated limitations? (ii) Is the contrast with GravNet ("learned geometry is
the tool when the sensor geometry is unknown; here a calibration run measures it") a fair
reading of that paper's scope? (iii) Is "object condensation constructs a loss; here the noise
model leaves no loss to construct" a fair reading, and is "β is the nearest relative of the
Fisher-rank diagnostic" defensible or a stretch? Quote the sentence in each paper that supports
or undermines each claim.

**F. The generality claim.** The comment says generality across physical regimes beats
generality across architectures. Both papers now say so. Does Paper 1 actually deliver it — are
CRESST, GWOSC and TIDMAD presented as regimes with a *predictor* of when the metric helps, or
still as three case studies? Is P1-E10's predictor frozen before the outcome is looked at (the
CRESST gains are already published in `docs/results/CRESST_RESULT.md`) — and does that make it a
prediction or a post-hoc fit? Say how you would make it a prediction.

**G. Voice, length, mess.** Paper 1 is 68 pages; ORACLE grew from five to six. For each added
passage, say whether it is in the paper's existing register (derivation and boundary statements
for Paper 1; contract and falsifiable prediction for ORACLE), whether it could be cut by half,
and whether the concept note still reads as something the Tier-2 partner would recognise.

**H. Verdict.** One paragraph per paper: does the adjustment make it stronger, messier, or
neither; the three edits you would make first, each with the exact sentence to cut, move or
reword; and one sentence on whether the programme as a whole now answers the comment's central
challenge — *so what can I reconstruct now that I couldn't before?* — or still defers it.

## 5. Rules

- Do not edit any file. Do not run experiments; you may run the two probe scripts with
  `--smoke` to check they execute.
- Do not read or cite `paper_rps2026/` — that submission is under double-blind review until
  29 September 2026 — and treat every number in `07b_separable_evidence.tex` as embargoed:
  you may check its consistency with the results directories, not repeat it in your report.
- Distinguish "wrong", "unsupported", "not in the paper's voice", and "fine but long".
- Do not propose new theorems, new experiments beyond what §4F asks, or a restructuring
  beyond moving or cutting the passages added on 2–3 September 2026.
- Findings first, verdicts last; two pages.
