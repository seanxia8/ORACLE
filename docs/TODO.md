# Paper 3 / ORACLE — TODO (as of 2026-09-03, branch `dev`)

Stage: proposal has its mechanism section (`0793eba`); Tier-1 implementation lives in the
experiment repository with dev-scale results; nothing here is pre-registered or citable yet.

## Theme adjustment 2026-09-03 (`docs/THEME_ADJUSTMENT_2026-09-03.md`)
Paper 3 is the main paper: *what determines which representation a detector model learns, and
what it lets a physicist reconstruct*.  Done today in the proposal: new title, abstract opening,
"The question" paragraph in §1 with Panda as the foil, mechanism §3 attributed to Paper 1
Props. 7.2–7.3, new claim **C0** (physical-variable organisation), E0b row, measurement-system
breadth sentence in §5.  Build 6 pp.
- [ ] **Junjie must see and agree** the new title and C0 before anything is frozen; the
      failure-diagnostics content is unchanged, the framing is not.
- [ ] Fold the three Tier-1 findings (below) and the P3-E0b dev result into §3–§4.
- [ ] Consequence in physical units per tier stated in one table (amplitude RMSE; angular
      resolution; exclusion limit) — the closing box of the chain.
- [ ] C0 into `PREREGISTRATION.md` §4 (experiment repo) with the acceptance rule; probe step into
      `TIER2_RUNBOOK.md` (NuBench labels) and `TIER3_RUNBOOK.md` (injected amplitude).

## Proposal (`latex/paper3_proposal.tex`)
- [ ] Fold the three Tier-1 findings of `docs/DEV_UPDATE_2026-09-03.md` into §3 and §4:
      (i) three-signature N/S table (covariance N: variance in T_S; signal-deforming N: mean
      shift like S; S along excited vs unexcited coordinates); (ii) abstention = feature-space
      novelty, not classifier margin; (iii) Tier-1 consequence = whitened reconstruction error,
      C4 primary = pooled ranking + designed dissociation, not within-(family, severity) strata.
- [ ] State where the designed dissociation is constructible (stages with a Jacobian null
      space) and that the pullback monitor uses the *assumed* covariance.
- [ ] Choose the one claim to lead with (recommendation: N-vs-S mechanism + C4) and trim C1/C3
      to supporting; C3 as written fails at 10 % sampling on Tier 1.
- [ ] Bibliography: `paper1` gets the arXiv id once posted; add Lu 2008 / Allen 2014 only if §3
      cites the separable class.
- [x] Rebuild and check the page count — now six after the 2026-09-03 additions (positioning
      paragraph, reporting standard, bridge sentence in E5, monitor-stack figure); README updated.
- [ ] If a five-page limit applies for the collaboration note, drop the `fig:testbeds` panel (b)
      or compress §7 roadmap — do not cut the reporting standard or the mechanism section.

## Method standard adopted from Kieseler (2026-09-03; notes in the experiment repo
`docs/background/KIESELER_METHOD_NOTES_2026-09.md`)
Done today in the proposal: "Position relative to learned-geometry reconstruction" (GravNet,
object condensation, Panda, the bias-aware physics-FM benchmark 2605.29283) in §1; "Reporting
standard" paragraph in §5 (parameter-matched arms, resource row, inclusive + stratum metrics,
threshold curves, extrapolation arm, hyperparameter table); Figure 1 monitor stack.
- [ ] `PREREGISTRATION.md`: add the reporting standard as hard fields — per arm `n_params`,
      `train_cost`, `infer_ms_per_event`, `mem_mb`; per endpoint inclusive + stratum; per
      threshold a curve; one hyperparameter table.
- [ ] Tier-1 `mlp_ae` vs `cwpca` vs `nfpa`: match parameter counts (review C1) before any
      confirmatory run; report the resource row.
- [ ] Tier 2: baseline is the *published* NuBench DynEdge configuration, not a reimplementation;
      any in-project re-emulation of the detector response is validated against the released
      per-geometry numbers (parity gate in `TIER2_RUNBOOK.md`).
- [ ] Extrapolation arm on Tier 1: monitors calibrated on severities ≤ s_max, evaluated at
      2 s_max and at the unseen family — report as its own row, not pooled.
- [ ] Bridge table with identical metric names on both tiers (within-T_S residual variance,
      along-T_S^⊥ displacement, abstention risk–coverage AUC, ρ(A,K)) — fill from the
      confirmatory Tier-1 output.

## Novelty positioning (2026-09-03 search; see `docs/NOVELTY_CHECK_2026-09-03.md`)
- [ ] Re-run the search before submission: "frozen representation" + "covariance shift" +
      detector; Panda follow-ups; NuBench follow-ups; any "diagnostics of foundation models in
      physics" paper. Update the positioning paragraph if a mechanism-level diagnostic appears.

## Pre-registration (in the experiment repo, `docs/plans/oracle/PREREGISTRATION.md`)
- [ ] Freeze `core` (endpoints, thresholds, families, margins) — write `FROZEN: <commit>`;
      confirmatory runners refuse to start without it.
- [ ] Raise clean windows to ≥ 100 per seed before any confirmatory run (FAR resolution).
- [ ] Fill the Tier-1 → Tier-2 bridge table (§6) with a row per ORACLE-Paired family.

## Packages here
- [ ] `src/prometheus_simulation`: Prometheus adapter (`prometheus_io`), NuBench response
      reimplementation, clean-twin matching validated on the toy set (WP9).
- [ ] Subject adapters for Tier 2/3 following the `Subject` interface in
      `experiments/oracle/oracle_cov/subjects.py` (`represent`, `outputs`, `jac_recon`, `jac_output`).
- [ ] `reference/papers/`: fetch 2609.00611 PDF (row added to `papers.tsv`).

## Collaboration
- [ ] Send Junjie the dev update + the two runbooks; agree the Tier-2 family list and the
      angular-error consequence before the bridge table is frozen.
- [ ] Panda V2 weights: watch for the release; until then DynEdge remains the Tier-2 subject.
