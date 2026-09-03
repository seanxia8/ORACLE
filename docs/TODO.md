# Paper 3 / ORACLE — TODO (as of 2026-09-03, branch `dev`)

Stage: proposal has its mechanism section (`0793eba`); Tier-1 implementation lives in the
experiment repository with dev-scale results; nothing here is pre-registered or citable yet.

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
- [ ] Rebuild and check the page count (currently five); README says five.

## Pre-registration (in the experiment repo, `docs/plans/oracle/PREREGISTRATION.md`)
- [ ] Freeze `core` (endpoints, thresholds, families, margins) — write `FROZEN: <commit>`;
      confirmatory runners refuse to start without it.
- [ ] Raise clean windows to ≥ 100 per seed before any confirmatory run (FAR resolution).
- [ ] Fill the Tier-1 → Tier-2 bridge table (§6) with a row per ORACLE-Paired family.

## Packages here
- [ ] `src/oracle_paired`: Prometheus adapter (`prometheus_io`), NuBench response
      reimplementation, clean-twin matching validated on the toy set (WP9).
- [ ] Subject adapters for Tier 2/3 following the `Subject` interface in
      `experiments/oracle/oracle_cov/subjects.py` (`represent`, `outputs`, `jac_recon`, `jac_output`).
- [ ] `reference/papers/`: fetch 2609.00611 PDF (row added to `papers.tsv`).

## Collaboration
- [ ] Send Junjie the dev update + the two runbooks; agree the Tier-2 family list and the
      angular-error consequence before the bridge table is frozen.
- [ ] Panda V2 weights: watch for the release; until then DynEdge remains the Tier-2 subject.
