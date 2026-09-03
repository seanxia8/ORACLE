# ORACLE `dev` — update of 2–3 September 2026

Two commits on `dev` since `e0c1299` (implementation plan rev. 2):

**`0793eba` — proposal: the mechanism section.**  `latex/paper3_proposal.tex` gains §3
"Mechanism: what a frozen representation can resolve", between the failure contracts and
the claims.  It derives the consequence metric (the Fisher/pullback form $J^\top\Sigma^{-1}J$,
i.e. the `PullbackWhitened` statistic the implementation plan already named, and its
$\Sigma$-free surrogate, the Jacobian-projected displacement); states the coverage result
(training excites only $T_S=\mathrm{span}\{\Sigma^{-1/2}\partial g/\partial z_j\}$, so a
support shift into the complement is invisible to the trained sensitivity, and a Fisher-rank
drop defines the abstention region); contrasts it with an acquisition shift (residual
variance along $T_S$ departs from one); and defines the two layerwise monitors (dominant
Jacobian-subspace alignment $a_{\ell,k}$ and contrastive directions $v_{\ell,h}$).  The gap
paragraph and Phase 2 now cite Panda Diplomacy (arXiv:2609.00611) as the cross-modality
frozen subject whose authors list detector and sim-to-real transfer as untested.  Builds to
five pages (was four).  `reference/papers/papers.tsv` gains the Panda V2 row.

**`5878250` — README.**  The layout table points at `docs/archive/` for the superseded
audit / open-decisions / revision-plan documents and lists the three current documents
(`EXPERIMENT_DESIGN.md`, `IMPLEMENTATION_PLAN.md`, `REVIEW_PROMPTS.md`).

**Decision recorded elsewhere (not a commit here).**  All experiment code for Paper 3 lives
in `noise-weighted-subspace-reconstruction/experiments/oracle/` (package `oracle_cov`), not in
this repository; ORACLE keeps the proposal and the four packages.  A Tier-1 (ORACLE-Cov)
implementation exists there with nine passing analytic tests and dev-scale results
(3 seeds).  Three findings from those runs bear on §3 and should be folded into the next
proposal revision:

1. The N-versus-S contract maps onto more than two geometric signatures: covariance-type N
   changes the residual variance in $T_S$ with no mean shift; jitter (N by contract) and
   channel loss move the mean like S; only $t_0$ and nonseparable shifts land in the unexcited
   complement.  §3's sentence "N in the excited span, S in the complement" needs that
   qualification.
2. Conformal abstention on classifier margin does not detect undeclared families
   (AUROC 0.25); a feature-space Mahalanobis nonconformity does (0.71).  C2's abstention rule
   should say so.
3. "C4 within severity strata" has no consequence variation on Tier 1; the informative tests
   are the designed dissociation (which holds: pullback 19.6 vs −0.1, Euclidean cannot
   separate) and ranking across cells.  The Tier-1 consequence should be the whitened
   reconstruction error, not the amplitude readout, which is noise-dominated at SNR 10.

Full detail: `noise-weighted-subspace-reconstruction/docs/reviews/2026-09-02_experiments_review.md`
and `docs/plans/oracle/PREREGISTRATION.md` (draft) in that repository.
