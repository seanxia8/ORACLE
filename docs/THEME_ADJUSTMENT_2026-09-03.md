# Theme adjustment after the Panda comparison (2026-09-03)

Author's decision: Paper 3 becomes the main paper; Paper 1 becomes the leading service paper
(先行服务型) — it supplies the theory, the instruments, and the controlled evidence Paper 3 uses.
This note fixes what each paper now claims, what moved, and what stays out.

## The programme question (both papers share it)

**What determines which scientific representation a detector model learns — and what does
that representation let a physicist reconstruct?**

Panda asks how detector-general a learning *recipe* can be (fixed recipe × detectors).
We hold the measurement controlled and vary the determinants of the representation:

| determinant | fixed by | Paper 1 | Paper 3 |
|---|---|---|---|
| Σ⁻¹ — the metric: which distinctions the measurement can make | the instrument (calibration) | theory + controlled evidence | inherited |
| F — the class: which representations the architecture can express | the architecture | theory (template → subspace → product) + controlled evidence | inherited; the frozen subjects are instances of F |
| S — the support: which latent directions the training excited | the training data | **theory (new §7.4)**: the objective identifies the representation only on T_S | the experimental lever: S families |
| Σ̂ vs Σ — assumed vs realised covariance | deployment | out of scope (Σ known) | the experimental lever: N families |

The chain the comment asks us to close:

```
detector → Σ⁻¹ → information geometry → representation → physical latent variables → better reconstruction
```

Paper 1 closes `detector → Σ⁻¹ → geometry → representation → physical latent variables` on
controlled data and states the mechanism; Paper 3 closes `→ physical latent variables → better
reconstruction / knowing when not to trust it` on frozen models and real detectors.

## Paper 1 — leading service paper

Title unchanged. One-sentence thesis, sharpened: *for a calibrated readout the representation
is not chosen, it is determined — by the measurement metric, the representable class, and the
excited training support — and this paper proves the first two, states the third, and shows on
planted physics that the wrong metric learns the wrong physics.*

What changes (all additive; nothing removed):

1. **Framing (§1, last contribution paragraph; §8 discussion).** Say explicitly that the paper is
   the theory and controlled-evidence half of a two-paper programme and name what it hands to
   the companion: the pullback metric, the tangent-support statement, the (Σ⁻¹, F) frame, and
   the physical-variable probe protocol.
2. **§7.4 "From observation geometry to representation geometry" (new, short).** Two
   propositions with proofs in App. E.5: (a) *pullback metric*: for x = g(z) + n the Fisher
   information of z is I(z) = J_gᵀΣ⁻¹J_g, so latent displacements are measured in detector-
   distinguishability units; the same construction through any frozen map h gives J_hᵀΣ⁻¹J_h on
   its representation — the instrument Paper 3 uses as its alarm and consequence metric.
   (b) *excited support*: in the population limit a reconstruction objective supplies
   first-order identifying information only along T_S = span{Σ^{-1/2}∂g/∂z_j : j ∈ S}; along
   T_S^⊥ the representation is fixed by architecture and initialisation, not by data. Exact in
   the linear case; local for nonlinear g. This is the third determinant, and it is what makes
   S a *prediction* in Paper 3 rather than a label.
3. **Evidence block (`07b`, after 29 Sep) reordered: physical variables first.** The headline is
   no longer a projector angle. Order: (i) which latent axes carry amplitude, rise time, decay
   time under each metric — probe R², axis participation (P1-E9); (ii) the planted factors
   recovered axis by axis (P1-E1); (iii) metric reversal at the separable class. The 0.198 → 0.143
   amplitude RMSE and 0.13 → 0.70 correlation move from a sentence to the first figure.
4. **Real detectors reframed as noise regimes, with a predictor (P1-E10).** The three instruments
   are three regimes — CRESST (stationary coloured, well-calibrated), GWOSC (misspecified /
   nonstationary), TIDMAD (stationary, calibrated, with recorded truth) — and the principle must
   *predict* when covariance weighting helps: a per-channel anisotropy predictor (how far Σ is
   from isotropic inside the signal subspace) against the observed same-rank gain, across the
   seven CRESST channels (1.6 % … 62 %) and the three instruments. Same principle, three
   physical systems: the generality the comment asks for, without adding architectures.
5. **Horizontal expansion demoted.** The attention ladder (P1-E8c) is no longer a main-text
   experiment; the PE uniqueness stays as theory (cheap, new), its numerical checks E8a/b stay,
   and E8c becomes an optional boundary. No architecture #5.
6. **Consequence stated where Paper 1 has it, deferred where it does not.** TIDMAD amplitude
   RMSE (physical units, recorded truth) and CRESST likelihood residuals are the consequences
   Paper 1 can claim; energy resolution, thresholds, and deployment are handed to Paper 3 with a
   sentence, not implied.

## Paper 3 / ORACLE — main paper

New working title: **"What a Frozen Detector Representation Resolves: Measurement Geometry,
Training Support, and Consequence-Aware Diagnosis"** (subtitle keeps "failure diagnostics" for
continuity with the collaboration note).

One-sentence thesis: *a learned detector representation is determined by the measurement metric,
the model class, and the excited training support; a frozen model therefore carries a contract,
and its representation — read in the pullback metric — can say which determinant moved, whether
the physical variables it encodes are still resolved, and whether the result matters in physical
units.*

What changes:

1. **Gap and contribution rewritten around the question**, with Panda as the foil: Panda shows
   one recipe learns useful representations on three detectors and probes them for physics; it
   does not say why the representation takes the form it does or when it stops resolving.
   ORACLE answers the second question on Panda-class models, with Paper 1's determinants as the
   predictions.
2. **New claim C0 — physical-variable organisation.** On every tier, linear probes from hooked
   representations to the physical variables the tier defines (Tier 1: amplitude, rise, decay,
   the planted latent; Tier 2: energy, zenith, azimuth, vertex; Tier 3: injected amplitude).
   Predictions from the determinants: probe R² is high only for variables in T_S; under N-family
   cells the probes for excited variables degrade while their axes stay put; under S-family cells
   the probes for the shifted coordinate were never there. This is the Panda probe carried into
   the contract, and it is where "the latent space contains physics" becomes a *prediction*.
3. **Consequence in physical units on every tier** (already: amplitude / angular resolution /
   exclusion limit) is named as the closing box of the chain, not as one endpoint among four.
4. **Regime breadth stated as the generality claim**: three physically different measurement
   systems (cryogenic waveform, neutrino-telescope point cloud, SQUID time series) under one
   principle — not one architecture across benchmarks.
5. **Everything else stays**: contracts N/S, mechanism §3, C1–C4, dissociation design, tiers,
   monitoring comparison, reporting standard, roadmap. The diagnostics are now the *test* of the
   determinants, which is a stronger reason for them to exist than "monitoring".

## What Paper 1 must not do now

- Not claim consequence beyond what it measures (TIDMAD amplitude RMSE; CRESST residuals).
- Not add architectures; not run E8c before the theory statement is written.
- Not move the N/S contracts or any failure-diagnostics content into Paper 1 — they belong to
  the main paper; Paper 1 only supplies the two propositions and the probe protocol.

## Division of labour, restated

Paper 1: Σ known; (Σ⁻¹, F); the third determinant stated; controlled evidence with planted
physics; three noise regimes with a predictor; instruments (pullback metric, probe protocol).
Paper 3: frozen models; determinants moved (N, S); semantic organisation (C0); consequence in
physical units; real detectors; abstention when the Fisher rank drops.

## Experiments added / changed

- **P1-E9** physical-variable probes (implemented; running): probe R², max |corr|, axis
  participation per variable and method at α = 0; figure = latent organisation MSE vs Σ⁻¹.
- **P1-E10** gain predictor across CRESST channels and the three instruments (cluster runbook:
  needs the per-channel PSDs on `/ceph`): predictor = anisotropy of Σ inside the fitted signal
  subspace; outcome = observed same-rank gain; acceptance = rank correlation over channels.
- **P3-C0** semantic probes: Tier 1 implemented on the existing subjects (`Subject.represent`,
  known z); Tier 2 added to the runbook (NuBench labels per event); Tier 3 injected amplitude.
- **P1-E8c** demoted to optional.
