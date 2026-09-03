# Novelty check against GravNet, object condensation, Panda, and the shift literature (2026-09-03)

Question asked: GravNet already learns detector geometry; Panda offers a detector-agnostic
sensor-level pre-training recipe. Does the geometry-aware reconstruction (Paper 1) or the
diagnostics program (ORACLE) lose its reason to exist? Is ORACLE endangered by either group,
or by the wider literature?

Short answer: neither paper collides with Kieseler or Young. Both of their programmes are
upstream of ours — they build the models; Paper 1 says what a model must contain in the
calibrated regime, ORACLE says what a frozen model can still resolve when the regime moves.
The real risks are different from the one feared, and are listed in §4.

## 1. Paper 1 versus GravNet / object condensation

| | GravNet / GarNet (1902.07987) | Paper 1 |
|---|---|---|
| Problem | Clustering overlapping showers on an irregular calorimeter (multi-object, point cloud) | Amplitude / subspace reconstruction of waveform readouts with a calibrated noise model |
| Geometry | Learned coordinate space S, potential exp(−d²), per model ≈100k params | Measured: Σ⁻¹ from calibration noise; Fourier basis with 1/J_k weights when stationary |
| Loss | Designed (energy-fraction weighted, Eq. 2) | Derived: unique G_noise-invariant residual functional |
| Claim type | Construction + benchmark | Uniqueness + exact equivalences + truth-referenced recovery |

Not the same problem, not the same claim. The relation is a contrast, now written into
Paper 1's related work ("Learned geometry versus measured geometry"): learned geometry is the
tool when the measurement geometry is irregular or unknown; Paper 1 is the case where a
calibration run measures it. Object condensation's potential-loss argument is the mirror image
of Paper 1 §7: he constructs a loss and shows its minimiser has the wanted structure; Paper 1
shows the noise model leaves no loss to construct.

Paper 1's novelty risk is elsewhere and was already known: its equivalences are "known in
pieces" (GLS, EMPCA, tied AE → PCA). The defence is the `What this paper adds` paragraph, the
exact finite-sample statements, the NFPA truth-referenced recovery (the only class where
recovery against planted truth has been measured), and — if it lands — the PE uniqueness
corollary, which would be the one theorem nobody has stated.

## 2. Paper 1 versus Panda Diplomacy (2609.00611, Young–Jesús-Valls–Terao)

Panda: point-cloud self-distillation pre-training, one recipe on LArTPC, water Cherenkov,
collider TPC; frozen representations show semantic organisation; adaptation with ~10³ labels
per detector. Its ambition is detector-agnostic *learning from data alone*.

Paper 1 is waveform readouts with a calibrated Gaussian noise model; no pre-training, no point
clouds. The two overlap only in ambition. The honest framing, already in Paper 1's discussion
and related work: in the noise-dominated calibrated regime a data-only recipe must rediscover
Σ⁻¹ or pay for the difference; Panda's regime (imaging detectors, sparse hits, semantics) is
one where a calibrated covariance is not the dominant structure. Complementary, not competing.
No sentence in Paper 1 needs to change because Panda exists.

## 3. ORACLE versus all three, and versus the shift literature

ORACLE does not build a reconstruction model. Its subjects are exactly the models the other
groups build: DynEdge (NuBench), a compact TIDMAD transformer, and — once weights are public —
Panda V2. The proposal's positioning paragraph (added today) says so.

What ORACLE has that none of them has: the N-versus-S contract under a stated forward model;
the pullback metric J^⊤Σ⁻¹J as the consequence measure; the excited/unexcited tangent
decomposition as the mechanism behind attribution; the Fisher-rank abstention region; the
designed null/aligned dissociation; and designed paired data (ORACLE-Cov, ORACLE-Paired) on
which the contract is falsifiable. Object condensation's β is the nearest relative of the Fisher
rank (a learned resolvability weight); ORACLE computes the same quantity for a frozen model with
no training signal.

Web search on 2026-09-03 (not a literature review; see §5) found no paper doing frozen-model,
mechanism-level N/S diagnosis for detector data. Nearest neighbours:
- Chu, Liu, Biswas, Shen, "Do Physics Foundation Models Learn Generalizable Physics?"
  (arXiv:2605.29283): a bias-aware benchmark of PDE-forecasting foundation models across
  regimes and shifts — score tables, "conditional generalists", no mechanism, no detector data.
  Cited in the proposal as the score-only contrast.
- DECIDER (Springer LNCS, ECCV 2024): failure detection in vision using foundation-model
  priors. Generic; no physics forward model.
- Real-time anomaly detection for LArTPCs (arXiv:2509.21817): DAQ-level anomaly detection, not
  representation diagnostics.
- HistoAE (arXiv:2511.22246): interpretable unsupervised charge/position measurement on
  silicon microstrips; no covariance, no shift.
- Calorimetry foundation model with MoE experts (arXiv:2603.28804): simulation surrogate;
  adaptation, not diagnostics.
- "Data Whitening Improves Sparse Autoencoder Learning" (arXiv:2511.13981) and DyWPE
  signal-aware wavelet PE (arXiv:2509.14640): whitening / data-adaptive PE as engineering,
  not derived from a noise likelihood — relevant to Paper 1's PE section as contrasts, not
  collisions.
The generic dataset-shift line (Rabanser 2019, Podkopaev–Ramdas 2022, Zhang et al. 2023,
Roschewitz 2025) is already cited and is what the "generic monitor" baseline arm represents.

## 4. Where the actual risks are

1. **A Panda/NuBench robustness benchmark paper.** The most plausible near-term event is a
   DeepLearnPhysics or NuBench author writing "how robust is Panda V2 / DynEdge to detector
   shifts" as a benchmark. It would be a score table, not a mechanism, but it would take the
   empirical "first" on the Tier-2 side. Mitigation: get the Tier-1 mechanism (ORACLE-Cov) on
   arXiv as a standalone, pre-registered result, independent of Tier 2; and offer Sam's group
   Panda V2 as the Tier-2 subject (turns the likeliest competitor into a co-author — already
   the plan).
2. **Scope, not scooping.** Three tiers, an own paired production, and a re-emulated detector
   response is more than one author can finish before someone else moves. The threat to
   ORACLE is not finishing. The reordering already made — Tier 1 decides every theoretical
   claim; Tier 2 is a single confirmatory run — is the right shape; keep it.
3. **Paper 1's "known in pieces" objection.** Not affected by Kieseler or Young; handled by the
   uniqueness results and the NFPA recovery. The PE uniqueness corollary would strengthen it
   further.
4. **Being read as a GravNet competitor.** Only if Paper 1 ever claims something about
   irregular-geometry detectors. It does not; the new related-work paragraph draws the line.

## 5. What this check is not

A web search with five queries is not a literature review. Before either submission, run an
INSPIRE and arXiv listing pass on: "frozen representation" + "covariance shift" + detector;
Panda Diplomacy citations; NuBench citations; "foundation model" + "diagnostics" + physics;
"noise-aware positional encoding"; "matrix-normal PCA" + detector. Update the positioning
paragraphs if a mechanism-level diagnostic or a likelihood-derived PE appears.
