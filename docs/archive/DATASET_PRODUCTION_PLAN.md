# Dataset production plan

_31 August 2026. Answers the meeting note's open question — "start with NuBench,
need same events across different geometries, simulation available?" — and the
questions behind it: what is actually blocking us, what do we need, can
Prometheus supply it, and what exactly do we build._

Companion to `MEETING_AGENDA.md`, `OPEN_DECISIONS.md` (D1–D8),
`NOVELTY_REVIEW.md`, `PAPER3_AUDIT.md`.

---

## 1. The bottlenecks, named

The meeting note circles five problems without naming the first one, which is
the one that matters.

**B1 — No existing dataset has the property the paper is about.** The agreed
claim (`NOVELTY_REVIEW.md` §5) rests on the consequence variable being *a
physics observable under a known measurement covariance*. Every candidate in
the note supplies at most half of what that needs:

| Candidate | Has | Missing |
|---|---|---|
| NuBench | six geometries, four architectures, real reconstruction task | no event pairing, no known Σ, §3.2 response unreleased, no licence |
| TIDMAD | real data, public, waveform-level; D2 salvages a usable consequence variable as `K_rel` | no geometry axis and no event pairing at all — it cannot answer this question whatever K we use |
| MicroBooNE / LIGO / JUNO / FASER | realism | no controllable covariance; weeks of domain entry each |
| Our controlled simulator | **known Σ̂ and Σ**, full intervention control | it is ours, so a referee calls it a toy unless a public arm corroborates it |

The conclusion the note has not yet drawn: **the dataset we need does not
exist, so producing it is not a prerequisite to the contribution — it is part
of the contribution.** "Simulation available?" has the answer "yes, and you
will have to run it yourself."

**B2 — The N/S confound is a dataset-design problem, not an analysis one**
(audit P1.1). Module dropout and hit thinning *manufacture* low-multiplicity
events, which are exactly members of the declared S strata. Without
content-matched cells, a positive Claim A result is fingerprint recognition of
our own corruption operator. This cannot be repaired after production: it
dictates the sample list and the overproduction factor (§5).

**B3 — Scope inflation is the largest schedule risk in the note.** Ten
candidate testbeds are listed (TIDMAD, NuBench, Panda, LIGO+Virgo+KAGRA,
MicroBooNE, plus five DORAEMON domains) against a diagnostics layer that is
still almost entirely unwritten — the six DynEdge hook points are settled (D4)
and `pooled_representation` is in, but there is no displacement or
principal-angle metric, no MMD or C2ST baseline, no conformal abstention and no
Jacobian projection. Each additional domain is weeks of entry cost for a
marginal replication. The audit already ranked these; the note re-opens a
settled question.

**B4 — Licence and collaboration risk, correctly flagged.** NuBench artifacts
carry no licence at all; SPINE is collaboration data. Our own production has
neither problem. This is an argument for producing rather than consuming, not
merely a caveat.

**B5 — The 0.944° parity blocker gates the whole NuBench arm.** Until D5's
cheap CPU-vs-GPU test is run, no NuBench-derived number is citable. It costs
an afternoon and it is upstream of every NuBench decision in the note.

**B6 — "Would people really use it?" is the right question with an
uncomfortable answer.** A universal monitoring framework nobody asked for will
not be adopted. A dataset with a property no other dataset has will be. This
should reorder how the repo is presented (§7).

---

## 2. What we actually want

Distilled from the note's diagnostics list and the audit's claims:

| | Requirement | Why |
|---|---|---|
| **R1** | Event pairing across the varied factor — same latent event, different geometry | the note's "same events across different geom"; without it the geometry axis is a distribution comparison, not a causal layout claim |
| **R2** | Content matching across the N/S contrast — same observed multiplicity, charge, time spread; different cause | closes B2; without it Claim A is indefensible |
| **R3** | **Known assumed Σ̂ and realized Σ** | the covariance-geometry result (audit §6.3) — the one part no one else can write |
| **R4** | A consequence variable with genuine physics meaning | angular error, not a denoising score (D2) |
| **R5** | A declared-vs-undeclared family split | C3 abstention needs families the model never saw |
| **R6** | Cross-architecture replication on identical data | the note's "cross ML architecture validation" |
| **R7** | Licence-clean and redistributable | closes B4; makes the dataset itself a citable artifact |

---

## 3. Can Prometheus meet this? Five of seven — and it misses the headline

| | Prometheus + GraphNeT | Notes |
|---|---|---|
| R1 pairing | **Yes** | verified in D1: run LeptonInjector once, then loop geometries with `injection.lepton_injector.inject = False` and `paths.injection_file = <existing .h5>`, setting `detector.geo_file` per run |
| R2 matching | **Partly** | matchable on *injected* quantities (vertex, direction, energy) at generation; matching on *observed* multiplicity requires post-hoc selection from a larger pool — hence the overproduction factor in §5 |
| R3 known Σ | **No** | decisive, but not for the obvious reason. We write the §3.2 response ourselves (§5.3), so the smearing widths and noise rate *are* ours to set. The gap is deeper: the measurement model here is photon-level Poisson counting through Cherenkov propagation, which induces no closed-form covariance over the pulse-level representation. There is no Σ̂ the encoder can be said to *assume*, hence nothing to misspecify and nothing to invert. κ(Σ̂⁻¹Σ) is undefined in this testbed |
| R4 consequence | **Yes** | angular error on direction reconstruction |
| R5 declared/undeclared | **Yes** | we choose which families enter the training support |
| R6 architectures | **Probably** | DynEdge, ParticleNeT, GRIT and DeepIce are released as GraphNeT model artifacts; writing our data in the GraphNeT Dataset formats should let them run unchanged. Hook points are verified in source for **DynEdge only** (D4) — assume nothing for the other three until checked |
| R7 licence | **Yes for ORACLE-Paired** | Prometheus is LGPL-2.1; our seeds, our provenance, redistributable. **Not yet true for ORACLE-Cov** — `src/noise_module/` has an unresolved IP position, the repo-level LICENSE names only seanxia8, and the standing instruction is not to publish the module before the ORACLE preprint. See §7 |

**The verdict that changes the plan: R3 is the paper's actual novelty, and
Prometheus cannot supply it.** So this is not one dataset with a geometry axis
bolted on. It is **two datasets with different jobs**:

- **ORACLE-Cov** — the controlled waveform simulator. Carries R3, and with it
  the covariance-geometry lemma and the output-null dissociation. This is the
  headline and it is unblocked today.
- **ORACLE-Paired** — the Prometheus production. Carries R1, R2, R4, R5, R6,
  R7. This is the realism and transfer arm, and a citable dataset in its own
  right.

The audit reached the same conclusion from the other direction (§6.3: "the
*controlled simulator*, not TIDMAD, should be the primary waveform testbed").
NuBench is corroboration, not foundation. Ordering the work the other way round
is the single most expensive mistake available here.

### On the physics axis

_Source note: the injector details below were read from Prometheus
`prometheus/config_types.py` and `prometheus/injection/__init__.py` on
31 Aug 2026. They are not yet recorded in `OPEN_DECISIONS.md`, whose
verified-in-source list covers only the injection-file reuse mechanism.
One unresolved point: `config_types.py` exposes a `prometheus_injector`
taking only an `injection_file`, but `injection/__init__.py` registers
loaders for `LEPTONINJECTOR` and `GENIE` only — so feeding it a hand-written
event list with an arbitrary primary is **not** confirmed. Do not design
around it before checking._

Same geometry, different physics *cannot* be event-paired — the primary is the
thing being varied, so no "same event" survives. It can be **matched**:
`lepton_injector`'s `final_state_1` / `final_state_2` give νμ CC (track), νe CC
(cascade), ντ CC (double-bang) and NC from one configuration family, and the
`genie` injector additionally exposes `placement: "fixed"` with explicit
`positions`, `direction_mode: "as-is"` and `interaction_filter: "CC"/"NC"` —
same vertex, same direction, different interaction. Write it in the paper as
matched, never as paired.

Two cautions. There is no dark-matter channel in a neutrino telescope: a
nuclear recoil makes no Cherenkov light, so "DM" here means neutrinos *from* DM
annihilation, i.e. a different spectrum and zenith band. DM recoils belong to
ORACLE-Cov. And Prometheus bundles no CORSIKA or MuonGun, so an atmospheric-muon
class must be approximated by νμ CC events injected with the vertex outside the
instrumented volume (`injection_radius`, `endcap_length`, `cylinder_*`), giving
a bare through-going track.

---

## 4. ORACLE-Cov — build first, it is unblocked

The controlled simulator (`src/noise_module/`) already returns both the implied
and the realized covariance from `MultiChannelNoiseGenerator`, which is the
instrument this arm needs. Strata:

| Cell | Construction | Predicted result |
|---|---|---|
| `matched` | Σ̂ = Σ | alarm ranks consequence |
| `kappa_sweep` | Σ̂ ≠ Σ with κ(Σ̂⁻¹Σ) swept over ~1 to 10³ | rank agreement between Euclidean and whitened displacement degrades in a way controlled by κ (the audit predicts control, not monotonicity — do not over-commit) |
| `null` | latent perturbation in the null space of the local output Jacobian; large displacement A, ≈ zero consequence K | Euclidean-displacement monitors fire, consequence does not follow — **they fail by construction** |
| `aligned` | norm-matched output-aligned perturbation; same A, large K | both fire |
| `random` | norm-matched random direction | intermediate |
| `clean` | no perturbation | neither |

`null` versus `aligned` is the designed dissociation: same alarm magnitude,
opposite consequence, predicted sign, unambiguous null, no external dependency.
The dissociation itself is an afternoon; the full κ sweep with the power
analysis D6 requires is a few days. It is the hedge against the scooping vector
in `NOVELTY_REVIEW.md` §7.

---

## 5. ORACLE-Paired — the Prometheus production

### 5.1 Geometry pair

**Flower S (`orca.geo`) versus Flower L (`arca.geo`).** Both KM3NeT-family and
both in water (the only ice geometry in the set is `icecube.geo` = Hexagon), so
the contrast is about layout and not about ice versus water. ORCA is the dense
low-energy array and ARCA the sparse high-energy one, a spacing ratio of roughly
four- to six-fold — **read the exact string positions out of
`resources/geofiles/orca.geo` and `arca.geo` and quote those**, rather than the
published design figures, which vary by source. Water also matters mechanically:
olympus runs are
*exactly* reproducible, whereas ice/PPC is seedable only to Poisson-level
photon variation (D1 caveat a). Place both at a common site and depth, and size
the injection volume against the larger geometry so both detectors are
contained (D1 caveat b).

Hexagon versus Hexagon Ice LE from the released data remains usable earlier as
an explicitly distribution-level *medium* comparison — never as a layout claim.

### 5.2 Injection

One LeptonInjector run, one recorded seed, reused for both geometries:

```
run.random_state_seed          : fixed, recorded
injection.lepton_injector      : nevents 120_000
  minimal_energy / maximal_energy : 1e1 / 1e4 GeV   (common to both detectors)
                                    NuBench ran Flower S at 1e1-1e3 and
                                    Flower L at 1e1-1e5. A shared injection
                                    forces one band; 1e1-1e4 is the compromise.
                                    Declare it: any released-checkpoint
                                    comparison must be restricted to that
                                    checkpoint's own training range, or it is
                                    extrapolation.
  power_law                       : 1.0
  min/max_zenith, min/max_azimuth : isotropic
  injection_radius, endcap_length,
  cylinder_radius, cylinder_height: sized to contain ARCA
detector.geo_file              : orca.geo | arca.geo    <- the only thing that varies
```

Then per geometry: `injection.lepton_injector.inject = False` and
`paths.injection_file = <the .h5 from run 1>`.

### 5.3 The response emulation — a cost that is really an asset

NuBench §3.2 is fully specified in the paper but was never released as code:
spherical OMs at r = 30 cm with 20% QE, a ≥ 5 µs trigger window, uniformly
sampled noise photons, TTS-window photon merging, 1 ns and 0.25 pe smearing,
events cut below 4 or above 10⁶ pulses.

Reimplementing it is roughly a day. It is worth doing gladly, because **this is
where every N-family knob lives** — the noise rate, the smearing widths, the
merging window, the QE. Downloading the released tarballs would have left all of
them fixed. Owning §3.2 is what makes the N interventions physical rather than
post-hoc array surgery.

### 5.4 The sample list

From one injection, per geometry. `injection_id` is the pairing key and the
column NuBench does not have.

**Clean**

| Cell | Construction |
|---|---|
| `clean` | no intervention; the reference arm and the matching pool |

**N — acquisition contract violations** (all applied *at* the §3.2 response stage)

| Cell | Construction |
|---|---|
| `N1_module_loss` | drop 5 / 10 / 25 / 50 % of OMs at random, string-correlated variant included |
| `N2_hit_thinning` | per-pulse drop matched in *expected* multiplicity loss to each N1 level |
| `N3_timing_jitter` | inflate the 1 ns arrival smearing to 3 / 10 / 30 ns |
| `N4_gain_drift` | per-string multiplicative charge miscalibration |
| `N5_noise_rate` | raise the uniform noise-photon rate in the trigger window |

**S — supported-but-rare physics**, each content-matched to an N cell on
observed multiplicity, total charge and time spread

| Cell | Construction | Matches |
|---|---|---|
| `S1_low_energy` | natural low-E tail | N1, N2 |
| `S2_edge_vertex` | vertex near the instrumented boundary, partial containment | N1, N2 |
| `S3_horizontal` | near-horizon zenith band | N3 |
| `S4_cascade` | νe CC cascade at matched *deposited* energy against a νμ CC track | N4 |
| `S5_nc` | NC — outgoing neutrino invisible, so low visible energy | N1, N5 |

**U — undeclared families, held entirely out of the declared contract** (C3
abstention)

| Cell | Construction |
|---|---|
| `U1_tau` | ντ CC double-bang topology |
| `U2_throughgoing_mu` | νμ CC injected with vertex outside the instrumented volume — bare track, no vertex |
| `U3_pileup` | two events overlaid in one trigger window |
| `U4_correlated_noise` | a spatially correlated OM noise mode with no analogue among N1–N5 |

**The matching procedure** (this is what closes B2). Two distinct things are
being matched, and the audit's requirement is the first:

1. **Content-matched control cells.** For every N event, select its nearest
   *uncorrupted* neighbour in (observed multiplicity, total charge, time spread)
   from the overproduced `clean` pool. This is the audit P1.1 fix: it gives each
   corrupted event a physically-clean twin with the same observed content, so a
   positive Claim A result cannot be fingerprint recognition.
2. **S physics families.** S1–S5 above are separate strata defined by physics,
   each *assigned* to the N cell it is most confusable with. They are the "clean
   but rare" arm of the N/S contrast, not the output of step 1.

Where energy enters the match — S4's track-versus-cascade comparison — it is
**deposited**, never true, energy; true-energy matching lets visible energy vary
with the class and confounds exactly the quantity under test.

Budget a **3–5× overproduction factor** on `clean` for step 1. Note that this
sizes the *matching pool* only. It says nothing about per-cell statistical
power, which D6 records as **unresolved** — the pilot's overlapping CIs at
n = 256 show the low-severity cells are badly undersized. **120k is provisional
pending D6.** Run the null simulation and size windows-per-cell for 80% power
before freezing the production.

### 5.5 Cost

D1's profiling figure is ~1–10 CPU-s per event per geometry **at 1–100 TeV** —
above most of our 10 GeV–10 TeV band, so treat it as a loose upper bound and
re-profile on a 1k-event pilot before committing the queue. On that figure,
120k × 2 geometries at ~5 CPU-s ≈ **330 CPU-hours**, embarrassingly parallel —
a day or two on the cluster. The §3.2 response and the interventions are numpy
and cost minutes. This is not the expensive part of the project; the diagnostic
code that does not yet exist is.

### 5.6 Format and provenance

Write GraphNeT-compatible SQLite and Parquet so the four released
architectures run unchanged (R6 for free). Per-event provenance columns:

```
injection_id     the pairing key across geometries
geometry         orca | arca
stratum          clean | N1..N5 | S1..S5 | U1..U4
intervention     JSON of the exact parameters applied
seed             injection seed, response seed
truth            E, direction, vertex, interaction type, deposited energy
```

Publish generation scripts and config alongside the tarballs, and a seed
manifest. That is the difference between a dataset people cite and one people
cannot reproduce — and it is precisely what we found missing in NuBench.

---

## 6. Sequencing

1. **Now — ORACLE-Cov §4.** No external dependency. First citable result, and
   the scooping hedge. Run D6's null simulation here too: it is the only place
   the power analysis can be done, and it sizes §5.
2. **In parallel — ORACLE-Paired §5.** Our own production with our own seeds.
   It is **not** downstream of the parity blocker: D5 is about re-inferring
   *released* checkpoints on *released* NuBench data, which our production does
   not touch. Only claims that quote NuBench's own numbers are gated.
3. **Gate, before any released-NuBench claim or any email to the NuBench
   authors — D5, both halves.** (i) Re-run the released-metric rescoring with
   the fixed interaction/topology grouping; the 0.055° discrepancy may dissolve
   on its own. (ii) The decisive CPU-versus-GPU re-inference test for the
   0.944° offset. `MEETING_AGENDA.md` §8 asks for agreement that neither the
   email nor the claim goes out until **both** are done.
4. **Later, one sentence of future work each —** Panda (D3), LIGO, MicroBooNE,
   JUNO, FASER. Not in this paper. B3 is the risk being managed here.

---

## 7. Repository organisation, and what to lead with

The note asks whether this should be several repos. Three, but not quite the
three listed:

- **`ORACLE`** — framework, diagnostics, the noise simulator, and the
  generation scripts. One repo, the one that exists.
- **A Hugging Face dataset** — **ORACLE-Paired only, for now.** Generation
  config and seed manifest mirrored from `ORACLE`. HF is the right home for
  tarballs; GitHub is not.
- **Overleaf for the manuscript** — but keep `latex/` in the repo as the source
  of truth and sync one way, or the two will diverge within a month.

**ORACLE-Cov does not go to Hugging Face yet.** `src/noise_module/` has an
unresolved IP position: the repo-level LICENSE names only seanxia8, the package
was written for this paper, and the standing decision is not to publish it
before the ORACLE preprint is on arXiv. Publishing the dataset it generates,
together with the generation scripts, is publishing the module. Hold both until
the preprint lands, then release together.

### On "would people really use it"

A monitoring framework nobody asked for will not be adopted; a dataset with a
property no other dataset has will be. So in the **repository and the release**,
lead with the data: the paired multi-geometry set with declared strata and
content-matched cells is what a reader can pick up in an afternoon.

This is a packaging argument, not a claim-ordering one, and the distinction
matters. The **paper's** ranking stays exactly as `NOVELTY_REVIEW.md` §4 sets
it: (1) consequence as a physics observable under a known measurement
covariance — which ORACLE-Cov carries and ORACLE-Paired explicitly does not;
(2) the conditional-on-alarm C4 endpoint with designed dissociations; (3) the
first monitoring result for learned neutrino-telescope reconstruction. The
agreed §5 sentence is built around the covariance consequence variable, and
nothing here changes it. The dataset is the on-ramp; the covariance result is
the contribution.

---

## 8. How the arms prove the paper

_Added after the "where does Prometheus fit if it cannot carry the covariance
result" question. Three tiers, one direction of travel: mechanism → realism →
real data. Each tier does the one job the others cannot._

### 8.1 The division of labour

**ORACLE-Cov (noise module + compact transformer) proves the mechanism.**
The encoder is trained under an *explicit* assumed covariance Σ̂ — the
inverse-PSD-weighted objective makes "the encoder assumes a covariance" a
literal property of the loss, not a metaphor — and deployed under a realized Σ
that `MultiChannelNoiseGenerator` reports exactly. Everything theoretical
happens here: the whitening lemma, the κ(Σ̂⁻¹Σ) sweep, the output-null /
output-aligned dissociation with its sign prediction, and the D6 power
analysis that sizes every other arm.

**ORACLE-Paired (Prometheus + GraphNeT) proves it survives realism.** No Σ̂
exists there — and that is the *point*, not a defect. The theory tier predicts
which monitor family fails on which perturbation family; the realism tier
tests those predictions out-of-sample, on a frozen model nobody in this
project trained, with matched N/S cells, undeclared U families, a causal
geometry axis, and a physics consequence variable (angular error). It also
answers the practical question the theory cannot: of the monitors, only the
Jacobian-projected one is computable *without* knowing Σ — so it is the one a
real experiment could actually deploy, and Tier 2 is where that deployability
claim is earned.

**TIDMAD is the real-data echo of the mechanism.** The TIDMAD arm already
trains the same compact transformer twice — MSE versus inverse-PSD-weighted —
which is *two different Σ̂ choices on identical real data*. That is a
naturally occurring covariance-assumption contrast: no simulator, real
electronics noise, and D2's `K_rel` for the consequence check. One paragraph
of the paper, but the paragraph that blocks the "it only works in your
simulator" objection.

### 8.2 Claims × arms

| Claim | ORACLE-Cov (mechanism) | ORACLE-Paired (realism) | TIDMAD (real data) |
|---|---|---|---|
| C1 detection at 1 % false-alert | controlled families; D6 power sizing lives here | frozen DynEdge, matched cells | smoke check |
| C2 attribution N/S + stage localization | full control of both families | matched N/S cells; geometry-transfer of the layerwise profile (E5) | — |
| C3 abstention on undeclared families | held-out perturbation families | U1–U4 (τ double-bang, through-going μ, pile-up, correlated noise) | — |
| C4 alarm ranks consequence, conditional on alarm | **designed** dissociation: output-null vs output-aligned, sign predicted; κ sweep | **observational**: conditional-on-alarm AUROC on angular error; Jacobian-projected vs Euclidean monitors | `K_rel` contrast across the two Σ̂ trainings |
| C5 diagnosed-stage repair | activation patching first (agenda §7), then LoRA | replication of the patching check | — |

The bridge that makes this a strong design rather than three demos:
**pre-register, on the Tier-1 results, which monitors will fail on which
ORACLE-Paired families — then run Tier 2 once.** A mechanism that predicts
out-of-sample failures in a testbed it never saw is the referee-proof version
of every claim above.

### 8.3 Where the noise module is used, exhaustively

1. **Trace generator for ORACLE-Cov** — `MultiChannelNoiseGenerator` returning
   implied *and* realized covariance is the Σ̂-versus-Σ instrument itself.
2. **The Σ̂ lever** — `al2o3_athermal` / `OptimalFilter` supply the assumed
   covariance that enters the training objective and the optimal filter.
3. **The waveform N families** — `TemporalNoiseWrapper` (drift,
   non-stationarity), `ArtifactInjector` (lines, glitches, bursts),
   `NonGaussianNoiseGenerator`: controlled perturbations with known statistics.
4. **Referee-proofing** — the `validation` suite certifies that the realized
   ensemble actually has the covariance the metadata claims, with bootstrap
   intervals; report it in the methods.
5. **`K_full` in D2's three-tier consequence metric.**
6. **DELight trace-simulator integration** — collaboration value on its own
   authorship terms; not part of this paper, and not before the preprint.
