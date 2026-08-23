# Paper 3 / ORACLE — Adversarial audit and development plan

_Compiled 17 August 2026. Sources: `latex/paper3_proposal.tex`, `README.md`,
`docs/`, `src/tidmad/`, `src/reconstruction_model/`, and the stale `paper3/`
folder (`REVISION_PLAN.md`, `PERSONAL_RESEARCH_GUIDE.md`, `scripts/`,
`results/`), plus a literature and testbed fact-check against primary sources._

**Read this as a hostile referee report, not as a status update.** Every finding
below is either verified from the file it cites or explicitly labelled as an
unconfirmed suspicion. Section 6 is the constructive half.

---

## 0. Executive summary

Three findings dominate everything else.

**S1 — The Phase-1 joint pilot rests on a false premise.** NuBench's seven
datasets do **not** share common latent events. They are independently simulated
with Prometheus, have different event counts (23.1M / 22.9M / 20.5M / 24.0M /
10.1M / 20.5M / 8.6M), different energy ranges (two datasets span 10–10³ GeV,
the rest 10–10⁵ GeV), and carry no event-ID correspondence or seed record. The
"paired detector geometry" experiment — the agreed first joint deliverable with
Junjie, and the thing gating Phases 2 and 3 — cannot be built from the released
artifacts. `REVISION_PLAN.md` open question #9 asks exactly this and the answer
is no.

**S2 — The TIDMAD score is not a scientific-consequence metric, and its own
authors say so.** The proposal's C4 is "primary on waveforms" precisely because
the consequence variable K is supposed to be externally defined and untunable.
The TIDMAD paper states its denoising score "lacks direct relevance to
fundamental science" and introduces a second benchmark (the axion exclusion
limit) *because* of that. Separately: the score is only computable on validation
files (the 208 science files have no injected ground truth, CH1 only), the
project's own frozen band range (328 bins, 0–800 kHz) excludes injections the
P1 survey found in files 0016–0019, and `score.py --in-band-only` therefore lets
the author choose which files enter K. C4's decisive testbed is the weakest link
in the design, not the strongest.

**S3 — The three headline claims each have a directly competing published
paper, and none of them is cited.** Claim 1 (frozen representations separate
corrupted acquisition from rare-but-clean physics) is Roschewitz et al.,
MICCAI 2025, arXiv:2411.07940 — same goal, same frozen-encoder design, published
shift-type decision pipeline. Claim 3 (diagnosed-stage adaptation beats
wrong-stage adaptation, with shift type determining the stage) is the headline
result of Lee et al., *Surgical Fine-Tuning*, ICLR 2023, arXiv:2210.11466.
Claim 2's motivating premise ("alarms decouple from harm") is the thesis of
Podkopaev & Ramdas, ICLR 2022, arXiv:2110.06177, and the malignancy analysis in
Rabanser et al., *Failing Loudly*, NeurIPS 2019. The proposal's 8-item
bibliography contains none of these. A referee will find all four in the first
hour.

Beneath these: the TIDMAD arm has two blocking inference bugs that would silently
produce a garbage benchmark score, one plan item marked done is demonstrably not
done, and the repository currently contains **zero** diagnostic code — no hooks,
no displacement metric, no MMD, no conformal abstention, no LoRA — despite
`README.md` and `docs/tidmad.md` describing that comparison in present tense.

---

## 1. Structural problems with the scientific design

### P1.1 — N and S are not separable in principle on the chosen interventions (BLOCKER-class)

The proposal's own N enumeration includes "hit thinning, and channel or module
loss". Its S definition is events "that occupy low-density regions of the clean
training support", with strata "in energy, topology, vertex/containment, and
**multiplicity**".

Module dropout applied to a high-multiplicity event *produces a low-multiplicity
event*. The N intervention manufactures examples that lie, by construction, in
the declared S region. Any classifier that then separates N from S is not
learning "corrupted versus rare"; it is learning the residual fingerprint of one
specific corruption operator (spatially clustered pulse removal at exact sensor
positions) against a naturally-occurring multiplicity distribution.

This is not a nuance the identifiability paragraph covers. §1's limitation is
about covariance decomposition (`Σ = Σ₁ + Σ₂` admits infinitely many PSD splits)
— true, but a non-sequitur here, since the attribution task never asks for a
covariance split. The real limit is that the N/S label is a property of the
*generating process*, not of the event, and the study's positive results will be
operator-fingerprint detection unless the design forces the two classes to be
matched in observable event content.

**What a referee says:** "You have shown your classifier can recognise your own
perturbation function."

**Minimum fix:** for every N cell, construct a *content-matched* S cell — real
events drawn from the training distribution with the same observed multiplicity,
charge, and time spread as the thinned events. If the diagnostic still separates
them, the claim survives. If it does not, that is the paper's most interesting
result and should be reported as an identifiability boundary.

### P1.2 — ρ(A, K) is mechanically positive under any severity-graded family

Both the alarm magnitude A and the consequence K are monotone in intervention
severity by construction. The feasibility pilot already demonstrates this:
across 0→50% dropout, median angular error rises 15.28°→25.12° *and*
standardized displacement rises 0.000→0.149 *and* 10-NN retention falls
1.000→0.393. All three axes move together because severity moves them all.

A positive ρ(A, K) pooled over severities therefore measures nothing. The
conditional-on-alarm framing in C4 helps but does not fix it, because the strong-
alarm cells are predominantly the high-severity cells.

The `REVISION_PLAN.md` already contains the correct instrument — the
**output-null perturbation** control (a shift approximately orthogonal to the
local output Jacobian: high A, low K) — but this is **absent from
`paper3_proposal.tex` entirely**. The proposal as written cannot deliver C4.

**Minimum fix:** C4 must be evaluated *within* severity strata, and the design
must include at least one family that dissociates A from K by construction. See
§6.2 — this is also the strongest available development direction.

### P1.3 — "Layerwise" is not instrumentable at meaningful depth on either testbed

The whole novelty rests on "the full layerwise signature" being the fifth
monitoring arm, and on "the layerwise sensitivity profile" and "the depth at
which a consequential change first becomes visible" as co-primary mechanism
endpoints.

- On NuBench, the proposal commits to exactly one hook: "a 128-dimensional
  pre-head event representation". One point is not a profile.
- On TIDMAD, there are exactly two stage boundaries (`_temporal_stage`,
  `_band_stage`). Two points is not a profile either, and the pooled third hook
  T2.1 asks for **does not exist** — `TidmadTransformer.forward` never pools
  (the base class does, at `reconstruction_model/model.py:265`, but the TIDMAD
  subclass overrides it away).

So "the full layerwise arm" versus "final embedding only" is, on NuBench, a
comparison of one representation against itself.

**Minimum fix:** either instrument all four `DynEdgeConv` block outputs on
NuBench (they are reachable via `register_forward_hook`, no model edit needed)
and add a real pooled hook to TIDMAD, or delete every "layerwise" and
"localization" claim and reposition the paper around consequence. Do not ship
the current version — it is the single easiest thing for a referee to falsify.

### P1.4 — C1's thresholds are asserted, not powered

"At a 1% false-alert rate … detects at least 80% … within five windows, and is
non-inferior to the strongest eligible baseline by a five-point power margin."
There is no sample-size calculation anywhere in the proposal or the plan
supporting an 80% detection target or a 5-point non-inferiority margin, and the
only empirical data point available — the pilot's bootstrap CIs — is
discouraging: 0% dropout gives 15.28° (12.83–18.71) and 10% gives 17.95°
(13.89–22.60). Those intervals overlap almost completely at *n*=256.

A preregistered threshold without a power analysis is not preregistration; it is
a number that will be quietly revised. Compute the required windows-per-cell
before freezing, and state it.

### P1.5 — Two incompatible claim ladders are in circulation

`paper3_proposal.tex` has C1 detection / C2 attribution+abstention / C3 **cost**
/ C4 consequence. `REVISION_PLAN.md` has C1 detection / C2 attribution+
localization / C3 **abstention** / C4 consequence / C5 repair. They also
disagree on E: the proposal says E "is reported as a gate result, **not as a
class in a classification score**"; the plan says C2 "distinguishes declared N,
S **and E** provenance classes". A collaborator reading both will implement the
wrong thing. Pick one numbering and delete the other document's version.

### P1.6 — The identifiability disclaimer is doing less work than it appears to

"All attribution here is over declared intervention families under a stated
simulator" is honest, but combined with P1.1 it collapses the contribution: if
attribution only holds over families you generated, and your N family
manufactures S-like events, then the paper's operational recommendation
("discarding S rejects physically valid events, whereas rejecting N removes
measurements that violate the acquisition contract") is not supported by
anything the study can measure. State plainly what an experimenter is supposed
to do with the result.

---

## 2. Testbed problems (verified against primary sources)

### P2.1 — NuBench geometries are not event-paired — **CONTRADICTED**

Verified from arXiv:2511.13111 v2, the NuBench repo, and the released artifact
listing. Consequences:

- Phase 1 (B, paired detector geometry) as written is not executable.
- `REVISION_PLAN.md` open question #9 is answered: no.
- The closest available contrast is **Hexagon (water) vs Hexagon Ice LE (ice)** —
  same geometry, different medium — but even that is 20.5M vs 8.6M events over
  different energy ranges, so it is a *distribution-level* comparison only. The
  plan's own validity gate ("a geometry response is not causal evidence without
  valid pairing") then forbids the causal claim.
- The only route to true event-level pairing is **re-simulating with Prometheus**
  yourself. Prometheus is open source. See §6.4 — this is a genuine opportunity,
  not just a cost.

Also worth recording, all verified from the NuBench paper:

- **Test partitions have noise-induced pulses removed; training partitions do
  not.** This is a train/test acquisition mismatch that directly interacts with
  an N-family defined as hit thinning and module loss.
- "The uncertainties do not contain effects stemming from stochasticity in the
  training procedure" — no seed-variance band on any published number.
- "PMT effects such as charge saturation are not modeled."
- The injection bias produces "a high rate of starting events" — a non-physical
  topology mix, which matters because topology is one of the declared S strata.
- The released artifacts have **no LICENSE file, no DOI, no version pinning** —
  they are `sid.erda.dk/share_redirect/<token>` tarballs. For a paper whose
  reproducibility discipline is a selling point, that is a real exposure.

The JINST publication is real and should be cited precisely: *JINST* **21**
(2026) 05, T05001, DOI `10.1088/1748-0221/21/05/T05001` — note it is a
**Technical Report ("T") article**, CC BY 4.0.

### P2.2 — TIDMAD's score cannot carry the "external scientific consequence" claim

Verified from the TIDMAD paper (arXiv:2406.04378, NeurIPS 2025 D&B, Spotlight)
and repo:

1. The authors state Benchmark 1 (the denoising score, log₅.₂₇Λ) "lacks direct
   relevance to fundamental science" and add Benchmark 2 (the exclusion limit)
   for that reason. Using Benchmark 1 as K adopts the metric its authors
   disclaim.
2. **Science files contain no injected signal** — CH1 only, no ground truth.
   `docs/tidmad.md` correctly uses them as noise-only for J(f), but any plan to
   compute K on science data is unworkable without the full Brazil-band pipeline
   (208 files, ~600 GB, `brazilband.py` + an external `AxionLimits` notebook).
3. K-denoising and K-physics are therefore measured on **disjoint data**.
4. The physics metric is **saturated**: the paper reports 1–2 orders of magnitude
   denoising improvement that nonetheless failed to exceed the prior ABRA Run 3
   limit. The downstream consequence variable is insensitive exactly where you
   need discrimination.
5. Fine vs Coarse score are not interchangeable (raw data scores 1.00 fine,
   1.10 coarse). K must specify which.
6. Score depends on injection amplitude (a 1/5-amplitude weak-signal variant
   exists, files 0020–0039), so K is not one number without fixing the variant.
7. Documented data defect: channel lengths differ in a few train/validation
   files; upstream recommends truncating to the first 2,000,000,000 samples.

And the project's own choice compounds it: `TidmadSTFTConfig.n_bands_used = 328`
covers 0–~800 kHz, and `config.py:19-21` records that "The P1 survey found
injections above that edge in files 0016--0019". `score.py --in-band-only` then
selects which files enter the score via `P1_injection_survey.json`. The score
function is external; **the sample it is computed on is a researcher degree of
freedom.** That must be declared in the paper, not handled by a flag.

### P2.3 — Panda is cheap; SPINE is not. The plan has this backwards.

Verified: Panda (arXiv:2512.01324) ships four **public** pretrained checkpoints
via `panda.load("base"|"particle"|"interaction"|"semantic")`, auto-downloads
PILArNet-M (~168 GB) from Hugging Face, and is pip-installable. It benchmarks
*against* SPINE.

SPINE, by contrast, requires a ~25 GB container, ROOT + LArCV2 + MinkowskiEngine
built from source, and its own README says `pip install spine` is "only for
post-processing/analysis/viz/light dev" — not for training or full-chain
inference. Its reconstructed HDF5 and physics datasets are marked `TODO` in the
2026 workshop repo, i.e. not yet released.

The plan makes SPINE the Phase-2 target and Panda an optional Stage 5. On
infrastructure cost alone that ordering should be reversed. Also: Graph-SPICE
(arXiv:2007.03083) is **preprint-only**, no journal DOI — do not cite it as
refereed.

### P2.4 — Douglas et al. is still an unrefereed preprint

Verified via INSPIRE-HEP: arXiv:2411.09864 remains v3 (4 Feb 2025) with no
`publication_info` and no DOI, unchanged for ~18 months. The proposal's current
citation ("preprint (2025)") is correct. Keep it that way.

---

## 3. Prior-art / novelty problems

The proposal's bibliography cites MMD, C2ST, conformal, selective classification
and calibration **as baselines**. It cites none of the work that already
attempts the actual task. Ranked by damage:

| # | Paper | Threatens | Why it hurts |
|---|---|---|---|
| 1 | Roschewitz et al., *Automatic dataset shift identification to support safe deployment of medical imaging AI*, MICCAI 2025 ([2411.07940](https://arxiv.org/abs/2411.07940)) | Claim 1 | Unsupervised, **frozen-encoder**, explicitly classifies *which type* of shift occurred (prevalence vs covariate vs mixed) via BBSD + MMD on frozen features. Same goal, same architectural choice. |
| 2 | Lee et al., *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts*, ICLR 2023 ([2210.11466](https://arxiv.org/abs/2210.11466)) | Claim 3 / C5 | Headline result: **"the type of distribution shift influences which subset [of layers] is more effective to tune"** — input corruptions → early layers, feature-level → middle, label shift → last. Across 7 datasets, with an automatic gradient-norm criterion for picking the block. Your diagnosed-vs-wrong-stage LoRA experiment is structurally this, with LoRA swapped for full-block tuning. |
| 3 | Podkopaev & Ramdas, *Tracking the risk of a deployed model and detecting harmful distribution shifts*, ICLR 2022 ([2110.06177](https://arxiv.org/abs/2110.06177)) | Claim 2 | "Detect harmful shifts while ignoring benign ones" is the paper's thesis, four years earlier. |
| 4 | Zhang et al., *"Why did the Model Fail?" Attributing Model Performance Changes to Distribution Shifts*, ICML 2023 ([2210.10769](https://arxiv.org/abs/2210.10769)) | Claims 1+2 jointly | Shapley attribution of a *performance change* to specific shift mechanisms — already couples attribution to consequence. |
| 5 | Hase et al., *Does Localization Inform Editing?*, NeurIPS 2023 ([2301.04213](https://arxiv.org/abs/2301.04213)) | C5's **validation logic** | Shows causal-tracing localization does **not** predict where editing works. Published evidence that "LoRA at the diagnosed stage works better" is a weak validator of a localization claim — a negative there would not falsify the diagnosis, and a positive would not confirm it. |

Also unavoidable and uncited:

- **Rabanser, Günnemann & Lipton, *Failing Loudly*, NeurIPS 2019**
  ([1810.11953](https://arxiv.org/abs/1810.11953)) — the canonical shift-detection
  benchmark, and it already pairs detection power against whether accuracy
  degraded. This is the first paper a referee names.
- **Castro, Walker & Glocker, *Causality matters in medical imaging*, Nat.
  Commun. 2020** ([1912.08142](https://arxiv.org/abs/1912.08142)) — establishes
  the **acquisition shift vs manifestation/population shift** taxonomy that N/S
  reproduces in detector language. Cite it or be told you borrowed it.
- **Yang, Zhou & Liu, *Full-Spectrum OOD Detection*, IJCV 2023**
  ([2204.05306](https://arxiv.org/abs/2204.05306)) — covariate-OOD (tolerate) vs
  semantic-OOD (reject) with a low/high feature split.
- **Lambert et al., *Multi-layer Aggregation as a Key to Feature-Based OOD
  Detection*, MICCAI-UNSURE 2023** ([2307.15647](https://arxiv.org/abs/2307.15647))
  — finds single-layer OOD detectors "have inconsistent behaviour depending on
  the type of anomaly". That is the empirical core of the localization claim,
  already published.
- **Slice discovery** — Domino ([2203.14960](https://arxiv.org/abs/2203.14960)),
  Spotlight ([2107.00758](https://arxiv.org/abs/2107.00758)) — the S branch alone
  is largely solved in frozen representation space.
- **ATC** ([2201.04234](https://arxiv.org/abs/2201.04234)) and **DoC**
  ([2107.03315](https://arxiv.org/abs/2107.03315)) — confidence-derived scalars
  already predict OOD accuracy, i.e. "alarm magnitude predicts consequence" is
  largely answered affirmatively for classification.
- **Bayram et al., *From concept drift to model degradation*, KBS 2022** — an
  entire survey of performance-aware drift detectors. Makes "nobody asks whether
  the drift matters" unsayable.
- **Conformal outlier machinery** — Bates et al., *Ann. Statist.* 2023
  ([2104.08279](https://arxiv.org/abs/2104.08279)); iDECODe, AAAI 2022
  ([2201.02331](https://arxiv.org/abs/2201.02331)). Claim only the *use*, not the
  method.
- **HEP context you must position against**: CMS ML data-quality monitoring
  ([2501.13789](https://arxiv.org/abs/2501.13789), ECAL autoencoders
  [2309.10157](https://arxiv.org/abs/2309.10157),
  [2407.20278](https://arxiv.org/abs/2407.20278)) — production-grade
  acquisition-violation detectors. Expect: *"why not just use DQM?"* Also
  systematics-aware training ([2105.08742](https://arxiv.org/abs/2105.08742))
  invites *"why not train the robustness in?"*

**What honestly survives:**

1. No prior work monitors a *learned particle-reconstruction* model's internals
   for N-vs-S, and none exists for neutrino telescopes. Real, but a
   domain-transfer contribution.
2. The **conditional-on-alarm consequence AUROC** and the explicit 2×2
   alarm×consequence contingency: I found no prior instance. Genuine but modest
   — frame it as an evaluation protocol refinement of harmful-shift detection.
3. The layerwise profile as a *diagnostic output* rather than a detection
   ingredient. Defensible, but Lambert + Surgical Fine-Tuning together already
   imply the result.
4. Integration novelty for the full N/S/abstain → localize → targeted-LoRA
   pipeline. Referees discount this heavily unless the integration produces
   something none of the parts predict.

**Recommendation:** drop the word "novel" from claims 1 and 3. Reposition as
"we port harmful-shift detection and shift-type identification to learned
physics reconstruction, where the consequence variable is a *physics observable*
rather than classification accuracy". And pre-empt Hase et al. explicitly in the
C5 design.

---

## 4. Code and repository problems

Everything below was read in the working tree. Severity is BLOCKER / MAJOR /
MINOR.

### BLOCKER

**C1 — `infer.py` writes a `+128`-shifted series into an `int8` dataset; every
denoised sample saturates at 127.**
`data.py:109` returns `squid + 128.0, ref + 128.0`. `infer.py:92-94`
un-standardises back into that shifted frame and hands the result to
`_write_denoised_channel`, which resolves `target_dtype` from the **source
dataset** (`_vendor/score.py:83`) — `int8` per the release contract — and then
`np.clip(np.rint(values), -128, 127)`. A series centred near +128 clips to a
constant 127. The benchmark would then score a DC-saturated array.
The bug is confirmed by asymmetry: `_copy_reference_channel` (`infer.py:53`)
reads channel0002 raw with **no** `+128` and writes it unshifted, so the two
channels in the output file live in different coordinate frames.
*Fix:* subtract `cfg.sample_shift` before writing, and assert the written range.

**C2 — `infer.py` writes uninitialised memory for the file tail.**
`infer.py:83` allocates `np.empty(n_samples)`; `infer.py:86` loops
`while pos + n <= n_samples`. With `n = 198_656` and ~2×10⁹ samples per file, up
to ~198k samples per file are never assigned and are written straight from
`np.empty` into the scored HDF5.

**C3 — `chi2_of` is still a silent alias, and both plan documents say it was
removed.**
`train.py:115-117`:
```python
def objective_key(run):
    """Which loss the arm optimises. ``chi2_of`` trains on the chi2 objective."""
    return "chi2" if run.train.loss == "chi2_of" else run.train.loss
```
`loss.reconstruction_losses` returns only `{"mse","chi2"}`; there is no
optimal-filter code in the tree. Because every other field including `seed` is
identical, `t_chi2_of.yaml` produces a run *bit-identical* to `t_chi2.yaml`.
Meanwhile `PERSONAL_RESEARCH_GUIDE.md:575` says `[x] T1.3 … Do not ship a silent
alias` and `REVISION_PLAN.md:462` says `[x] Remove the unimplemented chi2_of
alias`. Worse, `tests/test_train_loop.py:112` *asserts* the alias is correct, so
the test suite blocks the fix.

### MAJOR

**C4 — `assert_configs_differ_only_in_loss` is never called by `main()`.**
`train.py:5` claims the single-difference property is "asserted by" it. Its only
callers are in `tests/test_config.py`. It is a decorative gate. It is also blind
to three things that change the experiment and are CLI-only: `--data-dir`
(config's `data_root` is dead), `--steps` (mutates `num_steps` at
`train.py:319-322` *after* load), and `--resume`.

**C5 — T1.1 half-done: `num_workers` is declared and read by nothing; there is
no `DataLoader` anywhere in the repo.** A tree-wide grep for `DataLoader` returns
only the declaration at `config.py:88`. The data path is a bare generator
(`train.py:273-298`). `device_batch_size` *is* honoured, but via gradient
accumulation over B=1 forward passes, not batching. `docs/tidmad.md:18` claims
"batched training and held-out validation through a PyTorch `DataLoader`" —
false. `data_root` and the effective `eval_num_windows` behaviour have the same
declared-vs-executed gap (evals always score the same first 32 validation
windows).

**C6 — T1.2 not done: no epoch loop, no reachability assertion, and an
interrupted run is indistinguishable from a completed one.** `_stream(..., loop=
True)` cycles forever; nothing relates `num_steps × device_batch_size` to
`len(fit_positions)`. On SIGTERM the run stops early, still writes `latest.pt`,
and exits 0 — while `write_run_config` (called *before* training) has already
archived `num_steps: 25000`. `run_config.json` can therefore claim a 25k-step run
for a 3k-step checkpoint. That is a provenance defect on the primary artifact of
a study whose selling point is preregistration.

**C7 — The frozen PSD fields have no effect on training.** `FROZEN` declares
`psd_window="boxcar"`, `psd_average="median"`. `estimate_band_psd` computes
`j_band` from `stft(w, cfg)` — i.e. **Hann**, `config.py:42` — and passes the
boxcar/median arguments only to `welch_psd_from_windows`, whose result
`train.py:328` binds to `_` and discards. Changing `psd_window` in the frozen
config changes nothing about the arms. Either wire it or delete it; a frozen
config with inert fields is not a frozen config.

**C8 — The χ² weight has no numerical floor, and the repo vendors the floor it
does not use.** `loss.py:69-70` only checks `j > 0`; near-zero `J` is unguarded.
`_vendor/noise.py:15` implements exactly the right primitive
(`floor = max(median(positive) * floor_fraction, tiny)`) and
`inverse_psd_weights(..., zero_dc=True)` — neither is called. The DC band is the
concrete hazard, and the codebase knows it: `psd.py:105-107` explicitly excludes
band 0 from the whitening check as "degenerate", while `loss.py` divides by that
same value at full weight.

**C9 — PSD calibration and the training residual live in different coordinate
frames.** `_noise_only_windows` → `_vendor/loader.py` reads raw int8 with **no**
`+128`; training/eval windows come from `read_channel_pair`, which adds `+128`.
So `J(f)` is estimated in one frame and divides a residual computed in another.
Confined to the DC bin, but combined with C8 that is precisely the term with no
floor. Nothing tests this.

**C10 — `infer.py` crashes on its own default device, and T1.4 is not done.**
`infer.py:92-94` re-implements `loss.per_row_stats`/`loss.unstandardise` inline
(the plan marks this `[x]`). Beyond the mismatch it is a live bug: `out_std` is
float32 on `device`, `std_in`/`mean_in` are float64 on CPU, so with
`--device cuda` — the default — this raises a device-mismatch `RuntimeError`.

**C11 — `nubench_reference_metrics.py` groups by `is_track` but labels the
result CC/NC.** `scripts/nubench_reference_metrics.py:47-63` does
`.group_by("is_track")` and then emits `"cc_track"` / `"nc_cascade"`. `is_track`
is a *topology* flag; NuBench Table 6's `nc_cascade` is an *interaction* class.
νe-CC events are cascades but are CC — they will be pooled into the "NC" bucket.
The pilot script proves the right field is available in the same artifacts
(`nubench_smoke_pilot.py:48-54` reads `interaction`). This is a strong candidate
for the unexplained 0.055° / 0.051 pp V0-A gate failure that `RESULT.md:18-22`
escalates to a NuBench author. **Verified as a code-level mismatch; the causal
attribution to the 0.055° gap is a suspicion I could not confirm without the
parquet.** Re-run grouped on `interaction` before emailing anyone.

**C12 — The pilot's 256-event subset is drawn from the head of the parquet, not
at random.** `nubench_smoke_pilot.py:154-166` iterates batches and breaks as soon
as ~6400 of each class are collected, so the candidate pool is the **first
~6400 rows in file order** out of 2,989,339; `pool.sample(random_state=seed)`
randomises only within that prefix. If the parquet is ordered by run/event_no —
the normal case — the pilot is one contiguous slice of the test set. `RESULT.md`
discloses the ≥20-module enrichment but not this.

**C13 — The ≥20-module enrichment interacts with the severity axis, and the
stated justification is not the operative one.** Two thresholds never
reconciled: `minimum_modules: int = 9` (dataset floor, `:103`, used in
`n_drop = min(requested, max(0, n_modules - self._minimum_modules))` at `:121`)
and `--minimum-observed-modules` default 20 (selection, `:319`), which
`make_dataset` never forwards. The `n-9` cap binds only when `n < 18`; every
selected event has `n ≥ 20`, so the cap never binds for the sample and nominal
severity equals realised severity. On the unrestricted population, low-
multiplicity events would have realised dropout silently compressed toward
`(n-9)/n`, flattening the axis. So the enrichment's real function is to make the
severity axis clean — not, as `RESULT.md:46` says, "so 50% dropout leaves the
8-NN graph defined" (the cap already guarantees that at any multiplicity). The
monotone response is therefore partly a property of the selected stratum.

**C14 — Row alignment across severities is assumed, never checked.**
`nubench_smoke_pilot.py:379-384` does a strictly row-paired subtraction between
two separate `infer()` runs over two separate dataset instances, and never
compares `event_no` between them. The script carefully realigns the *released*
predictions by merge but not the clean-vs-perturbed pair. A one-line assert
closes it.

**C15 — Bootstrap CIs are unpaired in an explicitly paired design.** The
resampling *level* is correct (one value per event, resampling events). But a
single `rng` is threaded through the severity loop, so each severity gets an
independent resample. `RESULT.md:50-55` then prints four overlapping marginal
CIs side by side, inviting a paired inference the intervals do not license. The
CI that supports the claim is the CI of the paired *difference* on common
indices; none is computed. Separately `knn10_retention` is emitted as a bare
scalar with no uncertainty while its two table neighbours have error bars.

**C16 — Dropped-module sets are not nested across severities.**
`nubench_smoke_pilot.py:126-130` folds `severity_key` into the seed, so the 10 /
25 / 50% draws are *independent* subsets — an event can lose module A at 10% and
keep it at 25%. This injects perturbation-identity noise into the severity axis
that a nested design (shuffle once, take prefixes) removes at zero cost.
`RESULT.md:48`'s "paired across severities" is true at the event level only.

**C17 — The released checkpoint's preprocessing is replaced wholesale and never
validated.** `patch_released_detector` (`:57-92`) overwrites **five** methods of
the unpickled detector, and `:336-344` supplies three attributes by guess,
including `_add_static_features = True`, which changes the node feature vector
fed to a frozen checkpoint. The stated justification — that Python 3.10 bytecode
loaded under 3.11 "preserves the constants but corrupts its operators" — is not
a credible mechanism as written (cross-version bytecode normally fails to
unmarshal). The script contains no check that the patched pipeline reproduces
the released predictions, yet the open blocker is precisely a 0.944° / 7.708°
p95 disagreement with those predictions — and `RESULT.md:37-39` lists k-NN tie
handling, version drift and provenance as candidates while **omitting the
author's own five-method substitution**.

*Fact-check result on that blocker:* there is **no** GraphNeT issue or
documentation about k-NN determinism — do not cite one. The CPU/CUDA divergence
is real at source level (`torch_cluster` CPU uses a nanoflann KD-tree; CUDA uses
brute force with strict-`>` tie-breaking that resolves ties to the lowest index
and can return *fewer* edges — see [pyg#4906](https://github.com/pyg-team/pytorch_geometric/issues/4906)),
and DynEdge amplifies it because `DynEdgeConv.forward` **recomputes the kNN
graph in latent space every layer**. But a *median* shift of 0.9° is a systematic
effect, and tie-breaking should produce a heavy-tailed, roughly symmetric
scatter with a robust median. Rank the candidates:
1. the five-method preprocessing substitution and the guessed
   `_add_static_features` (C17);
2. standardisation constants / node definition from the released config bundle
   rather than reconstructed by hand;
3. the NuBench train/test noise-pulse asymmetry;
4. **[GraphNeT issue #880](https://github.com/graphnet-team/graphnet/issues/880)**
   (open, 30 Apr 2026) — `predict_as_dataframe` iterates the dataloader twice and
   the passes can **silently misalign**, verbatim: *"downstream analysis just
   produces wrong physics"* because *"the resulting DataFrame still looks
   valid."* Triggers include multi-worker loading with re-seeding. This is the
   strongest *documented* candidate;
5. only then kNN tie-breaking.

*Decisive cheap test:* run the identical re-inference twice, CPU and GPU, same
checkpoint, same preprocessing. If CPU-vs-GPU alone reproduces ≈0.9°, it is the
backend. If they agree with each other but both differ from the parquets, it is
preprocessing / alignment / version skew.

**C18 — `set_seed` does not make the CUDA path deterministic.**
`seeding.py:16-23` sets `random`, `numpy`, `torch`, `torch.cuda` — but not
`cudnn.deterministic`, `cudnn.benchmark = False`, `use_deterministic_algorithms`,
or `PYTHONHASHSEED`. `--device` defaults to cuda, and the model uses
`F.scaled_dot_product_attention`, whose backward uses non-deterministic atomics.
Initialisation *is* identical across arms; the trajectories are not bitwise
paired on GPU.

**C19 — No shuffling at all.** `_stream(fit_positions, run, loop=True)` iterates
in file/offset order forever, with no permutation and no reshuffle. Each
accumulation group of 4 windows is 4 temporally adjacent windows of a continuous
stream — maximally correlated gradients — and every pass is the identical
sequence. Defensible for a paired-arm contrast; not declared anywhere.

**C20 — `docs/reconstruction_model.md` documents a different repository.** It
says "`src/reconstruction_model/` … ha[s] been removed after [its] useful content
was migrated" — while `src/reconstruction_model/` is the live package that
`tidmad/model.py:20` and `tidmad/train.py:26,28` import from. It is a verbatim
README from an upstream repo telling a new collaborator that the package they
must import has been deleted. Junjie will read this.

### MINOR

- **C21 — `docs/tidmad.md` misstates five implemented features and both CLI
  checkpoint paths.** Claims a DataLoader (no), epochs (no), "model-only and
  resumable checkpoints" (only the combined one exists), and "a fail-fast check
  of the two-channel HDF5 output contract" (nothing verifies the output file —
  this is plan item T0.6, marked `[x]`, **not done**). It documents
  `checkpoints/reconstruction_model_20.pt` and `reconstruction_resume_<step>.pt`;
  `train.py:184,198` writes `checkpoint_00000020.pt` and `latest.pt`. The
  documented commands cannot run. It also says "From `transformer/`" — a
  directory that does not exist in this layout.
- **C22 — `whitening_identity_error` is called from no production path**, only
  `test_psd.py:33`, and its only threshold is a bare `assert err < 0.25` in that
  test. Worse, the test is vacuous: `x` is drawn once *outside* the list
  comprehension, so all 64 "windows" are the identical array and calibration and
  validation split the same waveform. The threshold can never fail. (T1.6 is
  correctly still `[ ]`.)
- **C23 — Model/registry duplication.** `reconstruction_model/model.py` and
  `models/current_compact.py` differ by 6 lines (a `__future__` import and the
  Muon import path), so the repo expects **both** `reconstruction_model/muon.py`
  and `reconstruction_model/models/muon.py`. `tidmad/registry.py` is a
  byte-identical fork of `models/registry.py` with one entry. In the original,
  5 of 8 registry entries reference modules absent from the tree and would
  `ImportError`.
- **C24 — `pyproject.toml` under-specifies.** `dependencies = []` while the code
  needs torch, numpy, h5py, jaxtyping, PyYAML, scipy. `README.md:101` claims
  `torch==2.5.1+cu124` is pinned; no such pin is in the root file.
- **C25 — The "~0.6M params dominated by 656×128 band embedding" rationale is
  arithmetically wrong.** The band positional embedding is dimensioned to the
  **328 physical** bands (`config.py:25-26`), i.e. 328×128 ≈ 42k. The two
  transformer layers dominate: each ≈ 4·128² + 2·128·512 ≈ 197k, so ≈ 394k of
  ~0.6M. The total is right; the stated reason is not. It is in
  `PERSONAL_RESEARCH_GUIDE.md` and will be repeated into the paper.
- **C26 — Dead code.** `train.py` imports `random`, `sys`, `numpy` — zero uses.
  `infer.py` imports `sys` and `FROZEN` — unused. `_checkpoint` round-trips
  through disk (`torch.save(torch.load(path), latest)`) where `shutil.copyfile`
  would do, and uses `weights_only=False` on both save paths.
- **C27 — `loss.py:30` `per_row_stats(x, eps=1e-6)` never uses `eps`.** The
  caller passes it and it is discarded; the round-trip is consistent only because
  `reconstruction_model/model.py:31` happens to use the same literal.
- **C28 — `nubench_reference_metrics.py`**: `prediction_norm` divides with no
  zero guard (a null `angle` is counted by `pl.len()` but skipped by the rate
  means, so `n` and the denominators can silently disagree); and
  `"tolerance_declared_before_run"` is a literal embedded in the script that
  computes the comparison — there is no external pre-registration record backing
  the word "before".

### What is genuinely done — no finding

Credit where due, because it matters for the migration plan:

- `FROZEN` is a real frozen dataclass, serialised into every run. The fields
  that *are* consumed (`device_batch_size` via accumulation, `num_steps`,
  `warmup_steps`, `eval_period`, `checkpoint_period`, `grad_clip`, `seed`) match
  execution. The gaps are confined to `num_workers`, `data_root`, `psd_window`
  and the epoch claim.
- T0.1 / T0.2 done: `_vendor/` holds the five pinned modules with the source
  commit recorded (`89a4a063…`); no `sys.path` injection remains.
- T0.3 / T0.4 done, and done well. `_checkpoint`/`_restore` round-trip model,
  both optimisers, both schedulers and the step counter, and
  `test_train_loop.py:44` is a genuine behavioural test (perturb every parameter
  → restore into a fresh model → per-tensor `allclose` + `get_last_lr()`
  equality). `chronological_split` implements a real guard gap and `evaluate` is
  a true held-out pass reporting both metrics for both arms.
- **The score really is external.** `_vendor/score.py:153-160` is a genuine
  `subprocess.run([python, "benchmark.py", ...])` with verbatim stdout capture,
  and `:132-139` refuses to run without the upstream script, stating "This module
  deliberately has no fallback scorer". There is no reimplementation. The
  "untunable metric" claim holds *for the metric function* — subject to the
  sample-selection caveat in P2.2.
- **The de-standardisation argument is correct**, and `test_loss.py:31` is a
  genuine behavioural test of it (equal-energy residuals in low- and high-noise
  bands, asserting `c_low > 10·c_high` and non-proportionality). The χ² weighting
  is genuinely inverse-PSD in form. The defects are the *estimator* (C7), the
  *floor* (C8) and the *frame* (C9), not the objective.
- **The two-stage hook points exist without model edits.** `_temporal_stage`
  (`model.py:107`) and `_band_stage` (`model.py:128`) are real named methods,
  both hookable. T2.1's "no model edits required" is accurate — except for the
  pooled hook, which has no anchor (see P1.3).
- **Module dropout drops modules, not pulses**, deterministically and
  reproducibly (`:116-131` groups by unique `(x,y,z)` and removes every pulse on
  chosen sensors). The weaknesses are non-nesting and threshold decoupling, not
  the mechanism.
- **T2/T3/T4 are honestly marked incomplete** in both plan documents — and they
  are. A repo-wide grep for
  `register_forward_hook|knn|principal angle|subspace|mmd|two-sample|conformal|abstention|lora|displacement|retention|diagnos|monitor`
  across every ORACLE `.py` returns **zero hits**. The only working
  implementations of displacement and kNN retention live in
  `paper3/scripts/nubench_smoke_pilot.py` and are not shared with the TIDMAD arm.
  No false claim in the checklists — but `README.md:29-32` and
  `docs/tidmad.md:9-12` describe the MMD / C2ST / abstention comparison in
  present-tense framing a skimming collaborator will read as implemented.

### Test-suite assessment

20 tests, 6 files. Real behavioural tests: checkpoint round-trip, loss
non-proportionality, chronological-split disjointness, STFT round-trip with
numeric bounds, config-gate behaviour including a negative case, PSD leakage
audit. Shape-only smoke: four of six `test_model.py` tests (and those four
construct `TidmadTransformer` **without** `init_weights()`, leaving
`band_pos_embd.pos_embed` as raw `torch.empty` — they pass only because they
never inspect a value). Actively harmful: `test_train_loop.py:108` pins the C3
alias as correct; `test_psd.py:24,37` are vacuous.

Nothing tests that `J` is the estimator claimed, that the loss is numerically
safe, that PSD and residual share a frame, that the declared budget is
reachable, that the config gate runs in production, or that inference produces a
valid file.

---

## 5. Migration: `paper3/` → `ORACLE/`

`paper3/` is the stale folder; `ORACLE` is the working repo. Current state of
the split:

| Item | Where it is | Verdict |
|---|---|---|
| `REVISION_PLAN.md` | paper3 only | **Migrate**, after reconciling the C1–C5 ladder with the LaTeX (P1.5). This is the canonical shared plan Junjie needs. |
| `PERSONAL_RESEARCH_GUIDE.md` | paper3 only | **Keep private, do not migrate** — it is explicitly a private record and contains meeting-strategy content. But its T0–T5 checklist is the only honest status record; extract the checklist into ORACLE and mark the three items that are wrong (T0.6, T1.3, T1.4). |
| `scripts/nubench_*.py` | paper3 only | **Migrate to `ORACLE/scripts/nubench/`** — `README.md` currently documents commands for scripts that are not in the repo, which is a broken onboarding path. Fix C11–C17 during the move, not after. |
| `results/nubench_hexagon_ice_le_dynedge/` | paper3 only | **Migrate to `ORACLE/results/`** with `RESULT.md` amended for C11–C16 (the CC/NC mislabel and the head-of-file sampling change what those numbers mean). |
| `2512.01324v1.pdf` (Panda) | paper3 only | Leave out of git; record the arXiv ID in `references.bib`. |
| `sources/2026-08-12_*.md` | paper3 only | **Migrate** — these are the provenance records for the citation decisions. Add a third for the NuBench pairing finding (P2.1). |
| Junjie's annotated PDF | not committed anywhere | Keep out of git (third-party comments), but record the 12 comments + responses in a committed `docs/REVIEW_LOG.md` so the reasoning survives. |
| `references.bib` | ORACLE | Currently unused — the LaTeX uses an inline `thebibliography`. Two drifting bibliographies. Pick `.bib` and switch the TeX to `\bibliography`. |

Two migration-specific hazards:

- **`docs/reconstruction_model.md` (C20) will actively mislead the first
  collaborator who opens the repo.** Fix before sharing the link.
- `README.md` "Reproduce the NuBench checks" documents scripts that "are not
  included in this repository". Either migrate them (recommended) or delete the
  section. A README with instructions for absent files is worse than silence.

---

## 6. How to develop this into a publishable paper

Ordered by expected value, not by effort.

### 6.1 — Decide what the paper is actually about, and it is not "layerwise diagnosis"

Given §3, "frozen layerwise representations distinguish shift types" and
"diagnosed-stage adaptation beats wrong-stage adaptation" are both re-derivations
with better-resourced published priors. Continuing to lead with them is the
highest-risk path.

The genuinely open lane, confirmed by the prior-art scan: **nobody has done
harmful-shift detection where the consequence variable is a physics observable
rather than classification accuracy, and nobody monitors learned
neutrino-telescope reconstruction at all.**

Recommended reframing of the title claim, from
> *can a frozen representation attribute a declared failure?*

to
> *when does a monitoring alarm rank scientific consequence, and when does it
> systematically fail to?*

with a **negative or dissociative headline result** being fully publishable in
MLST / JINST / EPJC — and considerably harder to scoop.

### 6.2 — Build the designed dissociation. This is the strongest single experiment. ⭐

The one thing that makes C4 a real result rather than a mechanical correlation
(P1.2) is a family where alarm and consequence are **decoupled by construction**:

| Arm | Construction | Predicted A | Predicted K |
|---|---|---|---|
| Output-null | perturb along the null space of the local output Jacobian `J = ∂y/∂h` | **large** | ≈ 0 |
| Output-aligned | perturb along `J`'s top right-singular directions, matched in norm | **same** | large |
| Norm-matched random | seeded random direction at the same norm | intermediate | intermediate |
| Clean / sham | none | ≈ 0 | ≈ 0 |

Then the claim becomes a falsifiable statement about *metric choice*, not about
detection: **a monitor that ranks by Euclidean displacement in representation
space fails this design by construction; a monitor that ranks by
Jacobian-projected displacement passes.** That is a designed experiment with a
predicted sign, an unambiguous null, and no dependence on NuBench parity, the
geometry gate, TIDMAD's data, or Junjie's availability. It runs on a synthetic
model in an afternoon.

It also converts every generic baseline in the comparison table from "another
detector" into "a detector that provably cannot see the distinction" — which is
exactly the kind of result a strong generic baseline cannot take away from you.

`REVISION_PLAN.md` already lists the output-null control. **Promote it from a
control to the headline experiment, and write it into the LaTeX.**

### 6.3 — Make the covariance geometry the theoretical contribution ⭐

This is the asset this group has and the ML monitoring literature does not: an
explicit noise covariance Σ, a whitening geometry, and Paper 1's OF/EMPCA
framework behind it. The proposal gestures at it — "structured acquisition noise
induces a likelihood geometry through the inverse covariance" — and then never
uses it.

The provable statement, and it composes directly with §6.2:

> Let the encoder be trained under an assumed covariance Σ̂ and deployed under a
> realized Σ. An alarm computed as an unweighted representation displacement is
> the wrong-metric statistic; the consequence-relevant quantity is the
> displacement in the Σ⁻¹-induced (whitened) metric pulled back through the
> output Jacobian. The rank correlation between the two degrades in a way
> controlled by the condition number of Σ̂⁻¹Σ.

That gives you: a lemma (you already derived a Whitening Lemma for the NPML
talk), a predicted *quantitative* mis-ranking as a function of `κ(Σ̂⁻¹Σ)`, and a
controlled experiment where Σ̂ and Σ are both known — i.e. exactly what the
DELight noise simulator in `src/noise_module/` already produces and what the
public benchmarks do not. It is the one part of this project no one else can
write, and it turns "we ported monitoring to physics" into "monitoring has a
metric-misspecification failure mode that physics data makes visible".

**Practical consequence:** the *controlled simulator*, not TIDMAD, should be the
primary waveform testbed. TIDMAD becomes the public-data realism check.

### 6.4 — Fix the testbeds

**NuBench.** Three options, in order of preference:

1. **Re-simulate paired events with Prometheus.** It is open source and it is the
   only route to true event-level pairing. A paired multi-geometry neutrino event
   set would itself be a citable dataset contribution, and it makes the A⊕B
   design executable as originally conceived. Cost is real but bounded, and it
   removes the dependency on undated, unlicensed ERDA tarballs.
2. **Hexagon (water) vs Hexagon Ice LE (ice)** as an explicitly *distribution-
   level* medium contrast. Cheap, but the plan's own validity gate then forbids
   the causal layout claim — report it as a consistency comparison and say so.
3. **Drop the geometry axis** and run NuBench purely as the cross-architecture
   replication of the acquisition-intervention study.

Whichever you choose: declare the **train/test noise-pulse asymmetry** in the
methods, since it directly confounds an N family defined as hit thinning.

**TIDMAD.** Either (a) demote it from "where C4 is decided" to a public-data
realism check and use the controlled simulator for C4 (recommended, and it
follows from §6.3), or (b) if you keep it as K, use the **exclusion limit**
rather than the denoising score and budget for the 208-file / ~600 GB pipeline.
Either way, the 328-bin band truncation and the `--in-band-only` file selection
must be declared in the paper, with the excluded files enumerated.

**Reverse the LArTPC ordering.** Panda before SPINE, on infrastructure cost
alone (P2.3). Panda is pip-installable with four public checkpoints and an
auto-downloading HF dataset; SPINE is a 25 GB container with ROOT + LArCV2 +
MinkowskiEngine from source and `TODO`-marked data. Panda also benchmarks
*against* SPINE, so a Panda result already speaks to the SPINE question. Tell
Junjie this — it is good news for his side of the workload, not a retreat.

### 6.5 — Pre-empt the five threat papers explicitly

Add a Related Work paragraph that names Roschewitz, Lee (Surgical Fine-Tuning),
Podkopaev & Ramdas, Zhang, and Rabanser, and states in one sentence each what
this study does that they do not. Referees forgive a crowded field; they do not
forgive not knowing it is crowded.

For C5 specifically, pre-register the Hase et al. confound: state *before*
running that a positive diagnosed-vs-wrong-stage LoRA result does not confirm
the localization, and a negative one does not falsify it — then say what would.
(The honest answer: an *activation-patching* style intervention, where you
substitute the clean representation at stage *k* into the perturbed forward pass
and measure consequence recovery. That is a direct causal test of localization
and it does not require training anything. It is also cheaper than the four-arm
LoRA study and it is the experiment the mechanistic-interpretability literature
would demand. Consider replacing C5's LoRA arms with it.)

### 6.6 — Sequencing (revised)

The current B → A → A⊕B sequence is blocked at step one by P2.1. Proposed
replacement:

| Phase | Content | Blocked by | Runs now? |
|---|---|---|---|
| **0** | Fix C1–C3 (inference bugs, alias). Fix `docs/reconstruction_model.md`. Migrate `scripts/` + `results/` into ORACLE. Ship a one-command synthetic smoke as Junjie's day-one path. | nothing | **yes** |
| **1** ⭐ | §6.2 output-null dissociation on a synthetic model. Predicted sign, unambiguous null, no data dependency. | nothing | **yes** |
| **2** ⭐ | §6.3 whitening-metric lemma + controlled-simulator experiment with known Σ̂ and Σ. | `noise_module` (already validated) | **yes** |
| **3** | Content-matched N-vs-S design (P1.1) on the controlled testbed, where you can construct matched pairs exactly. | Phase 1 metrics | soon |
| **4** | TIDMAD as public-data realism check. MSE vs χ² arms with K declared, band truncation disclosed. | C1–C3 fixed, T0.5 smoke, κₘ frozen | after Phase 0 |
| **5** | NuBench acquisition study, four `DynEdgeConv` hooks not one, unbiased sample, paired CIs, nested severities. Parity resolved by the CPU/GPU test in C17. | C11–C17 fixed | after Phase 0 |
| **6** | Geometry axis — Prometheus re-simulation *or* explicit distribution-level medium contrast. | Phase 5 | later |
| **7** | Localization causal test (activation patching, §6.5), then LoRA arms only if patching supports the localization. | Phases 3–5 | later |
| **8** | Panda scale transfer. | everything above | optional |

Phases 1 and 2 are the ones to start this week. They are unblocked, they are
where the defensible novelty is, and — unlike the current Phase 1 — they cannot
be invalidated by someone else's release decisions.

### 6.7 — Things to say to Junjie at the first meeting

1. NuBench does not provide common latent events; the paired-geometry pilot as
   agreed is not executable from released artifacts. Present options 1–3 from
   §6.4 and let him pick — this is a physics-validity call and it is his side of
   the ownership split.
2. Panda before SPINE, for infrastructure cost. Reversal of the plan, in his
   favour.
3. His A⊕B idea survives intact — but the labelled non-physical class is better
   instantiated as an **output-null perturbation** (§6.2) than as a random LoRA
   update, because output-null makes the contrast *predictive* rather than merely
   *labelled*.
4. The repo is shareable after Phase 0 only. Do not send the link while
   `docs/reconstruction_model.md` says the live package was deleted.

---

## 7. Ranked problem list (one screen)

**Blocking a scientific result**

1. P2.1 NuBench geometries are not event-paired → Phase 1 unexecutable.
2. P1.1 N and S are not separable in principle on the chosen interventions.
3. P1.2 ρ(A,K) is mechanically positive; no dissociation family in the proposal.
4. P2.2 TIDMAD score is not a scientific-consequence metric, per its authors.
5. C1 `infer.py` int8 saturation — any benchmark number produced today is garbage.
6. C2 `infer.py` writes uninitialised memory for the file tail.

**Blocking publication**

7. S3 / §3 Five directly competing papers, none cited.
8. P1.3 "Layerwise" is one hook on NuBench and two on TIDMAD.
9. C17 Re-inference parity blamed on the wrong cause; five-method preprocessing
   substitution unlisted; GraphNeT #880 is the stronger documented candidate.
10. C11 CC/NC vs `is_track` mislabel — likely explains the V0-A gate failure.
11. P1.4 C1's 80% / 5-point thresholds are unpowered.

**Blocking collaboration**

12. C20 `docs/reconstruction_model.md` says the live package was deleted.
13. C3 `chi2_of` alias still shipped, twice marked done, pinned by a test.
14. C4 The single-difference gate never runs in production.
15. C21 `docs/tidmad.md` documents five features that do not exist and two
    checkpoint paths that do not resolve.
16. §5 `scripts/` and `results/` still in the stale folder; README documents
    absent files.
17. P1.5 Two incompatible claim ladders (C1–C4 vs C1–C5).

**Correctness / integrity, fix before any confirmatory run**

18. C5–C7 Declared-vs-executed config gaps (`num_workers`, `data_root`,
    `psd_window` inert; no epoch loop; no reachability assert).
19. C6 Interrupted runs are indistinguishable from complete ones in
    `run_config.json`.
20. C8–C9 χ² weight unfloored, PSD frame mismatch at the unguarded DC bin.
21. C10 `infer.py` crashes on its default device.
22. C12–C16 Pilot sampling, alignment, pairing and CI defects.
23. C18–C19 Non-deterministic CUDA path; no shuffling, undeclared.
24. C22 Whitening positive control has a vacuous test and no declared threshold.

**Cosmetic but visible**

25. C23–C28 Duplicated model/registry modules, under-specified `pyproject.toml`,
    the wrong parameter-count rationale, dead imports, unused `eps`, unguarded
    division, self-declared "before-run" tolerance.
