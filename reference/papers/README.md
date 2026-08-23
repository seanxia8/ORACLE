# Reference papers

PDFs of the prior art the proposal must position against, plus the testbed and
simulation papers it depends on. **PDFs are committed on purpose** — this folder
is the shared reading list for the collaboration, and a bibliography entry is not
a substitute for the paper when the question is "does this already do what we
claim to do".

Every entry is on arXiv and redistributable under its arXiv licence or an open
licence; nothing here is a publisher-typeset copy.

## Fetching the PDFs

The manifest is [`papers.tsv`](papers.tsv) — tab-separated
`arxiv_id <TAB> folder <TAB> filename_stem <TAB> title`. Everything in it is on
arXiv, so one loop fetches the set:

```bash
cd reference/papers
while IFS=$'\t' read -r id dir name _; do
  out="$dir/${id}_${name}.pdf"
  if [ -s "$out" ]; then continue; fi
  echo "fetching $id"
  curl -fsSL --create-dirs -o "$out" "https://arxiv.org/pdf/$id" || echo "  FAILED $id"
  sleep 3
done < papers.tsv
```

Run it from a machine with ordinary internet access — arXiv is not reachable
from the sandboxed agent environments. It is idempotent, so re-running it after
a partial or failed fetch only picks up what is missing, and it sleeps between
requests, which is what arXiv asks of scripted clients.

`2512.01324` (Panda) is already committed, carried over from the old `paper3/`
folder, so the loop will skip it.

To add a paper: append a row to `papers.tsv`, re-run the loop, and add the
matching `references.bib` entry.

---

## Venue, and why both lists below matter

**Target: Machine Learning: Science and Technology (MLST).**

This is the venue where the prior-art burden is heaviest, because MLST draws
referees from *both* communities and asks for either ML novelty or a genuinely
new scientific application. Two consequences:

- The ML papers in §1 are a real novelty threat, not just citation hygiene. At a
  detector-instrumentation venue (JINST, Comp. Softw. Big Sci.) they would drop
  to a Related Work paragraph; at MLST at least one referee will know Surgical
  Fine-Tuning by name.
- The HEP monitoring papers in §2 are *also* live, because MLST's physics-side
  referees run or read CMS DQM.

The defensible position under both pools is the same one: the consequence
variable is a **physics observable**, not classification accuracy; labels never
arrive on real detector data; and the assumed and realized noise covariance are
both known. See "What survives" at the bottom.

---

## `prior_art/`

Full analysis in [`../../docs/PAPER3_AUDIT.md`](../../docs/PAPER3_AUDIT.md) §3.

### 1. Directly competing ML work (audit S3)

| arXiv | Paper | Threatens | Why |
|---|---|---|---|
| 2411.07940 | Roschewitz et al., *Automatic dataset shift identification*, MICCAI 2025 | Claim 1 | Unsupervised, frozen-encoder, classifies *which type* of shift occurred (prevalence / covariate / mixed) via BBSD + MMD on frozen features. Same goal, same architectural choice. |
| 2210.11466 | Lee et al., *Surgical Fine-Tuning*, ICLR 2023 | Claim 3 / C5 | "The type of distribution shift influences which subset of layers is more effective to tune" — with an automatic gradient-norm criterion for picking the block. C5 is structurally this, LoRA for full-block tuning. |
| 2110.06177 | Podkopaev & Ramdas, *Tracking the risk of a deployed model*, ICLR 2022 | Claim 2 | "Detect harmful shifts while ignoring benign ones" is the thesis, four years earlier. **Limit:** their risk tracking needs labels to arrive eventually. On real detector data they do not. |
| 2210.10769 | Zhang et al., *Why did the Model Fail?*, ICML 2023 | Claims 1+2 jointly | Shapley attribution of a *performance change* to specific shift mechanisms — already couples attribution to consequence. |
| 2301.04213 | Hase et al., *Does Localization Inform Editing?*, NeurIPS 2023 | C5's validation **logic** | Causal-tracing localization does not predict where editing works. A positive diagnosed-vs-wrong-stage LoRA result therefore does not confirm the localization, and a negative does not falsify it. **Pre-register this before running C5.** |

### 2. The physics referee's first question

These are production-grade acquisition-violation detectors on a real experiment.
Expect *"why not just use DQM?"* and *"why not train the robustness in?"* — and
answer both in the manuscript, not in the rebuttal.

| arXiv | Paper | Why it bites |
|---|---|---|
| 2501.13789 | CMS ML data-quality monitoring | Deployed ML DQM at scale. The N-detection half of this proposal is, from a HEP referee's seat, DQM by another name. |
| 2309.10157 | CMS ECAL autoencoder anomaly detection | Autoencoder-based detector-level anomaly detection, already in production use. |
| 2407.20278 | CMS ECAL autoencoder follow-up | Extends the above. |
| 2105.08742 | Systematics-aware learning | If the acquisition variation is known, train invariance to it rather than monitor for it. Needs an explicit answer. |

The honest distinction to draw against all four: DQM answers *"is the detector
behaving"*; this asks *"has the reconstruction model's internal representation
moved in a direction that costs physics"* — a question about the model, not the
apparatus, and one DQM does not pose.

### 3. Unavoidable, must be cited

| arXiv | Paper | Role |
|---|---|---|
| 1810.11953 | Rabanser et al., *Failing Loudly*, NeurIPS 2019 | Canonical shift-detection benchmark; already pairs detection power against whether accuracy degraded. Has crossed into HEP reading lists, so it is live in both referee pools. |
| 1912.08142 | Castro et al., *Causality matters in medical imaging*, Nat. Commun. 2020 | Establishes the acquisition-shift vs manifestation/population-shift taxonomy that N/S reproduces in detector language. |
| 2204.05306 | Yang et al., *Full-Spectrum OOD Detection*, IJCV 2023 | Covariate-OOD (tolerate) vs semantic-OOD (reject), with a low/high feature split. |
| 2307.15647 | Lambert et al., *Multi-layer Aggregation*, UNSURE 2023 | Single-layer OOD detectors behave inconsistently across anomaly types — the empirical core of the layerwise claim, already published. |
| 2203.14960 / 2107.00758 | Domino / Spotlight | Slice discovery: the S branch alone is largely solved in frozen representation space. |
| 2201.04234 / 2107.03315 | ATC / DoC | Confidence-derived scalars already predict OOD accuracy — "alarm magnitude predicts consequence" is largely answered affirmatively **for classification**. The gap is that K here is a continuous physics observable. |
| 2104.08279 / 2201.02331 | Bates et al. / iDECODe | Conformal outlier machinery. Claim only the *use*, not the method. |

Not on arXiv, so not in this folder — cite from the journal:
**Bayram, Ahmed & Kassler, *From concept drift to model degradation*, KBS 2022**
(DOI 10.1016/j.knosys.2022.108632), a survey of performance-aware drift detectors.

## `testbeds/` — what the study runs on

| arXiv | Paper | Role here |
|---|---|---|
| 2512.01324 | Young & Terao, *Panda* | Optional scale-transfer testbed; ordered before SPINE (`docs/OPEN_DECISIONS.md` D3). |
| 2511.13111 | NuBench | Cross-architecture testbed. §3.2 specifies the detector response but does not release it as code. |
| 2406.04378 | TIDMAD | Waveform testbed. Read the Benchmark 2 section — the denoising score is disclosed by its own authors as lacking direct scientific relevance (D2). |
| 2304.14526 | Prometheus | §5.4.2 is the mechanism for true event-paired multi-geometry simulation (D1) — the route around the NuBench pairing blocker. |
| 2411.09864 | Douglas et al. | Representation-shift preprint, still v3. Cite as a preprint, no publication-status claim. |

---

## What survives the prior-art scan

Ordered by how well each holds up under **both** referee pools.

1. ⭐ **The covariance-geometry result** (audit §6.3). An unweighted representation
   displacement is the wrong-metric statistic; the consequence-relevant quantity is
   displacement in the `Sigma^-1` metric pulled back through the output Jacobian, and
   their rank agreement degrades with the condition number of `Sigma_hat^-1 Sigma`.
   Requires a known assumed *and* realized covariance — which `src/noise_module/`
   provides and no public ML benchmark does. Untouched in either literature, and it
   is an ML-methods result rather than an application, which is what MLST asks for.
2. ⭐ **The output-null dissociation** (audit §6.2). A perturbation family where alarm
   and consequence are decoupled *by construction*, so the claim becomes a falsifiable
   statement about metric choice with a predicted sign and an unambiguous null.
3. **Physics-observable consequence variable.** Every paper in §1 uses classification
   accuracy. Angular resolution and an exclusion limit are continuous quantities with
   their own systematics; the alarm-consequence relation is not obviously the same object.
4. **The label-free regime.** Podkopaev & Ramdas and the ATC/DoC line assume labels
   or a proxy eventually arrive. On real detector data they do not — which is why the
   representation-side evidence has to carry the claim.
5. **No prior work monitors a learned particle-reconstruction model's internals**, and
   none exists for neutrino telescopes. Real, but a domain-transfer contribution.
6. The **conditional-on-alarm consequence AUROC** with the explicit 2x2 contingency.
   No prior instance found. Genuine but modest — frame it as an evaluation-protocol
   refinement of harmful-shift detection, not as the headline.

**Consequence for the manuscript:** drop "novel" from claims 1 and 3, add a Related
Work paragraph naming Roschewitz / Lee / Podkopaev & Ramdas / Zhang / Rabanser and the
CMS DQM line with one sentence each on what this study does that they do not, and lead
with items 1-2.
