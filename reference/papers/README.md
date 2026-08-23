# Reference papers

PDFs of the prior art the proposal must position against, plus the testbed and
simulation papers it depends on. **PDFs are committed on purpose** — this folder
is the shared reading list for the collaboration, and a bibliography entry is not
a substitute for the paper when the question is "does this already do what we
claim to do".

Every entry is on arXiv and redistributable under its arXiv licence or an open
licence; nothing here is a publisher-typeset copy.

## Fetching

The manifest is [`papers.tsv`](papers.tsv) — tab-separated
`arxiv_id <TAB> folder <TAB> filename_stem <TAB> title`. To download everything
listed there (idempotent; skips files already present):

```bash
bash reference/papers/fetch_papers.sh
```

Run it from a machine with ordinary internet access — arXiv is not reachable
from the sandboxed agent environments. `2512.01324` (Panda) is already present,
carried over from the old `paper3/` folder.

To add a paper: append a row to `papers.tsv`, re-run the script, and add the
`references.bib` entry.

## `prior_art/` — the papers a referee will name

Grouped by which claim they threaten. Full analysis in
[`../../docs/PAPER3_AUDIT.md`](../../docs/PAPER3_AUDIT.md) §3.

### Directly competing (audit S3)

| arXiv | Paper | Threatens | Why |
|---|---|---|---|
| 2411.07940 | Roschewitz et al., *Automatic dataset shift identification*, MICCAI 2025 | Claim 1 | Unsupervised, frozen-encoder, classifies *which type* of shift occurred (prevalence / covariate / mixed) via BBSD + MMD on frozen features. Same goal, same architectural choice. |
| 2210.11466 | Lee et al., *Surgical Fine-Tuning*, ICLR 2023 | Claim 3 / C5 | "The type of distribution shift influences which subset of layers is more effective to tune" — with an automatic gradient-norm criterion for picking the block. C5 is structurally this, LoRA for full-block tuning. |
| 2110.06177 | Podkopaev & Ramdas, *Tracking the risk of a deployed model*, ICLR 2022 | Claim 2 | "Detect harmful shifts while ignoring benign ones" is the thesis, four years earlier. |
| 2210.10769 | Zhang et al., *Why did the Model Fail?*, ICML 2023 | Claims 1+2 jointly | Shapley attribution of a *performance change* to specific shift mechanisms — already couples attribution to consequence. |
| 2301.04213 | Hase et al., *Does Localization Inform Editing?*, NeurIPS 2023 | C5's validation **logic** | Causal-tracing localization does not predict where editing works. A positive diagnosed-vs-wrong-stage LoRA result therefore does not confirm the localization, and a negative does not falsify it. Pre-register this. |

### Unavoidable, must be cited

| arXiv | Paper | Role |
|---|---|---|
| 1810.11953 | Rabanser et al., *Failing Loudly*, NeurIPS 2019 | Canonical shift-detection benchmark; already pairs detection power against whether accuracy degraded. First paper a referee names. |
| 1912.08142 | Castro et al., *Causality matters in medical imaging*, Nat. Commun. 2020 | Establishes the acquisition-shift vs manifestation/population-shift taxonomy that N/S reproduces in detector language. |
| 2204.05306 | Yang et al., *Full-Spectrum OOD Detection*, IJCV 2023 | Covariate-OOD (tolerate) vs semantic-OOD (reject), with a low/high feature split. |
| 2307.15647 | Lambert et al., *Multi-layer Aggregation*, UNSURE 2023 | Single-layer OOD detectors behave inconsistently across anomaly types — the empirical core of the layerwise claim, already published. |
| 2203.14960 / 2107.00758 | Domino / Spotlight | Slice discovery: the S branch alone is largely solved in frozen representation space. |
| 2201.04234 / 2107.03315 | ATC / DoC | Confidence-derived scalars already predict OOD accuracy — "alarm magnitude predicts consequence" is largely answered affirmatively for classification. |
| 2104.08279 / 2201.02331 | Bates et al. / iDECODe | Conformal outlier machinery. Claim only the *use*, not the method. |

Not on arXiv, so not in this folder — cite from the journal:
**Bayram, Ahmed & Kassler, *From concept drift to model degradation*, KBS 2022**
(DOI 10.1016/j.knosys.2022.108632), a survey of performance-aware drift detectors.

### HEP context to position against

| arXiv | Role |
|---|---|
| 2501.13789, 2309.10157, 2407.20278 | CMS ML data-quality monitoring and ECAL autoencoders — production-grade acquisition-violation detectors. Expect *"why not just use DQM?"* |
| 2105.08742 | Systematics-aware training — invites *"why not train the robustness in?"* |

## `testbeds/` — what the study runs on

| arXiv | Paper | Role here |
|---|---|---|
| 2512.01324 | Young & Terao, *Panda* | Optional scale-transfer testbed; ordered before SPINE (`docs/OPEN_DECISIONS.md` D3). |
| 2511.13111 | NuBench | Cross-architecture testbed. §3.2 specifies the detector response but does not release it as code. |
| 2406.04378 | TIDMAD | Waveform testbed. Read §Benchmark 2 — the denoising score is disclosed by its own authors as lacking direct scientific relevance (D2). |
| 2304.14526 | Prometheus | §5.4.2 is the mechanism for true event-paired multi-geometry simulation (D1) — the route around the NuBench pairing blocker. |
| 2411.09864 | Douglas et al. | Representation-shift preprint, still v3. Cite as a preprint, no publication-status claim. |

## What honestly survives the prior-art scan

1. No prior work monitors a learned **particle-reconstruction** model's internals for N-vs-S, and none exists for neutrino telescopes. Real, but a domain-transfer contribution.
2. The **conditional-on-alarm consequence AUROC** with the explicit 2x2 alarm-by-consequence contingency — no prior instance found. Genuine but modest.
3. The layerwise profile as a *diagnostic output* rather than a detection ingredient. Defensible, though Lambert + Surgical Fine-Tuning together already imply it.
4. **The covariance-geometry result** (audit §6.3) — that an unweighted representation displacement is the wrong-metric statistic, and its rank agreement with consequence degrades with the condition number of `Sigma_hat^-1 Sigma`. This needs a known assumed *and* realized covariance, which `src/noise_module/` provides and no public ML benchmark does. **This is the part of the project no one else can write.**

Consequence: drop "novel" from claims 1 and 3, and lead with 4 (plus the
output-null dissociation, audit §6.2).
