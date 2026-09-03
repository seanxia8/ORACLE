# Paper 3 LaTeX proposal

`paper3_proposal.tex` builds the collaboration concept note (six pages after the 2 September 2026 mechanism section and the 3 September positioning, reporting-standard and monitor-stack additions)
**Consequence-Aware Failure Diagnostics for Learned Particle-Reconstruction
Representations**.

Compile from this directory so the relative paths under `figures/` resolve:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper3_proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error paper3_proposal.tex
```

The second pass resolves internal references and citations from the document's
self-contained bibliography. The generated deliverable is
`paper3_proposal.pdf`; the repository-level [`README.md`](../README.md) explains
the study, feasibility results, and NuBench reproduction scripts.
