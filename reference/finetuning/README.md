# Fine-Tuning And Transformer Training References

This directory keeps a small subset of notebooks moved from
`modern_genai_bilibili-main/` before that unrelated source tree was removed.

These notebooks are not part of the DELight runtime. They are reference
material for improving the current detector-waveform Transformer training
pipeline.

## Potentially Useful For This Repo

- `llms/training/llama2.ipynb` and `llms/training/qwen3.ipynb`: examples of
  Transformer training/fine-tuning workflows. The model/data are LLM-specific,
  but the optimizer, scheduler, checkpoint, and experiment-structure ideas can
  inform detector-waveform pretraining.
- `agentic_rl/infra/hf-transformers.ipynb`: Hugging Face Transformer API notes;
  useful if the DELight model is later wrapped in a more standard training
  interface.
- `agentic_rl/infra/fsdp/fsdp_basics.ipynb`,
  `agentic_rl/3D/DDP_NCCL.ipynb`, and `agentic_rl/3D/fsdp_fsdp2.ipynb`:
  distributed-training references for larger runs.
- `agentic_rl/infra/misc/wandb.ipynb`: experiment tracking patterns relevant to
  the existing W&B integration.
- The selected `agentic_rl/verl` and `agentic_rl/training_practices` notebooks:
  monitoring, SFT, performance tuning, precision, and train/inference mismatch
  notes that may help when turning current supervised training into a cleaner
  pretraining/fine-tuning pipeline.

## How To Use

Treat these as reading material, not importable project code. If a concrete
idea is adopted, migrate it into `configs/`, `scripts/`, or
`reconstruction_model/` with a small reproducible experiment.
