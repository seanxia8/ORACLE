"""Inference and scoring for the TIDMAD arms (PLAN_04 §3.5).

``infer.py`` runs a trained ``tidmad_stft`` model over every validation file,
writes ``abra_validation_denoised_<arm>_00NN.h5`` in the release's layout
(denoised SQUID in ``channel0001``, the untouched reference in ``channel0002``),
reusing Paper 1's h5 writer so the file semantics match ``inference.py`` exactly.

``score.py`` then invokes the *unmodified* upstream ``benchmark.py -c`` and
captures its stdout/returncode. There is deliberately no internal reimplementation
of the score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from reconstruction_model.checkpoints import load_checkpoint

from .config import FROZEN
from .data import build_window_stft, spectrogram_to_series
from .train import build_model, load_run_config


def _write_denoised_channel(denoised: np.ndarray, source: Path, destination_dir: Path, model_tag: str):
    """Port of Paper 1's writer (tidmad._vendor.score.write_denoised_file)."""
    from ._vendor.score import DenoisedFileSpec, write_denoised_file

    spec = DenoisedFileSpec(source_file=source, destination_dir=destination_dir, model_tag=model_tag)
    write_denoised_file(
        denoised,
        spec,
        source_dataset="timeseries/channel0001/timeseries",
        attributes={"denoiser": model_tag},
    )


def _copy_reference_channel(source: Path, out_dir: Path, model_tag: str):
    """Copy the untouched reference channel into the denoised file (benchmark reads both).

    Uses the same ``DenoisedFileSpec`` as the denoised write so the reference
    lands in the same destination file via the writer's append mode.
    """
    from ._vendor.score import DenoisedFileSpec, write_denoised_file

    import h5py

    with h5py.File(source, "r") as f:
        ref = np.asarray(f["timeseries"]["channel0002"]["timeseries"], dtype=np.float64)
    write_denoised_file(
        ref,
        DenoisedFileSpec(source_file=source, destination_dir=out_dir, model_tag=model_tag),
        source_dataset="timeseries/channel0002/timeseries",
    )


def infer(config: Path, checkpoint: Path, data_dir: Path, out_dir: Path, model_tag: str, device: str):
    run = load_run_config(config)
    model = build_model(run).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    from ._vendor.loader import load_contract

    contract = load_contract(run.data.contract_path)
    contract.require_verified()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import h5py

    # Select the 20 raw release files only. The legacy server root also contains
    # pre-existing abra_validation_denoised_* products.
    for path in sorted(data_dir.glob("abra_validation_00??.h5")):
        index = path.name.split("_")[-1].split(".")[0]
        with h5py.File(path, "r") as f:
            n_samples = int(f["timeseries"]["channel0001"]["timeseries"].shape[0])
        n = run.data.window_samples
        denoised = np.empty(n_samples, dtype=np.float64)
        pos = 0
        with torch.no_grad():
            while pos + n <= n_samples:
                spec_in, _ = build_window_stft(
                    path, contract, pos, run.stft, run.data.window_samples
                )
                z = torch.as_tensor(spec_in, dtype=torch.float32).unsqueeze(0).to(device)
                out_std = model(z)
                mean_in = torch.nanmean(torch.as_tensor(spec_in), dim=-1, keepdim=True)
                std_in = torch.std(torch.as_tensor(spec_in), dim=-1, keepdim=True)
                out_meas = out_std * (std_in + 1e-6) + mean_in
                seg = spectrogram_to_series(out_meas.squeeze(0).cpu().numpy(), run.stft, n)
                denoised[pos : pos + n] = seg
                pos += n
        _write_denoised_channel(denoised, path, out_dir, model_tag)
        _copy_reference_channel(path, out_dir, model_tag)
        print(f"wrote {out_dir / f'abra_validation_denoised_{model_tag}_{index}.h5'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    infer(args.config, args.checkpoint, args.data_dir, args.out, args.model_tag, args.device)


if __name__ == "__main__":
    main()
