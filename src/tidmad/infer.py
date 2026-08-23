"""Inference and scoring for the TIDMAD arms (PLAN_04 §3.5).

``infer.py`` runs a trained ``tidmad_stft`` model over every validation file,
writes ``abra_validation_denoised_<arm>_00NN.h5`` in the release's layout
(denoised SQUID in ``channel0001``, the untouched reference in ``channel0002``),
reusing Paper 1's h5 writer so the file semantics match ``inference.py`` exactly.

``score.py`` then invokes the *unmodified* upstream ``benchmark.py -c`` and
captures its stdout/returncode. There is deliberately no internal reimplementation
of the score.

Coordinate frames (audit B1/C1): the data pipeline reads both channels as
``int8`` and shifts by ``+ sample_shift`` (release convention, see
``data.read_channel_pair``). The model therefore reconstructs in the SHIFTED
frame, and the shift must be subtracted again before the writer casts back to
the release's ``int8`` dtype — otherwise every sample saturates at +127.
``_verify_denoised_file`` fail-fast checks the written product (plan T0.6).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from reconstruction_model.checkpoints import load_checkpoint

from .data import build_window_stft, spectrogram_to_series
from .loss import per_row_stats, unstandardise
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


def _verify_denoised_file(path: Path, n_samples: int) -> None:
    """Fail-fast contract check on the written two-channel product (T0.6).

    Guards the two failure modes the audit found silent: a missing/short second
    channel, and a saturated (near-constant) denoised channel — the symptom of
    writing a ``+shift``-frame series into an ``int8`` dataset.
    """
    import h5py

    with h5py.File(path, "r") as f:
        for channel in ("channel0001", "channel0002"):
            key = f"timeseries/{channel}/timeseries"
            if key not in f:
                raise RuntimeError(f"{path.name}: missing dataset {key!r}")
            if int(f[key].shape[0]) != int(n_samples):
                raise RuntimeError(
                    f"{path.name}: {key} has {f[key].shape[0]} samples, expected {n_samples}"
                )
        probe = np.asarray(f["timeseries/channel0001/timeseries"][: 1 << 20])
    if np.unique(probe).size < 3:
        raise RuntimeError(
            f"{path.name}: denoised channel is (near-)constant over the first "
            f"2^20 samples — this is the saturation signature of a coordinate-"
            "frame error before the int8 cast. Refusing to hand this to the "
            "benchmark."
        )


def _denoise_window(model, spec_in: np.ndarray, run, device) -> np.ndarray:
    """One window: standardised forward pass, then back to measurement frame.

    Uses the shared ``loss.per_row_stats`` / ``loss.unstandardise`` helpers
    (plan T1.4) with every tensor on ``device`` (audit M9/C10: the previous
    inline copy mixed float64 CPU statistics with float32 device outputs and
    crashed under ``--device cuda``).
    """
    z = torch.as_tensor(spec_in, dtype=torch.float32, device=device).unsqueeze(0)
    out_std = model(z)
    mean_in, std_in = per_row_stats(z)
    out_meas = unstandardise(out_std, mean_in, std_in)
    return spectrogram_to_series(
        out_meas.squeeze(0).cpu().numpy(), run.stft, run.data.window_samples
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

    shift = float(run.stft.sample_shift)

    # Select the 20 raw release files only. The legacy server root also contains
    # pre-existing abra_validation_denoised_* products.
    for path in sorted(data_dir.glob("abra_validation_00??.h5")):
        index = path.name.split("_")[-1].split(".")[0]
        with h5py.File(path, "r") as f:
            n_samples = int(f["timeseries"]["channel0001"]["timeseries"].shape[0])
        n = run.data.window_samples
        if n_samples < n:
            raise RuntimeError(
                f"{path.name}: file has {n_samples} samples, shorter than one "
                f"window ({n}); cannot denoise"
            )
        denoised = np.empty(n_samples, dtype=np.float64)
        pos = 0
        with torch.no_grad():
            while pos + n <= n_samples:
                spec_in, _ = build_window_stft(path, contract, pos, run.stft, n)
                denoised[pos : pos + n] = _denoise_window(model, spec_in, run, device)
                pos += n
            # Tail (audit B2/C2): the final n_samples % n samples were
            # previously left as uninitialised np.empty memory. Denoise one
            # last window ending exactly at the file end and keep only the
            # part not already written.
            if pos < n_samples:
                tail_start = n_samples - n
                spec_in, _ = build_window_stft(path, contract, tail_start, run.stft, n)
                seg = _denoise_window(model, spec_in, run, device)
                denoised[pos:] = seg[pos - tail_start :]
        # Back to the release's unshifted int8 frame before the writer casts
        # (audit B1/C1). The reference channel is copied raw and unshifted, so
        # after this both channels share one frame.
        denoised -= shift
        _write_denoised_channel(denoised, path, out_dir, model_tag)
        _copy_reference_channel(path, out_dir, model_tag)
        out_path = out_dir / f"abra_validation_denoised_{model_tag}_{index}.h5"
        _verify_denoised_file(out_path, n_samples)
        print(f"wrote {out_path}")


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
