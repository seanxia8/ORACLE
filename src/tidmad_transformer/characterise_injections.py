"""Survey the injected tone in every TIDMAD validation file (checklist P1).

Why this exists
---------------
The frozen STFT keeps bins ``0..n_bands_used-1``, i.e. 0–~800 kHz of a 0–5 MHz
band. That was justified as covering "every characterised injection frequency",
but ``tidmad_data_contract.json`` characterises only files 0000, 0005 and 0010
(plus a sweep note for 0015), while its own coupling scan runs out to 3.4 MHz.
If any validation file carries an injection above the modelled band, the model
receives no signal for that file at all, and a band-limited reconstruction is
not performing the same task as the published full-band baselines.

This module settles the question empirically before the config is frozen.

What it reports, per file
-------------------------
The contract warns that the injected frequency is *sample-range specific* and
sweeps within a file, so a single peak is not enough. For each file we probe
three positions (start, middle, end) and record:

* ``peak_hz``       — argmax of the reference-channel spectrum in that window;
* ``prominence``    — peak power over the median power, i.e. how tone-like it is;
* ``coupling``      — |readout peak| / |reference peak| at the same bin, the
                      quantity the contract reports falling from 7.13 at 5.7 kHz
                      to 0.0021 at 3.4 MHz;
* ``in_band``       — whether ``peak_hz`` falls inside the modelled band.

Usage
-----
    python -m tidmad_transformer.characterise_injections \\
        --data-dir $TIDMAD_DATA_ROOT \\
        --glob 'abra_validation_*.h5' \\
        --out   injection_survey.json

Exit status is 0 if every probed peak is in band, 2 otherwise, so the caller can
gate on it. A non-zero exit is a finding, not an error: see checklist P1 for the
decision it forces.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from .config import FROZEN, TidmadSTFTConfig

# Probe positions as a fraction of the usable file length.
PROBE_FRACTIONS = (0.0, 0.5, 0.95)


def _read_window(fh, dataset: str, start: int, n: int, shift: int) -> np.ndarray:
    """Read ``n`` samples as float, applying the release's int8 + shift convention."""
    raw = fh[dataset][start : start + n]
    return np.asarray(raw, dtype=np.float64) + shift


def probe_position(
    fh,
    squid_dataset: str,
    reference_dataset: str,
    start: int,
    cfg: TidmadSTFTConfig,
    n_probe: int,
) -> dict:
    """Characterise the tone in one window at ``start``."""
    ref = _read_window(fh, reference_dataset, start, n_probe, cfg.sample_shift)
    sq = _read_window(fh, squid_dataset, start, n_probe, cfg.sample_shift)

    ref = ref - ref.mean()
    sq = sq - sq.mean()

    window = np.hanning(ref.size)
    ref_spec = np.abs(np.fft.rfft(ref * window))
    sq_spec = np.abs(np.fft.rfft(sq * window))
    freqs = np.fft.rfftfreq(ref.size, d=1.0 / cfg.sampling_frequency_hz)

    # Ignore DC and the first few bins: mean removal leaves a residual there.
    lo = 3
    peak_bin = int(np.argmax(ref_spec[lo:])) + lo
    peak_power = float(ref_spec[peak_bin])
    median_power = float(np.median(ref_spec[lo:]))

    prominence = peak_power / median_power if median_power > 0 else float("inf")
    coupling = float(sq_spec[peak_bin] / peak_power) if peak_power > 0 else 0.0

    return {
        "sample_start": int(start),
        "peak_hz": float(freqs[peak_bin]),
        "peak_bin": peak_bin,
        "prominence": prominence,
        "coupling": coupling,
    }


def characterise_file(path: Path, cfg: TidmadSTFTConfig, n_probe: int) -> dict:
    import h5py

    band_edge_hz = cfg.n_bands_used * (cfg.sampling_frequency_hz / cfg.n_fft)
    squid = "timeseries/channel0001/timeseries"
    reference = "timeseries/channel0002/timeseries"

    with h5py.File(path, "r") as fh:
        for required in (squid, reference):
            if required not in fh:
                raise KeyError(
                    f"{path.name}: {required} missing. The data contract no longer holds — STOP."
                )
        n_samples = int(fh[squid].shape[0])
        usable = max(n_samples - n_probe, 1)
        probes = [
            probe_position(fh, squid, reference, int(frac * usable), cfg, n_probe)
            for frac in PROBE_FRACTIONS
        ]

    peaks = [p["peak_hz"] for p in probes]
    return {
        "file": path.name,
        "n_samples": n_samples,
        "probes": probes,
        "peak_hz_min": min(peaks),
        "peak_hz_max": max(peaks),
        "sweep_hz": max(peaks) - min(peaks),
        "band_edge_hz": band_edge_hz,
        "in_band": bool(max(peaks) < band_edge_hz),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--glob", default="abra_validation_*.h5")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=1 << 20,
        help="samples per probe window; larger gives finer frequency resolution",
    )
    args = parser.parse_args()

    cfg = FROZEN.stft
    files = sorted(glob.glob(os.path.join(str(args.data_dir), args.glob)))
    if not files:
        print(f"no files matched {args.glob!r} in {args.data_dir}", file=sys.stderr)
        return 1

    results = []
    for path in files:
        record = characterise_file(Path(path), cfg, args.probe_samples)
        results.append(record)
        flag = "in band" if record["in_band"] else "OUT OF BAND"
        print(
            f"{record['file']}: {record['peak_hz_min']/1e3:9.2f}"
            f"–{record['peak_hz_max']/1e3:9.2f} kHz  "
            f"sweep {record['sweep_hz']/1e3:7.2f} kHz  "
            f"coupling {record['probes'][0]['coupling']:.4f}  [{flag}]"
        )

    out_of_band = [r["file"] for r in results if not r["in_band"]]
    band_edge_khz = results[0]["band_edge_hz"] / 1e3
    summary = {
        "band_edge_hz": results[0]["band_edge_hz"],
        "n_files": len(results),
        "n_out_of_band": len(out_of_band),
        "out_of_band_files": out_of_band,
        "probe_samples": args.probe_samples,
        "probe_fractions": list(PROBE_FRACTIONS),
        "files": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"\nmodelled band: 0–{band_edge_khz:.1f} kHz   survey written to {args.out}")
    if out_of_band:
        print(
            f"\n{len(out_of_band)} of {len(results)} files carry an injection ABOVE the "
            f"modelled band:\n  " + "\n  ".join(out_of_band),
            file=sys.stderr,
        )
        print(
            "\nThe band-limited model cannot reconstruct those files, and the comparison\n"
            "to the published full-band baselines is not valid as configured.\n"
            "Decide per checklist P1: widen coverage, or drop the baseline column.",
            file=sys.stderr,
        )
        return 2

    print(f"all {len(results)} files fall inside the modelled band; coverage confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
