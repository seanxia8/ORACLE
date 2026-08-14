"""Phase 5: write denoised files and score them with the *upstream* benchmark.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`` Phase 5.

The single most important rule in this module is that it **does not reimplement
the denoising score**. The score is defined by the release's ``benchmark.py``;
a reimplementation that disagreed by a normalization factor would produce a
number that looks comparable to the published scoreboard and is not. So this
module does exactly two things:

1. writes a denoised time series into the file layout ``inference.py`` produces,
   preserving the dataset names, dtype, and length the release expects;
2. shells out to the upstream ``benchmark.py`` and captures what it prints.

If the upstream script is unavailable, this fails loudly rather than falling
back to an internal approximation.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["DenoisedFileSpec", "write_denoised_file", "run_upstream_benchmark"]


@dataclass(frozen=True)
class DenoisedFileSpec:
    """Where a denoised file goes and what it must look like.

    ``model_tag`` becomes part of the filename because ``benchmark.py`` locates
    files by the pattern ``abra_validation_denoised_<model>_00NN.h5``. Using a
    distinctive tag (default ``wiener``) keeps our output from colliding with
    the release's own model outputs in the same directory.
    """

    source_file: Path
    destination_dir: Path
    model_tag: str = "wiener"
    dataset_name: str | None = None
    dtype: str | None = None

    @property
    def destination(self) -> Path:
        stem = self.source_file.stem  # e.g. abra_validation_0000
        parts = stem.split("_")
        index = parts[-1]
        prefix = "_".join(parts[:-1])
        return self.destination_dir / f"{prefix}_denoised_{self.model_tag}_{index}.h5"


def write_denoised_file(
    denoised: np.ndarray,
    spec: DenoisedFileSpec,
    *,
    source_dataset: str,
    attributes: dict | None = None,
) -> Path:
    """Write a denoised series in the release's file layout.

    The dtype of the source dataset is preserved. TIDMAD stores digitized SQUID
    samples in a narrow integer type; writing float64 would change the file
    semantics and could change the score for reasons that have nothing to do
    with the filter.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("h5py is required to write denoised TIDMAD files") from exc

    spec.destination_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(spec.source_file, "r") as source:
        if source_dataset not in source:
            raise KeyError(
                f"dataset {source_dataset!r} not in {spec.source_file.name}; "
                f"available: {list(source.keys())}"
            )
        target_dtype = np.dtype(spec.dtype) if spec.dtype else source[source_dataset].dtype
        # benchmark.py reads voltage_range_mV, sampling_frequency, etc. as
        # GROUP-level attrs (e.g. h5f['timeseries']['channel0001'].attrs[...]),
        # not on the leaf dataset -- copy them through, since h5py's implicit
        # intermediate-group creation below would otherwise leave them empty.
        group_attrs = dict(source[source_dataset].parent.attrs)

    values = np.asarray(denoised)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    values = values.astype(target_dtype, copy=False)

    name = spec.dataset_name or source_dataset
    # Append mode: the destination file must hold both the denoised SQUID
    # channel and the untouched reference channel (benchmark.py reads both,
    # for every model -- it locates the reference/injection frequency from
    # channel0002 regardless of which channel is under test). A second call
    # for the reference channel must not truncate the first call's dataset.
    with h5py.File(spec.destination, "a") as out:
        if name in out:
            del out[name]
        dataset = out.create_dataset(name, data=values, compression="gzip", compression_opts=1)
        dataset.attrs["denoiser"] = spec.model_tag
        dataset.attrs["source_file"] = spec.source_file.name
        for key, value in (attributes or {}).items():
            dataset.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        for key, value in group_attrs.items():
            dataset.parent.attrs[key] = value

    return spec.destination


def run_upstream_benchmark(
    upstream_dir: Path,
    data_dir: Path,
    model_tag: str,
    *,
    coarse: bool = True,
    extra_args: tuple[str, ...] = (),
    timeout_s: int = 3600,
) -> dict:
    """Invoke the release's ``benchmark.py`` and capture its output verbatim.

    Returns the command, exit status, and captured streams. Parsing is left to
    the caller and kept out of the archived record: the raw output is the
    evidence, and a parser that silently mis-reads a future output format should
    not be able to corrupt it.
    """
    script = Path(upstream_dir) / "benchmark.py"
    if not script.exists():
        raise FileNotFoundError(
            f"upstream benchmark not found at {script}. Clone jessicafry/TIDMAD and "
            "pass --upstream. This module deliberately has no fallback scorer: an "
            "internally reimplemented score would not be comparable to the "
            "published scoreboard."
        )

    # Prefer a venv inside the checkout, if one exists: the release's code was
    # written against numpy<2 (e.g. int8_array + 128 relies on value-based
    # scalar promotion, removed by NEP 50 in numpy 2.0), so running it under a
    # current-numpy interpreter raises OverflowError on that exact pattern.
    venv_python = Path(upstream_dir) / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else "python"

    command = [python, "benchmark.py", "-d", str(data_dir), "-m", model_tag]
    if coarse:
        command.append("-c")
    command.extend(extra_args)

    completed = subprocess.run(
        command,
        cwd=str(upstream_dir),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(upstream_dir),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "succeeded": completed.returncode == 0,
    }
