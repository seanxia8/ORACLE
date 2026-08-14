"""Score a TIDMAD arm through the *unmodified* upstream ``benchmark.py``.

Reuses Paper 1's wrapper (``tidmad._vendor.score.run_upstream_benchmark``),
which shells out to the release's script and captures the raw output. There is
deliberately no internal reimplementation of the score: a reimplementation that
disagreed by a normalization factor would not be comparable to the published
scoreboard.

Band-restricted scoring
-----------------------
The frozen representation covers 0-800.8 kHz. The P1 survey found four
validation files (0016-0019) whose injected tone lies above that edge, reaching
~4.8 MHz. A band-limited model cannot reconstruct those files at all, so a
20-file score mixes the quantity of interest with four files where every
band-limited arm necessarily fails by construction.

``--in-band-only`` restricts scoring to the files the survey marks in band, by
staging symlinks to just those denoised outputs into a scratch directory and
pointing the unmodified benchmark script at it. The script itself is never
touched or reimplemented; it simply sees a directory containing a subset.

This restriction applies identically to any method scored the same way, so if
baseline outputs become available (see ``--baseline-tag``) the comparison is
like-for-like on the same file set. A restricted score is **not** comparable to
the published 20-file numbers, and every record says so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from ._vendor.score import run_upstream_benchmark

BAND_NOTE_RESTRICTED = (
    "Band-restricted score over the in-band file subset. NOT comparable to the "
    "published 20-file baselines (FCNet/PUNet/Transformer/none), which were "
    "scored over all 20 validation files."
)
BAND_NOTE_FULL = (
    "Full 20-file score. For a band-limited model this includes files whose "
    "injection lies outside the modelled band and which cannot be reconstructed."
)


def in_band_files(survey_path: Path) -> list[str]:
    """File names the P1 survey marks as in band, in order."""
    survey = json.loads(Path(survey_path).read_text())
    files = [f["file"] for f in survey["files"] if f["in_band"]]
    if not files:
        raise SystemExit(f"{survey_path}: no in-band files; nothing to score")
    return files


def out_of_band_files(survey_path: Path) -> list[str]:
    survey = json.loads(Path(survey_path).read_text())
    return [f["file"] for f in survey["files"] if not f["in_band"]]


def _denoised_name(raw_name: str, model_tag: str) -> str:
    """``abra_validation_0007.h5`` -> ``abra_validation_denoised_<tag>_0007.h5``."""
    stem = Path(raw_name).stem                      # abra_validation_0007
    index = stem.rsplit("_", 1)[-1]                 # 0007
    return f"abra_validation_denoised_{model_tag}_{index}.h5"


def _stage_subset(data_dir: Path, model_tag: str, keep: list[str], scratch: Path) -> int:
    """Symlink the denoised outputs for ``keep`` into ``scratch``. Returns the count."""
    scratch.mkdir(parents=True, exist_ok=True)
    staged = 0
    missing = []
    for raw in keep:
        name = _denoised_name(raw, model_tag)
        src = data_dir / name
        if not src.exists():
            missing.append(name)
            continue
        link = scratch / name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(src.resolve(), link)
        staged += 1
    if missing:
        raise SystemExit(
            f"{len(missing)} denoised file(s) missing for tag {model_tag!r}: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )
    return staged


def score(
    upstream_dir: Path,
    data_dir: Path,
    model_tag: str,
    *,
    coarse: bool = True,
    survey: Path | None = None,
) -> dict:
    """Score ``abra_validation_denoised_<model_tag>_00NN.h5`` under ``data_dir``.

    With ``survey`` given, scoring is restricted to the in-band files it lists.
    Returns the captured command/returncode/stdout record.
    """
    if survey is None:
        record = run_upstream_benchmark(upstream_dir, data_dir, model_tag, coarse=coarse)
        record["file_set"] = "all_20"
        record["comparability_note"] = BAND_NOTE_FULL
    else:
        keep = in_band_files(survey)
        dropped = out_of_band_files(survey)
        with tempfile.TemporaryDirectory(prefix=f"tidmad_inband_{model_tag}_") as tmp:
            scratch = Path(tmp)
            staged = _stage_subset(Path(data_dir), model_tag, keep, scratch)
            print(f"scoring {staged} in-band files (dropped {len(dropped)}: {', '.join(dropped)})")
            record = run_upstream_benchmark(upstream_dir, scratch, model_tag, coarse=coarse)
        record["file_set"] = "in_band_subset"
        record["scored_files"] = keep
        record["excluded_files"] = dropped
        record["survey"] = str(survey)
        record["comparability_note"] = BAND_NOTE_RESTRICTED

    print(f"command: {' '.join(record['command'])}")
    print(f"returncode: {record['returncode']}")
    print(record["stdout"])
    if record["stderr"]:
        print(record["stderr"], file=sys.stderr)
    if record["returncode"] != 0:
        raise RuntimeError(f"upstream benchmark.py failed with code {record['returncode']}")
    print(f"\n[file set: {record['file_set']}] {record['comparability_note']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", type=Path, required=True, help="directory containing TIDMAD benchmark.py")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--coarse", action="store_true", default=True)
    parser.add_argument(
        "--in-band-only",
        type=Path,
        default=None,
        metavar="SURVEY_JSON",
        help="restrict to files the P1 survey marks in band (results/rps2026/P1_injection_survey.json)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the record as JSON")
    args = parser.parse_args()

    record = score(
        args.upstream,
        args.data_dir,
        args.model_tag,
        coarse=args.coarse,
        survey=args.in_band_only,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
        print(f"record written to {args.out}")


if __name__ == "__main__":
    main()
