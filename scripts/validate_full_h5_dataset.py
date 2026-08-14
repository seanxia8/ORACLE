#!/usr/bin/env python3
"""Validate the full six-energy ER/NR H5 training dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np

FILENAME_RE = re.compile(
    r"(?P<recoil>ER|NR)_traces_energy_(?P<energy>\d+)_batch_(?P<batch>\d+)\.h5$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--energies", default="10,20,50,100,200,500")
    parser.add_argument("--files-per-energy-recoil", type=int, default=250)
    parser.add_argument("--events-per-file", type=int, default=100)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Validate one shard per energy/recoil after checking the full manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    energies = [int(value) for value in args.energies.split(",")]
    expected_files = len(energies) * 2 * args.files_per_energy_recoil
    selected: list[Path] = []

    for recoil in ("ER", "NR"):
        recoil_dir = args.root / recoil
        if not recoil_dir.is_dir():
            raise FileNotFoundError(f"Missing recoil directory: {recoil_dir}")
        for energy in energies:
            files = sorted(
                recoil_dir.glob(f"{recoil}_traces_energy_{energy}_batch_*.h5")
            )
            if len(files) != args.files_per_energy_recoil:
                raise ValueError(
                    f"{recoil} energy {energy}: found {len(files)} shards, "
                    f"expected {args.files_per_energy_recoil}"
                )
            batch_ids = []
            for path in files:
                match = FILENAME_RE.fullmatch(path.name)
                if match is None:
                    raise ValueError(f"Unexpected shard name: {path}")
                batch_ids.append(int(match.group("batch")))
            expected_ids = list(range(args.files_per_energy_recoil))
            if batch_ids != expected_ids:
                raise ValueError(
                    f"{recoil} energy {energy}: batch IDs are not 0.."
                    f"{args.files_per_energy_recoil - 1}"
                )
            selected.extend(files[:1] if args.sample_only else files)

    if len(selected) not in {expected_files, len(energies) * 2}:
        raise AssertionError("Internal manifest validation error")

    total_events = 0
    for index, path in enumerate(selected, start=1):
        with h5py.File(path, "r") as handle:
            if "traces" not in handle or "events" not in handle:
                raise ValueError(f"{path}: missing traces or events dataset")
            traces = handle["traces"]
            events = handle["events"]
            if traces.dtype != np.float16:
                raise ValueError(f"{path}: traces dtype {traces.dtype}, expected float16")
            expected_shape = (args.events_per_file, 56, 65536)
            if traces.shape != expected_shape:
                raise ValueError(
                    f"{path}: traces shape {traces.shape}, expected {expected_shape}"
                )
            if len(events) != args.events_per_file:
                raise ValueError(
                    f"{path}: {len(events)} metadata rows, "
                    f"expected {args.events_per_file}"
                )
            total_events += len(events)
        if index % 250 == 0:
            print(f"Validated {index}/{len(selected)} shards", flush=True)

    mode = "sampled" if args.sample_only else "full"
    print(
        f"VALIDATION_OK mode={mode} manifest_files={expected_files} "
        f"validated_files={len(selected)} validated_events={total_events}"
    )


if __name__ == "__main__":
    main()
