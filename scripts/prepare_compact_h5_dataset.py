#!/usr/bin/env python3
"""Create small, balanced H5 shards suitable for HTCondor file transfer."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/ceph/srv/dwong/training_samples_h5"),
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--energies", default="10,20,50,100,200,500")
    parser.add_argument("--events-per-group", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_source(source: Path, recoil: str, energy: int) -> Path:
    candidates = sorted(
        (source / recoil).glob(f"{recoil}_traces_energy_{energy}_batch_*.h5")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No source shard for recoil={recoil}, energy={energy} under {source}"
        )
    return candidates[0]


def write_subset(source_path: Path, destination_path: Path, count: int) -> None:
    temporary_path = destination_path.with_suffix(".h5.tmp")
    temporary_path.unlink(missing_ok=True)

    with h5py.File(source_path, "r") as source:
        traces = source["traces"]
        events = source["events"]
        if count > len(events):
            raise ValueError(
                f"{source_path} contains {len(events)} events, fewer than requested {count}"
            )

        # Spread the selected events across the source shard instead of taking
        # only its first contiguous block.
        indices = np.linspace(0, len(events) - 1, num=count, dtype=np.int64)
        selected_events = events[indices]

        with h5py.File(temporary_path, "w") as destination:
            for key, value in source.attrs.items():
                destination.attrs[key] = value
            destination.attrs["compact_source"] = str(source_path)
            destination.attrs["compact_source_indices"] = indices
            destination.create_dataset(
                "events",
                data=selected_events,
                dtype=events.dtype,
            )
            output_traces = destination.create_dataset(
                "traces",
                shape=(count, *traces.shape[1:]),
                dtype=traces.dtype,
                chunks=(1, *traces.shape[1:]),
            )
            for output_index, source_index in enumerate(indices):
                output_traces[output_index] = traces[int(source_index)]

    temporary_path.replace(destination_path)


def validate_file(path: Path, expected_events: int) -> None:
    with h5py.File(path, "r") as handle:
        if handle["traces"].shape != (expected_events, 56, 65536):
            raise ValueError(f"Unexpected trace shape in {path}: {handle['traces'].shape}")
        if handle["traces"].dtype != np.float16:
            raise ValueError(f"Unexpected trace dtype in {path}: {handle['traces'].dtype}")
        if len(handle["events"]) != expected_events:
            raise ValueError(f"Unexpected event count in {path}")
    if path.stat().st_size >= 100_000_000:
        raise ValueError(
            f"{path} is {path.stat().st_size} bytes; compact transfer files must be <100 MB"
        )


def main() -> None:
    args = parse_args()
    energies = [int(value) for value in args.energies.split(",")]
    args.destination.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for recoil in ("ER", "NR"):
        recoil_dir = args.destination / recoil
        recoil_dir.mkdir(parents=True, exist_ok=True)
        for energy in energies:
            destination_path = (
                recoil_dir / f"{recoil}_traces_energy_{energy}_batch_0000.h5"
            )
            if args.overwrite or not destination_path.exists():
                source_path = select_source(args.source, recoil, energy)
                print(f"Creating {destination_path} from {source_path}", flush=True)
                write_subset(source_path, destination_path, args.events_per_group)
            validate_file(destination_path, args.events_per_group)
            created.append(destination_path)

    total_bytes = sum(path.stat().st_size for path in created)
    print(
        "COMPACT_DATASET_OK "
        f"files={len(created)} "
        f"events={len(created) * args.events_per_group} "
        f"bytes={total_bytes} "
        f"max_file_bytes={max(path.stat().st_size for path in created)}"
    )


if __name__ == "__main__":
    main()
