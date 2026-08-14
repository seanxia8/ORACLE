"""Deterministic chunked generation and lightweight performance benchmarks."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .NoiseGenerator import NoiseGenerator


@dataclass
class StreamingNoiseGenerator:
    """Generate buffered fixed-grid blocks with a persistent RNG stream.

    Every underlying FFT realization has exactly ``chunk_samples`` samples.
    Short requests consume a buffer rather than constructing a different FFT
    grid, so a final partial chunk has the same finite-grid spectral model as
    earlier samples. Correlations still do not cross underlying block
    boundaries.
    """

    config: dict
    chunk_samples: int
    seed: int | None = None

    def __post_init__(self):
        if self.chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive.")
        self.generator = NoiseGenerator(self.config, seed=self.seed)
        self.chunks_generated = 0
        self.blocks_generated = 0
        self._buffer = np.array([], dtype=float)

    def next_chunk(self, samples: int | None = None) -> np.ndarray:
        size = self.chunk_samples if samples is None else int(samples)
        if size <= 0 or size > self.chunk_samples:
            raise ValueError("samples must lie in (0, chunk_samples].")
        pieces = []
        needed = size
        while needed:
            if self._buffer.size == 0:
                self._buffer = self.generator.generate_noise(self.chunk_samples)
                self.blocks_generated += 1
            take = min(needed, self._buffer.size)
            pieces.append(self._buffer[:take])
            self._buffer = self._buffer[take:]
            needed -= take
        result = np.concatenate(pieces)
        self.chunks_generated += 1
        return result

    def iter_chunks(self, total_samples: int) -> Iterator[np.ndarray]:
        remaining = int(total_samples)
        if remaining < 0:
            raise ValueError("total_samples must be non-negative.")
        while remaining:
            size = min(remaining, self.chunk_samples)
            yield self.next_chunk(size)
            remaining -= size

    def generate(self, total_samples: int) -> np.ndarray:
        chunks = list(self.iter_chunks(total_samples))
        return np.concatenate(chunks) if chunks else np.array([], dtype=float)

    @property
    def contract(self) -> dict:
        return {
            "reproducibility": "exact for identical seed, chunk size, and call sequence",
            "one_shot_equality": False,
            "underlying_block_samples": self.chunk_samples,
            "partial_request_policy": "buffer_from_fixed_grid_block",
            "cross_block_correlation": "not modeled",
            "chunks_generated": self.chunks_generated,
            "blocks_generated": self.blocks_generated,
            "buffered_samples": int(self._buffer.size),
        }


def benchmark_generation(
    config: dict,
    *,
    realizations: int = 32,
    channels: int = 1,
    samples: int = 4096,
    seed: int = 0,
    minimum_samples_per_second: float = 1e4,
    maximum_peak_memory_bytes: int | None = None,
) -> dict:
    """Return runtime, throughput, and peak-memory metrics for a representative shape."""
    if min(realizations, channels, samples) <= 0:
        raise ValueError("benchmark dimensions must be positive.")
    generator = NoiseGenerator(config, seed=seed)
    tracemalloc.start()
    start = time.perf_counter()
    blocks = [generator.generate_ensemble(realizations, samples) for _ in range(channels)]
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    values = realizations * channels * samples
    throughput = values / max(elapsed, 1e-12)
    passed = throughput >= minimum_samples_per_second and (
        maximum_peak_memory_bytes is None or peak <= maximum_peak_memory_bytes
    )
    return {
        "shape": [realizations, channels, samples],
        "elapsed_seconds": elapsed,
        "samples_per_second": throughput,
        "peak_memory_bytes": peak,
        "output_bytes": sum(block.nbytes for block in blocks),
        "minimum_samples_per_second": minimum_samples_per_second,
        "maximum_peak_memory_bytes": maximum_peak_memory_bytes,
        "passed": passed,
    }
