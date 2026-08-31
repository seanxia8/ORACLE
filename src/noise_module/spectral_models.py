# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Composable one-sided PSD components with explicit physical normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class SpectralComponent:
    """Base PSD component.

    ``normalization='density'`` interprets ``scale`` as a density multiplier.
    ``normalization='power'`` normalizes the component's discrete integral to
    ``scale`` on the requested grid.
    """

    scale: float = 1.0
    normalization: str = "density"
    name: str | None = None

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(
        self, frequencies: np.ndarray, df: float, *, zero_dc: bool = False
    ) -> np.ndarray:
        values = np.clip(np.asarray(self.shape(frequencies), dtype=float), 0.0, None)
        if zero_dc and values.size:
            values[0] = 0.0
        if self.normalization == "density":
            return self.scale * values
        if self.normalization == "power":
            integral = float(np.sum(values) * df)
            if self.scale == 0.0:
                return np.zeros_like(values)
            if integral <= 0.0:
                raise ValueError(f"Component {self.label!r} has zero spectral support.")
            return values * (self.scale / integral)
        raise ValueError("normalization must be 'density' or 'power'.")

    @property
    def label(self) -> str:
        return self.name or self.__class__.__name__.lower()

    def __add__(self, other: "SpectralComponent | CompositeSpectrum") -> "CompositeSpectrum":
        if isinstance(other, CompositeSpectrum):
            return CompositeSpectrum([self, *other.components])
        return CompositeSpectrum([self, other])


@dataclass
class White(SpectralComponent):
    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        return np.ones_like(frequencies, dtype=float)


@dataclass
class PowerLaw(SpectralComponent):
    exponent: float = -1.0
    reference_hz: float = 1.0
    low_cutoff_hz: float | None = None
    high_cutoff_hz: float | None = None

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.exponent) or self.reference_hz <= 0.0:
            raise ValueError("PowerLaw exponent must be finite and reference_hz positive.")
        f = np.asarray(frequencies, dtype=float)
        output = np.zeros_like(f)
        active = f > 0.0
        if self.low_cutoff_hz is not None:
            active &= f >= self.low_cutoff_hz
        if self.high_cutoff_hz is not None:
            active &= f <= self.high_cutoff_hz
        output[active] = (f[active] / self.reference_hz) ** self.exponent
        return output


@dataclass
class Lorentzian(SpectralComponent):
    center_hz: float = 0.0
    half_width_hz: float = 1.0

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if self.center_hz < 0.0 or self.half_width_hz <= 0.0:
            raise ValueError("Lorentzian center must be non-negative and width positive.")
        return 1.0 / (
            1.0 + ((np.asarray(frequencies) - self.center_hz) / self.half_width_hz) ** 2
        )


@dataclass
class Resonance(Lorentzian):
    """Named Lorentzian resonance component."""


@dataclass
class BandLimited(SpectralComponent):
    low_hz: float = 0.0
    high_hz: float = 1.0

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if self.low_hz < 0.0 or self.high_hz < self.low_hz:
            raise ValueError("BandLimited requires 0 <= low_hz <= high_hz.")
        f = np.asarray(frequencies)
        return ((f >= self.low_hz) & (f <= self.high_hz)).astype(float)


@dataclass
class RollOff(SpectralComponent):
    corner_hz: float = 1.0
    order: float = 2.0
    kind: str = "lowpass"

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if self.corner_hz <= 0.0 or self.order <= 0.0:
            raise ValueError("RollOff corner_hz and order must be positive.")
        ratio = np.asarray(frequencies, dtype=float) / self.corner_hz
        if self.kind == "lowpass":
            return 1.0 / (1.0 + ratio**self.order)
        if self.kind == "highpass":
            response = ratio**self.order / (1.0 + ratio**self.order)
            response[0] = 0.0
            return response
        raise ValueError("RollOff kind must be 'lowpass' or 'highpass'.")


@dataclass
class Line(SpectralComponent):
    frequency_hz: float = 1.0
    width_hz: float = 0.0

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        f = np.asarray(frequencies, dtype=float)
        if self.frequency_hz < 0.0 or self.frequency_hz > f[-1]:
            raise ValueError("Line frequency must lie on the one-sided frequency band.")
        if self.width_hz < 0.0:
            raise ValueError("Line width_hz must be non-negative.")
        if self.width_hz == 0.0:
            output = np.zeros_like(f)
            output[int(np.argmin(np.abs(f - self.frequency_hz)))] = 1.0
            return output
        return np.exp(-0.5 * ((f - self.frequency_hz) / self.width_hz) ** 2)


@dataclass
class CompositeSpectrum:
    components: list[SpectralComponent]

    def __init__(self, components: Iterable[SpectralComponent]):
        self.components = list(components)
        if not self.components:
            raise ValueError("CompositeSpectrum requires at least one component.")

    def __add__(self, other: SpectralComponent | "CompositeSpectrum") -> "CompositeSpectrum":
        if isinstance(other, CompositeSpectrum):
            return CompositeSpectrum([*self.components, *other.components])
        return CompositeSpectrum([*self.components, other])

    def evaluate(
        self, frequencies: np.ndarray, df: float, *, zero_dc: bool = False
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        total = np.zeros_like(frequencies, dtype=float)
        metadata = []
        for component in self.components:
            density = component.evaluate(frequencies, df, zero_dc=zero_dc)
            total += density
            metadata.append(
                {
                    "name": component.label,
                    "type": component.__class__.__name__,
                    "integrated_power": float(np.sum(density) * df),
                    "normalization": component.normalization,
                    "scale": float(component.scale),
                }
            )
        return total, metadata


_COMPONENT_TYPES = {
    "white": White,
    "powerlaw": PowerLaw,
    "power_law": PowerLaw,
    "lorentzian": Lorentzian,
    "resonance": Resonance,
    "bandlimited": BandLimited,
    "band_limited": BandLimited,
    "rolloff": RollOff,
    "roll_off": RollOff,
    "line": Line,
}


def component_from_config(config: dict[str, Any]) -> SpectralComponent:
    data = dict(config)
    kind = str(data.pop("type")).lower()
    try:
        cls = _COMPONENT_TYPES[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported spectral component type: {kind}") from exc
    return cls(**data)
