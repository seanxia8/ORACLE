# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Declared N-family acquisition interventions.

Each intervention transforms (photons, geometry, response config) before or
during the response stage, carries its own seed, and serialises its exact
parameters into the provenance record. The N contract is closed: these five
families, at declared severities, and nothing else.

  N1 ModuleLoss      — drop a fraction of OMs (optionally whole strings)
  N2 HitThinning     — drop each signal photon independently
  N3 TimingJitter    — inflate the pulse-time smearing
  N4 GainDrift       — per-string multiplicative charge miscalibration
  N5 NoiseRateScale  — scale the uniform noise-photon rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .events import EventPhotons
from .detector import DetectorGeometry
from .response import ResponseConfig


@dataclass(frozen=True)
class Intervention:
    """Base class. Subclasses override the hooks they need."""

    seed: int = 0

    @property
    def name(self) -> str:
        return type(self).__name__

    def params(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def apply_photons(
        self, photons: EventPhotons, geometry: DetectorGeometry
    ) -> EventPhotons:
        return photons

    def apply_config(self, config: ResponseConfig) -> ResponseConfig:
        return config

    def apply_pulses(self, pulses, geometry: DetectorGeometry):
        return pulses


@dataclass(frozen=True)
class ModuleLoss(Intervention):
    """N1 — remove a fraction of OMs. by_string=True drops whole strings,
    the spatially correlated variant."""

    fraction: float = 0.1
    by_string: bool = False

    def dropped_oms(self, geometry: DetectorGeometry) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        if self.by_string:
            strings = geometry.strings
            n_drop = max(1, round(self.fraction * strings.size))
            dead = rng.choice(strings, size=n_drop, replace=False)
            return np.flatnonzero(np.isin(geometry.string_id, dead))
        n_drop = round(self.fraction * geometry.n_om)
        return rng.choice(geometry.n_om, size=n_drop, replace=False)

    def apply_photons(self, photons, geometry):
        dead = self.dropped_oms(geometry)
        keep = ~np.isin(photons.om_id, dead)
        return EventPhotons(
            photons.event_id, photons.om_id[keep], photons.t_ns[keep],
            photons.is_signal[keep], dict(photons.truth),
        )

    def apply_pulses(self, pulses, geometry):
        # A dead module produces nothing at all — including noise pulses,
        # which the response stage samples uniformly over every OM. Filtering
        # here guarantees the dropped modules are silent end to end.
        dead = self.dropped_oms(geometry)
        keep = ~np.isin(pulses.om_id, dead)
        pulses.om_id = pulses.om_id[keep]
        pulses.t_ns = pulses.t_ns[keep]
        pulses.charge_pe = pulses.charge_pe[keep]
        pulses.signal_fraction = pulses.signal_fraction[keep]
        pulses.meta.setdefault("interventions", []).append(self.name)
        return pulses


@dataclass(frozen=True)
class HitThinning(Intervention):
    """N2 — drop each *signal* photon independently with probability p.
    Matched in expected multiplicity loss to a ModuleLoss level by choosing
    p equal to that level's fraction."""

    p: float = 0.1

    def apply_photons(self, photons, geometry):
        rng = np.random.default_rng(self.seed + photons.event_id)
        drop = photons.is_signal & (rng.random(photons.n_photons) < self.p)
        keep = ~drop
        return EventPhotons(
            photons.event_id, photons.om_id[keep], photons.t_ns[keep],
            photons.is_signal[keep], dict(photons.truth),
        )


@dataclass(frozen=True)
class TimingJitter(Intervention):
    """N3 — inflate the response's 1 ns time smearing to sigma_ns."""

    sigma_ns: float = 10.0

    def apply_config(self, config):
        return config.with_(time_smear_ns=self.sigma_ns)


@dataclass(frozen=True)
class GainDrift(Intervention):
    """N4 — per-string multiplicative charge factor ~ LogNormal(0, sigma)."""

    sigma: float = 0.2

    def apply_pulses(self, pulses, geometry):
        rng = np.random.default_rng(self.seed)
        strings = geometry.strings
        gains = rng.lognormal(mean=0.0, sigma=self.sigma, size=strings.size)
        gain_of = dict(zip(strings.tolist(), gains.tolist()))
        factor = np.array(
            [gain_of[int(geometry.string_id[o])] for o in pulses.om_id]
        )
        pulses.charge_pe = pulses.charge_pe * factor
        pulses.meta.setdefault("interventions", []).append(self.name)
        return pulses


@dataclass(frozen=True)
class NoiseRateScale(Intervention):
    """N5 — scale the uniform noise-photon rate by factor."""

    factor: float = 5.0

    def apply_config(self, config):
        return config.with_(
            noise_rate_per_om_us=config.noise_rate_per_om_us * self.factor
        )


def apply_interventions(
    photons: EventPhotons,
    geometry: DetectorGeometry,
    config: ResponseConfig,
    interventions: list[Intervention],
):
    """Run every hook in declaration order; returns (photons, config, log)."""
    log: list[dict[str, Any]] = []
    for iv in interventions:
        photons = iv.apply_photons(photons, geometry)
        config = iv.apply_config(config)
        log.append({"name": iv.name, **iv.params()})
    return photons, config, log
