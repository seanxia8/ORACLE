# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Generic analytic noise budget for an athermal magnetic calorimeter channel.

This module replaces the tabulated ``data/Al2O3_Al_athermal/*.dat`` reference
curves with the closed-form model they were sampled from. Every component is a
textbook spectral form; the only inputs are a handful of scalar constants.

Why this exists
---------------
The supplied ``.dat`` files are not measurements. They are an analytic noise
budget evaluated on a synthetic grid, which the checks in
``tests/test_reference_budget.py`` demonstrate:

* the frequency grid is 16384 points, geometric from 1 Hz to 30 MHz, with a
  constant ratio to 15 digits;
* ``Johnson_noise.dat`` is exactly constant across all 7.5 decades;
* ``total_noise.dat`` equals the quadrature sum of the four components to
  3e-16, i.e. it was computed rather than measured;
* the thermal corner is 1/(2*pi*1 ms) and the SQUID knee is exactly 100 Hz;
* the signal magnitude is a two-pole response with a 5 us rise and a 1 ms
  decay.

Reproducing the tables therefore needs no third-party file, and the module
gains a configuration that can be retuned for any comparable channel rather
than being pinned to one supplied dataset.

Spectral forms
--------------
All PSDs are one-sided, in (arbitrary readout unit)^2 / Hz; the ``.dat`` files
store the corresponding amplitude spectral density, i.e. ``sqrt(PSD)``.

* Johnson (resistive) noise: white, ``S(f) = S_J``.
* SQUID readout: white floor plus a 1/f tail,
  ``S(f) = S_sq * (1 + f_knee / f)``. This is the standard two-parameter SQUID
  noise model; ``f_knee`` is where the 1/f term equals the white floor.
* Thermodynamic (thermal-carrier) noise: a first-order thermal pole seen in
  power, ``S(f) = S_TD / (1 + (f / f_TD)^2)`` with ``f_TD = 1/(2*pi*tau_TD)``.
* Paramagnetic spin (erbium) noise: ``S(f) = S_Er * (f / 1 Hz)^alpha`` with
  ``alpha`` slightly shallower than -1.
* Total: the incoherent (quadrature) sum of the four.

The detector signal response is the magnitude of a two-pole transfer function,
``|H(f)| = A / sqrt((1 + (f/f_decay)^2) * (1 + (f/f_rise)^2))``, the standard
rise/decay pulse shape for a calorimetric channel.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .spectral_models import CompositeSpectrum, PowerLaw, RollOff, White

__all__ = [
    "AL2O3_AL_ATHERMAL",
    "AthermalNoiseBudget",
    "BudgetGrid",
    "COMPONENT_FILES",
    "write_reference_asd",
]

#: Which ``.dat`` file each named budget channel is written to.
COMPONENT_FILES = {
    "Johnson": "Johnson_noise.dat",
    "SQUID": "SQUID_noise.dat",
    "TD": "TD_noise.dat",
    "Er": "Er_noise.dat",
    "total": "total_noise.dat",
}

SIGNAL_FILE = "signal.dat"

#: ``numpy.savetxt`` format used by the original tables: 18 decimal digits.
_ASD_FMT = "%.18e"


@dataclass(frozen=True)
class BudgetGrid:
    """The frequency grid the reference tables are evaluated on."""

    f_min_hz: float = 1.0
    f_max_hz: float = 3.0e7
    n_points: int = 16384

    def frequencies(self) -> np.ndarray:
        """Geometric grid, inclusive of both endpoints."""
        if self.f_min_hz <= 0.0 or self.f_max_hz <= self.f_min_hz:
            raise ValueError("Require 0 < f_min_hz < f_max_hz.")
        if self.n_points < 2:
            raise ValueError("Require at least two grid points.")
        return np.geomspace(self.f_min_hz, self.f_max_hz, self.n_points)


@dataclass(frozen=True)
class AthermalNoiseBudget:
    """Closed-form noise budget for one athermal calorimeter readout channel.

    All ``*_psd`` constants are one-sided power spectral densities at the
    reference frequency, in squared readout units per hertz.
    """

    name: str

    # Johnson (resistive) noise: white.
    johnson_psd: float

    # SQUID readout: white floor plus 1/f, crossing at ``squid_knee_hz``.
    squid_white_psd: float
    squid_knee_hz: float

    # Thermodynamic noise: single thermal pole with time constant ``thermal_tau_s``.
    thermal_psd: float
    thermal_tau_s: float

    # Paramagnetic spin noise: power law referenced to 1 Hz.
    spin_psd: float
    spin_exponent: float

    # Two-pole detector signal response.
    signal_amplitude: float
    signal_rise_s: float
    signal_decay_s: float

    grid: BudgetGrid = field(default_factory=BudgetGrid)

    # -- derived corner frequencies -------------------------------------

    @property
    def thermal_corner_hz(self) -> float:
        return 1.0 / (2.0 * np.pi * self.thermal_tau_s)

    @property
    def signal_decay_corner_hz(self) -> float:
        return 1.0 / (2.0 * np.pi * self.signal_decay_s)

    @property
    def signal_rise_corner_hz(self) -> float:
        return 1.0 / (2.0 * np.pi * self.signal_rise_s)

    @property
    def squid_one_over_f_psd(self) -> float:
        """Amplitude of the SQUID 1/f term at 1 Hz."""
        return self.squid_white_psd * self.squid_knee_hz

    # -- spectra ---------------------------------------------------------

    def component_psds(self, frequencies: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """One-sided PSD of each named channel, plus their quadrature sum."""
        f = self.grid.frequencies() if frequencies is None else np.asarray(frequencies, float)
        johnson = self.johnson_psd * np.ones_like(f)
        squid = self.squid_white_psd + self.squid_one_over_f_psd * f**-1.0
        thermal = self.thermal_psd / (1.0 + (f / self.thermal_corner_hz) ** 2.0)
        spin = self.spin_psd * f**self.spin_exponent
        out = {"Johnson": johnson, "SQUID": squid, "TD": thermal, "Er": spin}
        out["total"] = johnson + squid + thermal + spin
        return out

    def component_asds(self, frequencies: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Amplitude spectral densities, i.e. the contents of the tables."""
        return {k: np.sqrt(v) for k, v in self.component_psds(frequencies).items()}

    def signal_magnitude(self, frequencies: np.ndarray | None = None) -> np.ndarray:
        """Magnitude of the two-pole detector response."""
        f = self.grid.frequencies() if frequencies is None else np.asarray(frequencies, float)
        return self.signal_amplitude / np.sqrt(
            (1.0 + (f / self.signal_decay_corner_hz) ** 2.0)
            * (1.0 + (f / self.signal_rise_corner_hz) ** 2.0)
        )

    def to_composite(self) -> CompositeSpectrum:
        """The same budget as a :class:`CompositeSpectrum`.

        Component names match ``data/Al2O3_Al_athermal/al2o3_athermal_fit.json``
        so this is interchangeable with :func:`al2o3_athermal.load_composite`.
        """
        return CompositeSpectrum(
            [
                White(scale=self.johnson_psd, name="johnson"),
                White(scale=self.squid_white_psd, name="squid_white"),
                PowerLaw(
                    scale=self.squid_one_over_f_psd,
                    exponent=-1.0,
                    reference_hz=1.0,
                    name="squid_1_f",
                ),
                RollOff(
                    scale=self.thermal_psd,
                    corner_hz=self.thermal_corner_hz,
                    order=2.0,
                    kind="lowpass",
                    name="td",
                ),
                PowerLaw(
                    scale=self.spin_psd,
                    exponent=self.spin_exponent,
                    reference_hz=1.0,
                    name="er",
                ),
            ]
        )

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["derived"] = {
            "thermal_corner_hz": self.thermal_corner_hz,
            "squid_one_over_f_psd_at_1hz": self.squid_one_over_f_psd,
            "signal_decay_corner_hz": self.signal_decay_corner_hz,
            "signal_rise_corner_hz": self.signal_rise_corner_hz,
        }
        return payload

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    def write_reference_asd(self, out_dir: str | Path) -> list[Path]:
        """Regenerate the two-column ``.dat`` reference tables.

        Reproduces the supplied tables to within a few units in the last
        place: the five noise channels agree to <= 4e-16 relative, the signal
        response to <= 1e-15 in its deepest tail. Returns the paths written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        f = self.grid.frequencies()
        written: list[Path] = []
        for channel, filename in COMPONENT_FILES.items():
            asd = self.component_asds(f)[channel]
            target = out_dir / filename
            np.savetxt(target, np.column_stack([f, asd]), fmt=_ASD_FMT)
            written.append(target)
        target = out_dir / SIGNAL_FILE
        np.savetxt(target, np.column_stack([f, self.signal_magnitude(f)]), fmt=_ASD_FMT)
        written.append(target)
        return written


#: Reference budget for the Al2O3/Al athermal channel.
#:
#: The three round constants (SQUID white floor, SQUID knee, signal amplitude)
#: and the two round time constants (1 ms decay, 5 us rise) are design values.
#: The remaining three amplitudes are the constants that reproduce the supplied
#: reference curves; see ``data/Al2O3_Al_athermal/README.md``.
AL2O3_AL_ATHERMAL = AthermalNoiseBudget(
    name="Al2O3_Al_athermal",
    johnson_psd=0.0030307426626081345,
    squid_white_psd=0.09,
    squid_knee_hz=100.0,
    thermal_psd=0.05976157531368565,
    thermal_tau_s=1.0e-3,
    spin_psd=0.5442319404674618,
    spin_exponent=-0.9,
    signal_amplitude=2.0e-3,
    signal_rise_s=5.0e-6,
    signal_decay_s=1.0e-3,
)


def write_reference_asd(
    out_dir: str | Path | None = None,
    budget: AthermalNoiseBudget = AL2O3_AL_ATHERMAL,
) -> list[Path]:
    """Regenerate the reference tables for ``budget`` into ``out_dir``."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "data" / budget.name
    return budget.write_reference_asd(out_dir)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate the analytic reference ASD tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination directory (default: the package data directory).",
    )
    args = parser.parse_args(argv)
    for path in write_reference_asd(args.out_dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
