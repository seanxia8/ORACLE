"""Single-channel stationary spectral noise generator."""

from __future__ import annotations

import os
import warnings
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np

from scipy.fft import irfft, rfftfreq
from scipy.linalg import toeplitz

from .config import CONFIG_SCHEMA_VERSION, NoiseConfig
from .spectral_models import CompositeSpectrum, component_from_config
from .utils import resolve_rng


class NoiseGenerator:
    """Generate stationary single-channel Gaussian noise from a target PSD.

    FFT synthesis defines a periodic finite-grid process whose covariance is
    circulant on the generated grid. Use ``sample_from_covariance`` when an
    exact finite Toeplitz covariance vector, rather than a PSD-driven periodic
    realization, is the primary contract.

    Public unit convention:

    * ``build_psd_density`` returns a one-sided PSD density in ADC^2 / Hz.
    * ``build_rfft_power`` converts that density to expected rFFT-bin power
      for NumPy's unnormalised ``rfft`` / ``irfft`` convention.
    * ``build_psd`` is kept as a backward-compatible alias for
      ``build_rfft_power`` because earlier notebooks used that name for the
      synthesis-domain quantity.
    """

    def __init__(
        self,
        config: dict[str, Any] | NoiseConfig,
        rng: Any = None,
        seed: int | None = None,
        *,
        strict_config: bool = True,
    ):
        self.seed = seed
        self.rng = resolve_rng(rng=rng, seed=seed)
        self.config: dict[str, Any] = {}
        self._noise_path: str | None = None
        self._set_spectra()
        self.set_config(config, strict=strict_config)

    def set_config(
        self, config: dict[str, Any] | NoiseConfig, *, strict: bool = True
    ) -> None:
        """Validate and store a new generator configuration."""
        raw = config.to_dict() if isinstance(config, NoiseConfig) else dict(config)
        expected_fields = ["noise_type", "noise_power", "sampling_frequency"]
        missing = [field for field in expected_fields if field not in raw]
        if missing:
            raise RuntimeError(
                f"Configuration missing required field(s): {', '.join(missing)}."
            )

        model = NoiseConfig.from_mapping(raw, strict=strict)
        self.config_model = model
        self.config = model.to_dict()
        if isinstance(model.noise_type, CompositeSpectrum):
            self.config["noise_type"] = "composite"
            self.config["components"] = [
                {"type": item.__class__.__name__.lower(), **asdict(item)}
                for item in model.noise_type.components
            ]
        self.power_definition = model.power_definition
        self.custom_psd_scaling = model.custom_psd_scaling
        self.psd_scale = model.psd_scale
        self.psd_exponent = model.psd_exponent
        self.deterministic_mean = model.deterministic_mean
        self.custom_out_of_band = model.custom_out_of_band
        self.custom_interpolation = model.custom_interpolation
        self.low_frequency_cutoff = model.low_frequency_cutoff
        self.high_frequency_cutoff = model.high_frequency_cutoff
        self.composite_psd_scaling = model.composite_psd_scaling
        self.sampling_frequency = model.sampling_frequency
        self.set_noise_power(model.noise_power)
        self.set_noise_type(model.noise_type, components=model.components)

    def set_noise_type(
        self, noise_type: str | CompositeSpectrum, components: list[dict[str, Any]] | None = None
    ) -> None:
        """Set the analytic noise colour or load a custom PSD file."""
        analytic_types = "white blue violet brownian pink".split()
        self.composite_spectrum: CompositeSpectrum | None = None
        if isinstance(noise_type, CompositeSpectrum):
            self.noise_type = "composite"
            self.composite_spectrum = noise_type
            self._noise_path = None
        elif isinstance(noise_type, str) and noise_type.lower() == "composite":
            if not components:
                raise ValueError("Composite noise requires at least one component.")
            self.noise_type = "composite"
            self.composite_spectrum = CompositeSpectrum(
                [component_from_config(item) for item in components]
            )
            self._noise_path = None
        elif isinstance(noise_type, str) and noise_type.lower() in analytic_types:
            self.noise_type = noise_type.lower()
            self._noise_path = None
        elif os.path.isfile(str(noise_type)):
            self._noise_path = os.path.abspath(str(noise_type))
            self.noise_type = "custom"
            self._load_psd()
        else:
            raise RuntimeError(
                f"Configuration noise_type field {noise_type} is neither a supported "
                "noise type nor a PSD file path."
            )
        if self.noise_type != "composite":
            self.spectrum = self._spectra[self.noise_type]

    def set_noise_power(self, noise_power: float) -> None:
        """Set the target integrated noise power."""
        self.psd_area = float(noise_power)
        if not np.isfinite(self.psd_area) or self.psd_area < 0.0:
            raise ValueError("noise_power must be finite and non-negative.")

    def _set_spectra(self) -> None:
        self._spectra = {
            "white": lambda f: np.ones_like(f, dtype=float),
            "blue": lambda f: f,
            "violet": lambda f: f**2,
            "brownian": lambda f: 1.0 / np.where(f == 0, np.inf, f**2),
            "pink": lambda f: 1.0 / np.where(f == 0, np.inf, f),
        }
        self._normalize = {
            "white": self._normalize_white,
            "blue": self._normalize_blue,
            "violet": self._normalize_violet,
            "brownian": self._normalize_brownian,
            "pink": self._normalize_pink,
        }

    def _load_psd(self) -> None:
        if self._noise_path is None:
            raise RuntimeError("Custom PSD path is not set.")
        self.noise_psd_data = np.load(self._noise_path)
        if (
            self.noise_psd_data.ndim != 2
            or self.noise_psd_data.shape[0] != 2
            or self.noise_psd_data.shape[1] < 2
        ):
            raise ValueError("A custom PSD file must contain a 2 x F array.")
        frequencies, density = self.noise_psd_data
        if (
            np.any(~np.isfinite(frequencies))
            or np.any(~np.isfinite(density))
            or np.any(np.diff(frequencies) <= 0.0)
            or np.any(density < 0.0)
        ):
            raise ValueError(
                "Custom PSD frequencies must increase strictly and densities must "
                "be finite and non-negative."
            )
        self._spectra["custom"] = self._interpolate_custom_psd

    def build_psd_density(
        self, N: int, return_metadata: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return the one-sided PSD density in ADC^2 / Hz.

        ``noise_power`` is the discrete integral of the one-sided PSD over the
        exact rFFT grid. With the default ``power_definition='variance'``, DC is
        forced to zero so this integral is expected variance rather than random
        mean-square power.

        Custom PSD handling is explicit: ``absolute`` preserves file units,
        ``normalize`` treats the file as a shape and normalizes its integral to
        ``noise_power``, and ``scale`` scales the file density by
        ``psd_scale``.
        """
        if N <= 0:
            raise ValueError("N must be positive.")
        if N == 1 and self.power_definition == "variance" and self.psd_area > 0.0:
            raise ValueError(
                "N=1 cannot represent positive variance when the DC bin is zero."
            )

        frequencies = rfftfreq(N, d=1.0 / self.sampling_frequency)
        component_metadata: list[dict[str, Any]] = []
        if self.noise_type == "custom":
            density = np.clip(
                np.asarray(self.spectrum(frequencies), dtype=float),
                a_min=0.0,
                a_max=None,
            )
            if self.power_definition == "variance":
                density[0] = 0.0
            if self.custom_psd_scaling == "normalize":
                density = self._normalize_shape_to_power(density, N)
            elif self.custom_psd_scaling == "scale":
                density = density * self.psd_scale
        elif self.noise_type == "composite":
            if self.composite_spectrum is None:
                raise RuntimeError("Composite spectrum was not initialized.")
            density, component_metadata = self.composite_spectrum.evaluate(
                frequencies,
                self.sampling_frequency / N,
                zero_dc=self.power_definition == "variance",
            )
            density = self._apply_frequency_cutoffs(density, frequencies)
            if self.composite_psd_scaling == "normalize":
                before = float(np.sum(density) * self.sampling_frequency / N)
                density = self._normalize_shape_to_power(density, N)
                factor = 0.0 if before == 0.0 else self.psd_area / before
                for item in component_metadata:
                    item["integrated_power_after_global_scaling"] = (
                        item["integrated_power"] * factor
                    )
        else:
            if self.psd_exponent is None:
                shape = np.asarray(self.spectrum(frequencies), dtype=float)
            else:
                shape = self._power_law_shape(frequencies, self.psd_exponent)
            if self.power_definition == "variance":
                shape[0] = 0.0
            shape = self._apply_frequency_cutoffs(shape, frequencies)
            density = self._normalize_shape_to_power(shape, N)

        if not return_metadata:
            return frequencies, density

        df = self.sampling_frequency / int(N)
        density_integral = float(np.sum(density) * df)
        dc_power = float(density[0] * df)
        expected_variance = density_integral - dc_power
        metadata = {
            "metadata_schema_version": CONFIG_SCHEMA_VERSION,
            "noise_type": self.noise_type,
            "requested_noise_power": self.psd_area,
            "noise_power": self.psd_area,  # compatibility alias
            "power_definition": self.power_definition,
            "custom_psd_scaling": (
                self.custom_psd_scaling if self.noise_type == "custom" else None
            ),
            "psd_exponent": self.psd_exponent,
            "deterministic_mean": self.deterministic_mean,
            "low_frequency_cutoff": self.low_frequency_cutoff,
            "high_frequency_cutoff": self.high_frequency_cutoff,
            "sampling_frequency": self.sampling_frequency,
            "seed": self.seed,
            "n_samples": N,
            "units": "ADC^2/Hz",
            "integration_rule": "sum(one_sided_psd_density) * (fs / N)",
            "finite_grid_covariance": "periodic/circulant implied by the DFT grid",
            "density_integral": density_integral,
            "expected_variance": expected_variance,
            "stochastic_dc_power": dc_power,
            "expected_mean_square": density_integral + self.deterministic_mean**2,
            "component_contributions": component_metadata,
            "config": deepcopy(self.config),
        }
        if self.noise_type == "custom":
            metadata["custom_psd_source"] = {
                "path": self._noise_path,
                "frequency_range_hz": [
                    float(self.noise_psd_data[0, 0]),
                    float(self.noise_psd_data[0, -1]),
                ],
                "interpolation": self.custom_interpolation,
                "out_of_band": self.custom_out_of_band,
                "scaling": self.custom_psd_scaling,
                "psd_scale": self.psd_scale,
                "resulting_integral": density_integral,
            }
        return frequencies, density, metadata

    def build_rfft_power(
        self, N: int, return_metadata: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return expected rFFT coefficient power used for synthesis."""
        if return_metadata:
            frequencies, density, density_meta = self.build_psd_density(
                N,
                return_metadata=True,
            )
            power = self.psd_density_to_rfft_power(
                density,
                self.sampling_frequency,
                N,
            )
            metadata = {
                **density_meta,
                "units": "ADC^2 rFFT-bin power",
                "psd_density_units": "ADC^2/Hz",
                "rfft_power_total": float(np.sum(power)),
            }
            return frequencies, power, metadata

        frequencies, density = self.build_psd_density(N)
        power = self.psd_density_to_rfft_power(density, self.sampling_frequency, N)
        return frequencies, power

    def build_psd(
        self, N: int, return_metadata: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Backward-compatible alias for ``build_rfft_power``.

        The returned second array is not a density; use ``build_psd_density``
        when a physical PSD in ADC^2 / Hz is needed.
        """
        warnings.warn(
            "build_psd() is deprecated because it returns rFFT-bin power, not "
            "a PSD density; use build_psd_density() or build_rfft_power().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.build_rfft_power(N, return_metadata=return_metadata)

    def sample_stationary_gaussian_from_rfft_power(
        self,
        rfft_power: np.ndarray,
        N: int | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Sample a stationary Gaussian time series from rFFT-bin power."""
        rfft_power = np.asarray(rfft_power, dtype=float)
        if rfft_power.ndim != 1:
            raise ValueError("rfft_power must be a one-dimensional array.")
        if N is None:
            N = 2 * (len(rfft_power) - 1)
        if N <= 0:
            raise ValueError("N must be positive.")
        if len(rfft_power) != N // 2 + 1:
            raise ValueError("rfft_power length does not match N.")
        if np.any(~np.isfinite(rfft_power)) or np.any(rfft_power < 0.0):
            raise ValueError("rfft_power must contain finite, non-negative values.")

        amplitude = np.sqrt(rfft_power)
        spectrum = np.zeros_like(amplitude, dtype=complex)

        # True complex-Gaussian bins (E|eta_k|^2 = psd_k, Rayleigh modulus).
        # The previous implementation used CONSTANT-modulus random-phase bins
        # (eta_k = sqrt(psd_k) e^{i phi}): correct second-order statistics,
        # but every trace then has identical per-bin power, so any
        # distributional statistic built from weighted residual energies
        # (chi^2 goodness-of-fit, KS residual-whiteness tests) is degenerate
        # and meaningless. Gaussian bins are required for those tests and
        # match the Gaussian-ML assumptions of the framework.
        if len(amplitude) > 0:
            spectrum[0] = amplitude[0] * self.rng.standard_normal()
        if len(amplitude) > 2:
            re = self.rng.standard_normal(len(amplitude) - 2)
            im = self.rng.standard_normal(len(amplitude) - 2)
            spectrum[1:-1] = amplitude[1:-1] * (re + 1j * im) / np.sqrt(2.0)
        if len(amplitude) > 1:
            if N % 2 == 0:
                spectrum[-1] = amplitude[-1] * self.rng.standard_normal()
            else:
                re, im = self.rng.standard_normal(2)
                spectrum[-1] = amplitude[-1] * (re + 1j * im) / np.sqrt(2.0)

        signal = irfft(spectrum, n=N) + self.deterministic_mean
        if not return_metadata:
            return signal

        metadata = {
            "n_samples": N,
            "variance": float(np.var(signal)),
            "mean": float(np.mean(signal)),
            "mean_square": float(np.mean(signal**2)),
        }
        return signal, metadata

    def sample_stationary_gaussian_from_psd(
        self,
        psd: np.ndarray,
        N: int | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Backward-compatible wrapper for rFFT-bin power input.

        Earlier code used ``psd`` to mean expected rFFT coefficient power.
        New code should call ``sample_stationary_gaussian_from_rfft_power`` or
        pass physical densities through ``psd_density_to_rfft_power`` first.
        """
        return self.sample_stationary_gaussian_from_rfft_power(
            psd,
            N=N,
            return_metadata=return_metadata,
        )

    def generate_noise(
        self, N: int, return_metadata: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Convenience wrapper combining PSD construction and sampling."""
        if return_metadata:
            frequencies, density, density_meta = self.build_psd_density(
                N,
                return_metadata=True,
            )
            rfft_power = self.psd_density_to_rfft_power(
                density,
                self.sampling_frequency,
                N,
            )
            signal, sample_meta = self.sample_stationary_gaussian_from_rfft_power(
                rfft_power,
                N=N,
                return_metadata=True,
            )
            metadata = {
                **density_meta,
                **sample_meta,
                "frequencies": frequencies,
                "psd_density": density,
            }
            return signal, metadata

        _, rfft_power = self.build_rfft_power(N)
        return self.sample_stationary_gaussian_from_rfft_power(rfft_power, N=N)

    def generate_ensemble(
        self, n_realizations: int, N: int, return_metadata: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Vectorized generation with output shape (realizations, samples)."""
        if n_realizations <= 0:
            raise ValueError("n_realizations must be positive.")
        frequencies, density = self.build_psd_density(N)
        power = self.psd_density_to_rfft_power(density, self.sampling_frequency, N)
        amplitude = np.sqrt(power)
        F = len(amplitude)
        spectrum = np.zeros((n_realizations, F), dtype=complex)
        spectrum[:, 0] = amplitude[0] * self.rng.standard_normal(n_realizations)
        upper = N // 2 + 1 - (N + 1) % 2
        if upper > 1:
            z = (
                self.rng.standard_normal((n_realizations, upper - 1))
                + 1j * self.rng.standard_normal((n_realizations, upper - 1))
            ) / np.sqrt(2)
            spectrum[:, 1:upper] = amplitude[None, 1:upper] * z
        if N % 2 == 0 and F > 1:
            spectrum[:, -1] = amplitude[-1] * self.rng.standard_normal(n_realizations)
        output = irfft(spectrum, n=N, axis=1) + self.deterministic_mean
        if return_metadata:
            return output, {
                "n_realizations": n_realizations,
                "n_samples": N,
                "output_shape": output.shape,
                "vectorized": True,
                "frequencies": frequencies,
                "psd_density": density,
            }
        return output

    def _normalize_shape_to_power(self, shape: np.ndarray, N: int) -> np.ndarray:
        """Scale a non-negative one-sided shape to the configured noise power."""
        values = np.asarray(shape, dtype=float)
        values = np.clip(values, a_min=0.0, a_max=None)
        total = float(np.sum(values))
        if self.psd_area == 0.0:
            return np.zeros_like(values)
        if total <= 0.0:
            raise ValueError(f"Noise type {self.noise_type!r} has zero PSD support.")
        df = self.sampling_frequency / int(N)
        return values * (self.psd_area / (total * df))

    @staticmethod
    def _power_law_shape(frequencies: np.ndarray, exponent: float) -> np.ndarray:
        """Return an f**exponent shape with a finite, zero-valued DC bin."""
        if not np.isfinite(exponent):
            raise ValueError("psd_exponent must be finite.")
        frequencies = np.asarray(frequencies, dtype=float)
        shape = np.zeros_like(frequencies)
        positive = frequencies > 0.0
        shape[positive] = frequencies[positive] ** exponent
        return shape

    def _apply_frequency_cutoffs(
        self, values: np.ndarray, frequencies: np.ndarray
    ) -> np.ndarray:
        output = np.array(values, dtype=float, copy=True)
        if self.low_frequency_cutoff is not None:
            output[frequencies < self.low_frequency_cutoff] = 0.0
        if self.high_frequency_cutoff is not None:
            output[frequencies > self.high_frequency_cutoff] = 0.0
        return output

    def _interpolate_custom_psd(self, frequencies: np.ndarray) -> np.ndarray:
        source_f, source_p = self.noise_psd_data
        target = np.asarray(frequencies, dtype=float)
        outside = (target < source_f[0]) | (target > source_f[-1])
        if np.any(outside) and self.custom_out_of_band == "error":
            raise ValueError("Target frequency grid exceeds custom PSD support.")

        if self.custom_interpolation == "linear":
            result = np.interp(target, source_f, source_p)
        else:
            valid = (source_f > 0.0) & (source_p > 0.0)
            if np.sum(valid) < 2:
                raise ValueError("loglog interpolation requires two positive PSD points.")
            result = np.exp(
                np.interp(
                    np.log(np.maximum(target, np.finfo(float).tiny)),
                    np.log(source_f[valid]),
                    np.log(source_p[valid]),
                )
            )
            if source_f[0] == 0.0:
                result[target == 0.0] = source_p[0]

        if self.custom_out_of_band == "zero":
            result[outside] = 0.0
        elif self.custom_out_of_band == "edge":
            result[target < source_f[0]] = source_p[0]
            result[target > source_f[-1]] = source_p[-1]
        elif self.custom_out_of_band == "power_law":
            result = self._power_law_extrapolate(target, result, source_f, source_p)
        return result

    @staticmethod
    def _power_law_extrapolate(
        target: np.ndarray,
        result: np.ndarray,
        source_f: np.ndarray,
        source_p: np.ndarray,
    ) -> np.ndarray:
        output = np.array(result, copy=True)
        positive = (source_f > 0.0) & (source_p > 0.0)
        indices = np.flatnonzero(positive)
        if indices.size < 2:
            raise ValueError("power_law extrapolation requires two positive PSD points.")
        lo0, lo1 = indices[:2]
        hi0, hi1 = indices[-2:]
        low_slope = np.log(source_p[lo1] / source_p[lo0]) / np.log(
            source_f[lo1] / source_f[lo0]
        )
        high_slope = np.log(source_p[hi1] / source_p[hi0]) / np.log(
            source_f[hi1] / source_f[hi0]
        )
        low = (target < source_f[0]) & (target > 0.0)
        high = target > source_f[-1]
        output[low] = source_p[lo0] * (target[low] / source_f[lo0]) ** low_slope
        output[high] = source_p[hi1] * (target[high] / source_f[hi1]) ** high_slope
        if source_f[0] == 0.0:
            output[target == 0.0] = source_p[0]
        return output

    def sample_from_covariance(
        self,
        covariance: np.ndarray,
        *,
        method: str = "auto",
        return_metadata: bool = False,
        tolerance: float = 1e-12,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Sample an exact finite Gaussian vector from a Toeplitz covariance.

        ``covariance`` is the first row of the requested Toeplitz covariance.
        Circulant embedding is exact for the retained vector when its embedding
        eigenvalues are non-negative. ``method='auto'`` falls back to a dense
        eigendecomposition when that condition is not satisfied.
        """
        first_row = np.asarray(covariance, dtype=float)
        if first_row.ndim != 1 or first_row.size == 0:
            raise ValueError("covariance must be a non-empty one-dimensional array.")
        if np.any(~np.isfinite(first_row)) or first_row[0] < 0.0:
            raise ValueError("covariance must be finite with non-negative variance.")
        if method not in {"auto", "circulant", "dense"}:
            raise ValueError("method must be 'auto', 'circulant', or 'dense'.")

        N = first_row.size
        embedding_row = (
            first_row.copy()
            if N == 1
            else np.concatenate([first_row, first_row[-2:0:-1]])
        )
        eigenvalues = np.fft.fft(embedding_row).real
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        minimum = float(np.min(eigenvalues))
        used = method
        if method in {"auto", "circulant"} and minimum >= -tolerance * scale:
            eigenvalues = np.clip(eigenvalues, 0.0, None)
            z = self.rng.standard_normal(embedding_row.size)
            full = np.fft.ifft(np.sqrt(eigenvalues) * np.fft.fft(z)).real
            sample = full[:N]
            used = "circulant"
            implied_first_row = np.fft.ifft(eigenvalues).real[:N]
        elif method == "circulant":
            raise ValueError(
                f"Circulant embedding is not positive semidefinite; minimum "
                f"eigenvalue is {minimum:.6g}."
            )
        else:
            matrix = toeplitz(first_row)
            values, vectors = np.linalg.eigh(matrix)
            if np.min(values) < -tolerance * max(float(np.max(np.abs(values))), 1.0):
                raise ValueError("Requested Toeplitz covariance is not positive semidefinite.")
            sample = vectors @ (
                np.sqrt(np.clip(values, 0.0, None))
                * self.rng.standard_normal(N)
            )
            used = "dense"
            reconstructed = (vectors * np.clip(values, 0.0, None)[None, :]) @ vectors.T
            implied_first_row = reconstructed[0]
        sample = sample + self.deterministic_mean
        metadata = {
            "method": used,
            "n_samples": N,
            "minimum_embedding_eigenvalue": minimum,
            "exact_finite_covariance": True,
            "finite_covariance_model": (
                "leading principal block of a circulant covariance"
                if used == "circulant"
                else "explicit Toeplitz covariance eigendecomposition"
            ),
            "maximum_covariance_error": float(
                np.max(np.abs(implied_first_row - first_row))
            ),
            "requested_covariance_first_row": first_row.copy(),
        }
        if return_metadata:
            return sample, metadata
        return sample

    @staticmethod
    def psd_density_to_rfft_power(
        psd_density: np.ndarray,
        sampling_frequency: float,
        N: int,
    ) -> np.ndarray:
        """Convert one-sided PSD density to expected unnormalised rFFT power."""
        if N <= 0:
            raise ValueError("N must be positive.")
        density = np.asarray(psd_density, dtype=float)
        if density.ndim != 1:
            raise ValueError("psd_density must be one-dimensional.")
        expected_bins = int(N) // 2 + 1
        if density.shape[0] != expected_bins:
            raise ValueError(
                f"PSD density length {density.shape[0]} does not match rFFT bins {expected_bins}."
            )
        power = np.clip(density, a_min=0.0, a_max=None) * float(sampling_frequency) * int(N)
        if int(N) > 2:
            upper = int(N) // 2 + 1 - (int(N) + 1) % 2
            power[1:upper] *= 0.5
        return power

    @staticmethod
    def _normalize_white(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        return 1.0 / max(float(np.max(frequencies) - np.min(frequencies)), np.finfo(float).eps)

    @staticmethod
    def _normalize_blue(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        denom = float(np.max(frequencies) ** 2 - np.min(frequencies) ** 2)
        return 2.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_violet(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        denom = float(np.max(frequencies) ** 3 - np.min(frequencies) ** 3)
        return 3.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_brownian(frequencies: np.ndarray) -> float:
        positive = np.sort(frequencies[frequencies > 0])
        if len(positive) == 0:
            return 1.0
        if len(positive) == 1:
            return float(positive[0])
        denom = float(1.0 / positive[0] - 1.0 / positive[-1])
        return 1.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_pink(frequencies: np.ndarray) -> float:
        positive = np.sort(frequencies[frequencies > 0])
        if len(positive) < 2:
            return 1.0
        denom = float(np.log(positive[-1]) - np.log(positive[0]))
        return 1.0 / max(denom, np.finfo(float).eps)
