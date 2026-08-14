"""First-class non-Gaussian innovation and colored-noise models."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.stats import kurtosis, levy_stable, norm, skew

from .NoiseGenerator import NoiseGenerator
from .utils import resolve_rng


class NonGaussianNoiseGenerator:
    """Generate iid or spectrally colored non-Gaussian noise.

    Coloring a non-Gaussian innovation sequence is a linear mixture and does
    not, in general, preserve its named marginal distribution. Metadata reports
    the innovation family and whether coloring was applied separately.
    """

    def __init__(self, config: dict[str, Any], rng: Any = None, seed: int | None = None):
        self.config = dict(config)
        self.rng = resolve_rng(rng=rng, seed=seed)

    def generate(self, N: int, *, return_metadata: bool = False):
        if N <= 0:
            raise ValueError("N must be positive.")
        family = str(self.config.get("family", "student_t")).lower()
        power = float(self.config.get("noise_power", 1.0))
        if power < 0 or not np.isfinite(power):
            raise ValueError("noise_power must be finite and non-negative.")
        meaning = "variance"
        if family == "student_t":
            df = float(self.config.get("degrees_of_freedom", 5.0))
            if df <= 2:
                raise ValueError("Student-t degrees_of_freedom must exceed 2 for finite variance.")
            x = self.rng.standard_t(df, N) * np.sqrt(power * (df - 2) / df)
        elif family == "laplace":
            x = self.rng.laplace(0.0, np.sqrt(power / 2.0), N)
        elif family in {"gaussian_scale_mixture", "scale_mixture"}:
            probability = float(self.config.get("contamination_probability", 0.05))
            ratio = float(self.config.get("contamination_scale", 5.0))
            scales = np.where(self.rng.random(N) < probability, ratio, 1.0)
            x = self.rng.standard_normal(N) * scales
            x *= np.sqrt(power / np.mean(scales**2))
        elif family in {"compound_poisson", "shot_noise"}:
            rate = float(self.config.get("event_probability", 0.01))
            counts = self.rng.poisson(rate, N)
            x = self.rng.normal(0.0, 1.0, N) * np.sqrt(counts)
            x *= np.sqrt(power / max(rate, np.finfo(float).eps))
        elif family in {"alpha_stable", "stable"}:
            alpha = float(self.config.get("alpha", 1.7))
            beta = float(self.config.get("beta", 0.0))
            warnings.warn(
                "Alpha-stable noise with alpha < 2 has undefined variance; "
                "noise_power is interpreted as scale^2.",
                RuntimeWarning,
                stacklevel=2,
            )
            x = levy_stable.rvs(
                alpha, beta, scale=np.sqrt(power), size=N, random_state=self.rng
            )
            meaning = "scale_squared; variance undefined"
        else:
            raise ValueError(f"Unsupported non-Gaussian family: {family}")

        colored = False
        psd_config = self.config.get("psd_config")
        if psd_config is not None:
            target = NoiseGenerator(psd_config, rng=self.rng)
            _, density = target.build_psd_density(N)
            spectrum = np.fft.rfft(x - np.mean(x))
            spectrum *= np.sqrt(np.maximum(density, 0.0))
            x = np.fft.irfft(spectrum, n=N)
            if meaning == "variance" and np.std(x) > 0:
                x *= np.sqrt(power) / np.std(x)
            colored = True
        metadata = {
            "family": family,
            "power_parameter": power,
            "power_parameter_meaning": meaning,
            "colored": colored,
            "marginal_preserved_after_coloring": not colored,
        }
        return (x, metadata) if return_metadata else x

    @staticmethod
    def diagnostics(x: np.ndarray, *, tail_sigma: float = 4.0) -> dict[str, Any]:
        values = np.asarray(x, dtype=float)
        centered = (values - np.mean(values)) / max(np.std(values), 1e-12)
        probabilities = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        return {
            "skewness": float(skew(centered)),
            "excess_kurtosis": float(kurtosis(centered, fisher=True)),
            "tail_exceedance_probability": float(np.mean(np.abs(centered) > tail_sigma)),
            "tail_sigma": float(tail_sigma),
            "qq_probabilities": probabilities,
            "sample_quantiles": np.quantile(centered, probabilities),
            "gaussian_quantiles": norm.ppf(probabilities),
        }
