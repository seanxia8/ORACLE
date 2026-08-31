# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Synthetic multi-channel noise generation built on the single-channel core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.fft import irfft
from scipy.signal import csd as scipy_csd

from .config import CONFIG_SCHEMA_VERSION, MultiChannelConfig, NoiseConfig
from .NoiseGenerator import NoiseGenerator
from .utils import match_target_std, mean_offdiag_corrcoef, resolve_rng, sample_range, spawn_rng


class MultiChannelNoiseGenerator:
    """Generate independent or synthetically correlated multichannel noise."""

    DEFAULT_CONFIG = {
        "mode": "shared_private",
        "n_channels": 56,
        "corr_strength": 0.3,
        "channel_gain_jitter": 0.05,
        "n_latent": 2,
        "latent_strength_range": [0.1, 0.4],
        "private_strength_range": [0.8, 1.2],
        "normalize_channel_variance": False,
    }

    def __init__(
        self,
        base_config: dict[str, Any] | NoiseConfig,
        config: dict[str, Any] | MultiChannelConfig | None = None,
        rng: Any = None,
        seed: int | None = None,
        *,
        strict_config: bool = True,
    ):
        self.base_config_model = NoiseConfig.from_mapping(
            base_config, strict=strict_config
        )
        self.base_config = self.base_config_model.to_dict()
        resolved = deepcopy(self.DEFAULT_CONFIG)
        if isinstance(config, MultiChannelConfig):
            resolved.update(config.to_dict())
        elif config:
            resolved.update(config)
        self.config_model = MultiChannelConfig.from_mapping(
            resolved, strict=strict_config
        )
        self.config = self.config_model.to_dict()
        self.seed = seed
        self.rng = resolve_rng(rng=rng, seed=seed)

    def generate_independent(
        self,
        C: int,
        N: int,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate C independent channels from the base single-channel model."""
        X = self._make_base_generator().generate_ensemble(C, N)
        if self.config.get("normalize_channel_variance", True):
            X = self._normalize_channels(X)

        if return_metadata:
            covariance = np.eye(C) * self._base_expected_variance(N)
            covariance, covariance_meta = self._covariance_after_normalization(covariance, N)
            return X, {
                "metadata_schema_version": CONFIG_SCHEMA_VERSION,
                "mode": "independent",
                "n_channels": C,
                "mean_offdiag_corr": mean_offdiag_corrcoef(X),
                "implied_covariance": covariance,
                "implied_correlation": np.eye(C),
                "realized_covariance": np.cov(X),
                "realized_correlation": np.corrcoef(X),
                "per_realization_normalization": bool(self.config.get("normalize_channel_variance")),
                **covariance_meta,
            }
        return X

    def generate_shared_private(
        self,
        C: int,
        N: int,
        corr_strength: float = 0.3,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate one shared latent process plus per-channel private noise."""
        corr_strength = float(corr_strength)
        if not np.isfinite(corr_strength) or not 0.0 <= corr_strength < 1.0:
            raise ValueError("corr_strength must be finite and lie in [0, 1).")
        shared = self._make_base_generator().generate_noise(N)
        gains = 1.0 + self.rng.normal(0.0, self.config.get("channel_gain_jitter", 0.05), size=C)
        private_strengths = sample_range(
            self.rng,
            self.config.get("private_strength_range", [0.8, 1.2]),
            size=C,
        )

        private = self._make_base_generator().generate_ensemble(C, N)
        shared_weight = gains * np.sqrt(corr_strength)
        private_weight = private_strengths * np.sqrt(max(1.0 - corr_strength, 0.0))
        X = shared_weight[:, None] * shared[None, :] + private_weight[:, None] * private

        if self.config.get("normalize_channel_variance", True):
            X = self._normalize_channels(X)

        if return_metadata:
            power = self._base_expected_variance(N)
            covariance = power * (
                corr_strength * np.outer(gains, gains)
                + (1.0 - corr_strength) * np.diag(private_strengths**2)
            )
            covariance, covariance_meta = self._covariance_after_normalization(covariance, N)
            return X, {
                "metadata_schema_version": CONFIG_SCHEMA_VERSION,
                "mode": "shared_private",
                "n_channels": C,
                "corr_strength": corr_strength,
                "requested_mixing_strength": corr_strength,
                "mean_offdiag_corr": mean_offdiag_corrcoef(X),
                "gains": gains,
                "private_strengths": private_strengths,
                "implied_covariance": covariance,
                "implied_correlation": self._covariance_to_correlation(covariance),
                "realized_covariance": np.cov(X),
                "realized_correlation": np.corrcoef(X),
                "per_realization_normalization": bool(self.config.get("normalize_channel_variance")),
                **covariance_meta,
            }
        return X

    def generate_lowrank_correlated(
        self,
        C: int,
        N: int,
        n_latent: int = 2,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate channels from a low-rank latent colored-process model."""
        n_latent = max(int(n_latent), 1)
        latent = self._make_base_generator().generate_ensemble(n_latent, N)
        weights = self.rng.normal(0.0, 1.0, size=(C, n_latent))
        latent_strengths = sample_range(
            self.rng,
            self.config.get("latent_strength_range", [0.1, 0.4]),
            size=(C, n_latent),
        )
        private_strengths = sample_range(
            self.rng,
            self.config.get("private_strength_range", [0.8, 1.2]),
            size=C,
        )

        private = self._make_base_generator().generate_ensemble(C, N)
        mixing = weights * latent_strengths
        X = mixing @ latent + private_strengths[:, None] * private

        if self.config.get("normalize_channel_variance", True):
            X = self._normalize_channels(X)

        if return_metadata:
            covariance = self._base_expected_variance(N) * (
                mixing @ mixing.T + np.diag(private_strengths**2)
            )
            covariance, covariance_meta = self._covariance_after_normalization(covariance, N)
            return X, {
                "metadata_schema_version": CONFIG_SCHEMA_VERSION,
                "mode": "lowrank",
                "n_channels": C,
                "n_latent": n_latent,
                "mean_offdiag_corr": mean_offdiag_corrcoef(X),
                "mixing_matrix": mixing,
                "private_strengths": private_strengths,
                "implied_covariance": covariance,
                "implied_correlation": self._covariance_to_correlation(covariance),
                "realized_covariance": np.cov(X),
                "realized_correlation": np.corrcoef(X),
                "per_realization_normalization": bool(self.config.get("normalize_channel_variance")),
                **covariance_meta,
            }
        return X

    def generate_from_csd(
        self,
        target_csd: np.ndarray,
        N: int,
        return_metadata: bool = False,
        hermitian_tolerance: float = 1e-10,
        psd_tolerance: float = 1e-10,
        target_frequencies: np.ndarray | None = None,
        repair_policy: str = "error",
        n_realizations: int = 1,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate real Gaussian channels from a one-sided target CSD density.

        ``target_csd`` has shape ``(N // 2 + 1, C, C)`` and units of
        signal-unit^2 / Hz. Each frequency matrix must be Hermitian positive
        semidefinite. DC and, for even N, Nyquist matrices must be real because
        the corresponding Fourier coefficients of a real signal are real.
        """
        if N <= 0:
            raise ValueError("N must be positive.")
        # Always own this buffer: Hermitian symmetrization and optional repair
        # must never mutate a caller's scientific target.
        csd = np.array(target_csd, dtype=complex, copy=True)
        expected_f = N // 2 + 1
        if csd.ndim != 3 or csd.shape[1] != csd.shape[2]:
            raise ValueError(
                "target_csd must have shape (F, C, C)."
            )
        target_grid = np.fft.rfftfreq(N, 1.0 / float(self.base_config["sampling_frequency"]))
        if target_frequencies is not None:
            source_grid = np.asarray(target_frequencies, dtype=float)
            if source_grid.shape != (csd.shape[0],) or np.any(np.diff(source_grid) <= 0):
                raise ValueError("target_frequencies must be strictly increasing with length F.")
            if len(source_grid) < 2:
                raise ValueError("CSD interpolation requires at least two source frequencies.")
            if target_grid[0] < source_grid[0] or target_grid[-1] > source_grid[-1]:
                raise ValueError("Target rFFT grid exceeds supplied CSD frequency support.")
            # Piecewise linear matrix interpolation is a convex combination of
            # adjacent PSD matrices and therefore preserves the PSD cone.
            right = np.searchsorted(source_grid, target_grid, side="right")
            right = np.clip(right, 1, len(source_grid)-1)
            left = right - 1
            weight = (target_grid-source_grid[left])/(source_grid[right]-source_grid[left])
            csd = (1-weight)[:, None, None]*csd[left] + weight[:, None, None]*csd[right]
        elif csd.shape[0] != expected_f:
            raise ValueError("target_csd frequency count does not match N.")
        if csd.shape[1] == 0:
            raise ValueError("target_csd must contain at least one channel.")
        if np.any(~np.isfinite(csd)):
            raise ValueError("target_csd must contain only finite values.")

        C = csd.shape[1]
        if n_realizations <= 0:
            raise ValueError("n_realizations must be positive.")
        factors = np.zeros((expected_f, C, C), dtype=complex)
        endpoint_indices = {0}
        if N % 2 == 0 and expected_f > 1:
            endpoint_indices.add(expected_f - 1)
        min_eigenvalue = np.inf
        regularization = []

        for k, matrix in enumerate(csd):
            scale = max(float(np.max(np.abs(matrix))), 1.0)
            if not np.allclose(
                matrix, matrix.conj().T, rtol=hermitian_tolerance,
                atol=hermitian_tolerance * scale,
            ):
                raise ValueError(f"target_csd[{k}] is not Hermitian.")
            matrix = 0.5 * (matrix + matrix.conj().T)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            min_eigenvalue = min(min_eigenvalue, float(np.min(eigenvalues)))
            minimum_at_frequency = float(np.min(eigenvalues))
            if minimum_at_frequency < -psd_tolerance * scale:
                matrix, amount = self._repair_csd_matrix(matrix, repair_policy)
                regularization.append({"frequency_index": k, "amount": amount, "policy": repair_policy})
                eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            elif minimum_at_frequency < 0.0:
                clipped = np.clip(eigenvalues, 0.0, None)
                repaired = (eigenvectors * clipped[None, :]) @ eigenvectors.conj().T
                amount = float(np.linalg.norm(repaired - matrix))
                matrix = repaired
                eigenvalues = clipped
                regularization.append({
                    "frequency_index": k,
                    "amount": amount,
                    "policy": "numerical_eigenvalue_clip",
                })
            eigenvalues = np.clip(eigenvalues, 0.0, None)
            factor = eigenvectors * np.sqrt(eigenvalues)[None, :]
            factors[k] = factor
            csd[k] = matrix

            if k in endpoint_indices:
                if np.max(np.abs(matrix.imag)) > hermitian_tolerance * scale:
                    raise ValueError(
                        f"target_csd[{k}] must be real at a real-FFT endpoint."
                    )
                factors[k] = factors[k].real

        spectrum = np.zeros((n_realizations, C, expected_f), dtype=complex)
        fsn = float(self.base_config["sampling_frequency"]) * N
        for k in range(expected_f):
            if k in endpoint_indices:
                z = self.rng.standard_normal((n_realizations, C))
                spectrum[:, :, k] = np.sqrt(fsn) * (z @ factors[k].T)
            else:
                z = (self.rng.standard_normal((n_realizations, C)) +
                     1j*self.rng.standard_normal((n_realizations, C))) / np.sqrt(2)
                spectrum[:, :, k] = np.sqrt(0.5*fsn) * (z @ factors[k].T)
        X = irfft(spectrum, n=N, axis=2)
        if n_realizations == 1:
            X = X[0]
        if return_metadata:
            return X, {
                "metadata_schema_version": CONFIG_SCHEMA_VERSION,
                "mode": "target_csd",
                "n_channels": C,
                "n_samples": N,
                "sampling_frequency": float(self.base_config["sampling_frequency"]),
                "csd_units": "signal^2/Hz",
                "minimum_input_eigenvalue": float(min_eigenvalue),
                "interpolation": "convex_linear_psd_preserving" if target_frequencies is not None else None,
                "n_realizations": n_realizations,
                "repair_policy": repair_policy,
                "regularization": regularization,
                "maximum_regularization": max([item["amount"] for item in regularization], default=0.0),
            }
        return X

    def generate_from_csd_factor(
        self, factor: np.ndarray, N: int, *, return_metadata: bool = False,
        n_realizations: int = 1,
    ):
        """Generate directly from factors ``S(f) = L(f)L(f)^H``.

        The implementation stores only the supplied ``(F, C, R)`` factors and
        the output Fourier coefficients. It never materializes ``(F, C, C)``.
        Factors at DC and even-length Nyquist must be real so the corresponding
        real-FFT coefficients have a real covariance factorization.
        """
        if N <= 0 or n_realizations <= 0:
            raise ValueError("N and n_realizations must be positive.")
        factors = np.asarray(factor, dtype=complex)
        if factors.ndim != 3 or factors.shape[0] != N//2+1:
            raise ValueError("factor must have shape (N // 2 + 1, C, R).")
        if factors.shape[1] == 0 or factors.shape[2] == 0:
            raise ValueError("factor must contain at least one channel and one latent factor.")
        if np.any(~np.isfinite(factors)):
            raise ValueError("factor must contain only finite values.")

        F, C, rank = factors.shape
        endpoints = {0}
        if N % 2 == 0 and F > 1:
            endpoints.add(F - 1)
        for endpoint in endpoints:
            scale = max(float(np.max(np.abs(factors[endpoint]))), 1.0)
            if np.max(np.abs(factors[endpoint].imag)) > 1e-10 * scale:
                raise ValueError(
                    "CSD factors at DC and Nyquist must be real for real-valued output."
                )

        spectrum = np.zeros((n_realizations, C, F), dtype=complex)
        fsn = float(self.base_config["sampling_frequency"]) * N
        for k in range(F):
            if k in endpoints:
                z = self.rng.standard_normal((n_realizations, rank))
                spectrum[:, :, k] = np.sqrt(fsn) * (z @ factors[k].real.T)
            else:
                z = (
                    self.rng.standard_normal((n_realizations, rank))
                    + 1j * self.rng.standard_normal((n_realizations, rank))
                ) / np.sqrt(2.0)
                spectrum[:, :, k] = np.sqrt(0.5 * fsn) * (z @ factors[k].T)
        X = irfft(spectrum, n=N, axis=2)
        if n_realizations == 1:
            X = X[0]
        if return_metadata:
            return X, {
                "metadata_schema_version": CONFIG_SCHEMA_VERSION,
                "mode": "target_csd",
                "input_representation": "low_rank_factor",
                "factor_rank": rank,
                "n_channels": C,
                "n_samples": N,
                "sampling_frequency": float(self.base_config["sampling_frequency"]),
                "csd_units": "signal^2/Hz",
                "n_realizations": n_realizations,
                "dense_csd_materialized": False,
                "factor_storage_elements": int(factors.size),
            }
        return X

    @staticmethod
    def csd_diagnostics(X: np.ndarray, sampling_frequency: float) -> dict[str, Any]:
        """Estimate CSD diagnostics.

        Ensembles are averaged on their common rFFT grid. A single record uses
        Welch segment averaging; raw single-record bin coherence is identically
        one and is not a meaningful estimator.
        """
        values = np.asarray(X, dtype=float)
        single_record = values.ndim == 2
        if single_record:
            C, N = values.shape
            nperseg = min(256, N)
            if nperseg < 4:
                raise ValueError("Single-record CSD diagnostics require at least four samples.")
            matrices = None
            for i in range(C):
                for j in range(C):
                    frequencies, estimate = scipy_csd(
                        values[i],
                        values[j],
                        fs=float(sampling_frequency),
                        nperseg=nperseg,
                        noverlap=nperseg // 2,
                        detrend="constant",
                        scaling="density",
                    )
                    if matrices is None:
                        matrices = np.zeros((len(frequencies), C, C), dtype=complex)
                    # SciPy defines Pxy = conj(X)Y; this module's target-CSD
                    # convention is E[X Y*], so conjugate the estimate.
                    matrices[:, i, j] = estimate.conj()
            csd = matrices
            estimator = "welch_segment_average"
        else:
            if values.ndim != 3:
                raise ValueError("X must have shape (C, N) or (R, C, N).")
            R, _, N = values.shape
            coeff = np.fft.rfft(values, axis=2)
            csd = np.einsum("rcf,rdf->fcd", coeff, coeff.conj()) / R
            csd /= float(sampling_frequency) * N
            if N > 2:
                upper = N//2 + 1 - (N+1)%2
                csd[1:upper] *= 2
            frequencies = np.fft.rfftfreq(N, 1/float(sampling_frequency))
            estimator = "ensemble_periodogram_average"
        diagonal = np.real(np.diagonal(csd, axis1=1, axis2=2))
        denom = np.maximum(
            diagonal[:, :, None] * diagonal[:, None, :], 1e-30
        )
        coherence = np.abs(csd)**2 / denom
        return {
            "frequencies": frequencies,
            "csd": csd,
            "psd": diagonal,
            "coherence": coherence,
            "phase": np.angle(csd),
            "estimator": estimator,
        }

    def generate(
        self,
        N: int,
        C: int | None = None,
        mode: str | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate according to the configured multichannel mode."""
        C = int(C or self.config.get("n_channels", 1))
        mode = (mode or self.config.get("mode", "shared_private")).lower()

        if mode == "independent":
            return self.generate_independent(C, N, return_metadata=return_metadata)
        if mode == "shared_private":
            return self.generate_shared_private(
                C,
                N,
                corr_strength=float(self.config.get("corr_strength", 0.3)),
                return_metadata=return_metadata,
            )
        if mode == "lowrank":
            return self.generate_lowrank_correlated(
                C,
                N,
                n_latent=int(self.config.get("n_latent", 2)),
                return_metadata=return_metadata,
            )
        raise ValueError(f"Unsupported multichannel mode: {mode}")

    def _make_base_generator(self) -> NoiseGenerator:
        return NoiseGenerator(self.base_config, rng=spawn_rng(self.rng))

    def _normalize_channels(self, X: np.ndarray) -> np.ndarray:
        target_std = np.sqrt(self._base_expected_variance(X.shape[-1]))
        return match_target_std(X, target_std=target_std, axis=1)

    def _base_expected_variance(self, N: int) -> float:
        generator = NoiseGenerator(self.base_config, seed=0)
        _, density = generator.build_psd_density(N)
        df = float(self.base_config["sampling_frequency"]) / N
        return float(np.sum(density[1:]) * df)

    def _covariance_after_normalization(
        self, covariance: np.ndarray, N: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.config.get("normalize_channel_variance", False):
            return covariance, {}
        target_variance = self._base_expected_variance(N)
        standard_deviations = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        scale = np.sqrt(target_variance) / np.maximum(standard_deviations, 1e-15)
        normalized = scale[:, None] * covariance * scale[None, :]
        return normalized, {
            "pre_normalization_implied_covariance": covariance,
            "normalization_contract": (
                "population covariance after per-channel variance standardization; "
                "finite-record realized covariance varies"
            ),
        }

    @staticmethod
    def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.clip(np.diag(covariance), 0, None))
        return covariance / np.maximum(np.outer(std, std), 1e-15)

    @staticmethod
    def _repair_csd_matrix(matrix: np.ndarray, policy: str):
        values, vectors = np.linalg.eigh(0.5*(matrix+matrix.conj().T))
        minimum = float(np.min(values))
        if policy == "error":
            raise ValueError("target_csd is not positive semidefinite.")
        if policy in {"clip", "nearest_psd"}:
            repaired = (vectors * np.clip(values, 0, None)[None, :]) @ vectors.conj().T
            return repaired, float(np.linalg.norm(repaired-matrix))
        if policy == "diagonal_loading":
            amount = -minimum + np.finfo(float).eps
            return matrix + amount*np.eye(matrix.shape[0]), amount
        raise ValueError("repair_policy must be error, clip, nearest_psd, or diagonal_loading.")
