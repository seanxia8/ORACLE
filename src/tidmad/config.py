"""Frozen configuration for the TIDMAD rung-4 arms (PLAN_04).

Every number below is a *predeclared* choice recorded before any training run.
Change any of them and the two-arm contrast is no longer the predeclared
experiment: revise the frozen commit, not the values silently.

Convention follows the upstream release (``~/src/TIDMAD/train.py``):
``channel0001`` is the SQUID readout (model input), ``channel0002`` is the
injected reference (model target), both read as ``int8`` and shifted by ``+128``
exactly as upstream does.

STFT parameters were chosen on scientific grounds (round-2 correction T2):
(a) the frame count ``M`` is an exact multiple of ``patch_len``, so ``unfold``
covers every frame and the reconstruction head folds back losslessly; (b) the
window is ~20 ms (198 656 samples at 10 MHz), a length comparable to the
upstream transformer's 20 000-sample windows but long enough for ``M = 96``
frames at a 4096-sample/2048-hop spectrogram — Paper 1's 4096-sample analysis
window scaled to a sensible frame count; and (c) the band range ``[0, 328)
bins`` covers 0 to ~800 kHz. The P1 survey found injections above that edge in
files 0016--0019; the representation is deliberately retained for the paired
metric contrast and the published full-band baseline comparison is dropped.
Real and imaginary components
are paired as two features of each of the 328 physical frequency tokens, rather
than treated as 656 unrelated bands. With ``patch_len = 4``, the temporal stage
attends over 24 positions. The band positional embedding is dimensioned to the
328 physical bands, independently of ``max_seq_len // patch_len``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TidmadSTFTConfig:
    n_fft: int = 4096
    hop_length: int = 2048
    win_length: int = 4096
    window: str = "hann"  # window function applied in the STFT
    center: bool = False  # frames extracted without padding; see frames_per_window
    n_bands_used: int = 328  # bins 0..327 -> 0..~800 kHz; paired contrast only after P1
    sample_shift: int = 128  # upstream convention: int8 + 128
    sampling_frequency_hz: float = 10_000_000.0

    @property
    def n_bands_stacked(self) -> int:
        """External measurement representation: real then imaginary rows."""
        return 2 * self.n_bands_used

    def frames_for_window(self, window_samples: int) -> int:
        """Number of STFT frames for one window of ``window_samples`` samples.

        Non-centered frames: ``M = (L - n_fft)//hop + 1``. ``hop`` and ``L`` are
        chosen so ``M`` is a multiple of ``patch_len``; consecutive windows then
        tile the file contiguously and the reconstruction head folds back
        losslessly.
        """
        return (int(window_samples) - self.n_fft) // self.hop_length + 1

    @property
    def band_centre_frequencies_hz(self) -> list[float]:
        """Centre frequency of each kept bin (positive frequencies)."""
        bin_width = self.sampling_frequency_hz / self.n_fft
        return [b * bin_width for b in range(self.n_bands_used)]


@dataclass(frozen=True)
class TidmadModelConfig:
    d_model: int = 128
    d_ff: int = 512
    n_head: int = 2
    n_time_layers: int = 1
    n_channel_layers: int = 1
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    # M=96 -> 24 temporal positions. Four frames keeps the temporal factor
    # meaningful while retaining non-overlapping patches at modest cost (D2).
    patch_len: int = 4


@dataclass(frozen=True)
class TidmadTrainConfig:
    num_steps: int = 25_000
    # Effective batch: gradients are accumulated over ``device_batch_size``
    # single-window forward passes per optimiser step (train.train_step). There
    # is deliberately no DataLoader/worker pool; the former ``num_workers``
    # field was declared but consumed by nothing and has been removed so the
    # frozen config matches execution exactly.
    device_batch_size: int = 4
    adamw_lr: float = 1e-3
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_weight_decay: float = 0.0
    muon_lr: float = 1e-3
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    warmup_steps: int = 1000
    eval_period: int = 500
    eval_num_windows: int = 32   # held-out windows scored at each eval
    grad_clip: float = 1.0       # 0 disables clipping
    checkpoint_period: int = 2500
    seed: int = 20260808
    loss: str = "mse"  # "mse" | "chi2" | "chi2_of"  -- THE ONLY field that differs between arms


@dataclass(frozen=True)
class TidmadDataConfig:
    # The data directory comes from the ``--data-dir`` CLI argument and is
    # archived into ``run_config.json`` as provenance; the former ``data_root``
    # field was declared but consumed by nothing and has been removed so the
    # frozen config matches execution exactly.
    contract_path: str = "docs/tidmad_data_contract.json"
    train_files: tuple[str, ...] = ()
    window_samples: int = 198_656  # ~19.9 ms at 10 MHz; M = (L-n_fft)//hop+1 = 96 frames (scientific choice, T2)
    fit_fraction: float = 0.7
    guard_windows: int = 4
    n_calibration_windows: int = 512  # Paper 1 Phase 4 used 512
    calibration_window_samples: int = 4096  # Phase 4: dof 4095
    # NOTE (audit M4/C7): ``psd_window`` and ``psd_average`` parameterize ONLY
    # the archived Welch PSD *density* record (``estimate`` in
    # ``psd.estimate_band_psd``). The ``j_band`` that weights the chi2 loss is
    # deliberately the data pipeline's own Hann-window STFT periodogram
    # (median-averaged), so the whitened residual |resid|^2/J is measured in the
    # loss's coordinate system. These two estimators are intentionally distinct;
    # see ``psd.estimate_band_psd``.
    psd_average: str = "median"
    psd_window: str = "boxcar"
    psd_source_glob: str = "abra_science_*.h5"  # noise-only science files for J(f)
    # T1.6 whitening positive control. The whitening identity error is measured
    # at every training start on held-out calibration windows and archived into
    # ``run_config.json``. ``None`` = record only (development); set a number to
    # enforce it as a hard gate. The value MUST be frozen from a measurement on
    # the real release files before any confirmatory run (T1.6), not invented.
    whitening_error_max: float | None = None


@dataclass(frozen=True)
class TidmadRunConfig:
    stft: TidmadSTFTConfig
    model: TidmadModelConfig
    train: TidmadTrainConfig
    data: TidmadDataConfig

    @property
    def frames_per_window(self) -> int:
        """Number of STFT frames for one training window (must divide patch_len)."""
        return self.stft.frames_for_window(self.data.window_samples)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")


#: The frozen primary configuration. Frozen at the step-8 commit; do not edit
#: after the freeze without a new timestamped commit.
_FROZEN_STFT = TidmadSTFTConfig()
_FROZEN_MODEL = TidmadModelConfig()
_FROZEN_TRAIN = TidmadTrainConfig()
_FROZEN_DATA = TidmadDataConfig()
FROZEN = TidmadRunConfig(_FROZEN_STFT, _FROZEN_MODEL, _FROZEN_TRAIN, _FROZEN_DATA)
