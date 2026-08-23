"""TIDMAD readout → band–frame spectrogram tensors (PLAN_04 §3.1).

Follows the release's own convention (``train.py``/``inference.py``/``benchmark.py``
all hardcode it): ``channel0001`` is the SQUID readout (model input) and
``channel0002`` is the injected reference (model target), both read as ``int8``
and shifted by ``+128`` exactly as upstream does.

The STFT uses a non-centered frame extraction with a Hann window at 50% hop
(COLA condition), so consecutive windows tile the file contiguously and the
round-trip ``istft(stft(x)) == x`` is exact in the interior.
"""

from __future__ import annotations

import numpy as np

from .config import TidmadSTFTConfig


def window_function(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(n)
    if name == "boxcar":
        return np.ones(n, dtype=np.float64)
    raise ValueError(f"unsupported window: {name!r} (use 'hann' or 'boxcar')")


def stft(x: np.ndarray, cfg: TidmadSTFTConfig) -> np.ndarray:
    """Short-time Fourier transform, ``(F, M)`` complex.

    ``x`` is one window of ``window_samples`` samples. Frames start at
    ``m*hop`` for ``m = 0..M-1`` with ``M = (L - n_fft)//hop + 1``. Returns the
    one-sided complex spectrum ``(n_fft//2 + 1, M)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n_fft, hop = cfg.n_fft, cfg.hop_length
    win = window_function(cfg.window, cfg.win_length)
    m = (x.size - n_fft) // hop + 1
    if m <= 0:
        raise ValueError(f"window_samples {x.size} too short for n_fft {n_fft}")
    out = np.empty((n_fft // 2 + 1, m), dtype=np.complex128)
    for j in range(m):
        seg = x[j * hop : j * hop + n_fft] * win
        out[:, j] = np.fft.rfft(seg, n=n_fft)
    return out


def istft(X: np.ndarray, cfg: TidmadSTFTConfig, length: int | None = None) -> np.ndarray:
    """Inverse STFT by weighted overlap-add (COLA).

    The window is applied once in the forward pass; the overlap-add divides by
    the running *sum of the window* (``sum_m w[n - m*hop]``), which is constant
    in the interior for a Hann window at 50% hop. Dividing by the squared
    window sum would leave a cos² ripple and break the round trip.
    """
    X = np.asarray(X, dtype=np.complex128)
    n_fft, hop = cfg.n_fft, cfg.hop_length
    win = window_function(cfg.window, cfg.win_length)
    m = X.shape[1]
    length = length if length is not None else (m - 1) * hop + n_fft
    out = np.zeros(length, dtype=np.float64)
    wsum = np.zeros(length, dtype=np.float64)
    for j in range(m):
        seg = np.fft.irfft(X[:, j], n=n_fft)
        start = j * hop
        stop = start + n_fft
        out[start:stop] += seg
        wsum[start:stop] += win
    nz = wsum > 1e-12
    out[nz] /= wsum[nz]
    return out


def series_to_spectrogram(series: np.ndarray, cfg: TidmadSTFTConfig) -> np.ndarray:
    """One window → ``(2F, M)`` real spectrogram (real and imaginary stacked).

    Only the first ``n_bands_used`` frequency bins are kept; the rest are
    dropped, which is what makes the reconstruction band-limited.
    """
    X = stft(series, cfg)
    X = X[: cfg.n_bands_used, :]  # (F_used, M)
    return np.concatenate([X.real, X.imag], axis=0)  # (2*F_used, M)


def spectrogram_to_series(Z: np.ndarray, cfg: TidmadSTFTConfig, length: int) -> np.ndarray:
    """Inverse of :func:`series_to_spectrogram`: ``(2F, M)`` → ``(length,)``.

    Dropped frequency bins are reconstructed as zero.
    """
    f_used = cfg.n_bands_used
    re, im = Z[:f_used], Z[f_used:]
    X = re + 1j * im
    return istft(X, cfg, length=length)


def read_channel_pair(path, contract, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(readout, reference)`` float windows following the release convention.

    Both channels are read as ``int8`` and shifted by ``+128`` exactly as
    upstream ``train.py`` does before the STFT.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # Contract fields are full release paths, e.g.
        # ``timeseries/channel0001/timeseries``; do not prepend a group.
        squid = np.asarray(f[contract.squid_dataset][start : start + length], dtype=np.float64)
        ref = np.asarray(f[contract.reference_dataset][start : start + length], dtype=np.float64)
    return squid + 128.0, ref + 128.0


def build_window_stft(
    path, contract, start: int, cfg: TidmadSTFTConfig, window_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(input_spectrogram, target_spectrogram)`` for one window of a file."""
    squid, ref = read_channel_pair(path, contract, start, window_samples)
    return series_to_spectrogram(squid, cfg), series_to_spectrogram(ref, cfg)
