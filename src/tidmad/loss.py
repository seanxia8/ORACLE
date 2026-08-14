"""The two training objectives, both evaluated in measurement coordinates.

This is the heart of the rung-4 contrast (PLAN_04 §2, PLAN_03 §2A.4).

``normalise_input_sequence`` (tidmad/backbone/model.py) z-scores each row
of the ``(B*C, M)`` input independently, so for noise-dominated bands it is
*approximately whitening*. If the loss were computed on the standardised
representation, plain MSE would already be the ``Σ⁻¹``-weighted residual and the
two arms would optimise the same objective. Therefore:

1. The model consumes the standardised representation (both arms, identically).
2. The reconstruction head's output is mapped back to measurement
   (unstandardised) STFT coordinates using the per-band scale factors recorded
   during normalisation.
3. Both losses are evaluated on that measurement-coordinate residual:
   ``loss_mse = resid.pow(2).mean()`` and
   ``loss_chi2 = (resid.pow(2) / J).mean()`` with ``J`` the per-band noise PSD.

``J`` is the Paper 1 Phase-4 PSD (boxcar, median, 512 calibration windows)
evaluated at the STFT band centres; a global periodogram normalization constant
is immaterial to the contrast and is absorbed into the loss/lr scale.
"""

from __future__ import annotations

import torch
from torch import Tensor


def per_row_stats(x: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Mean and standard deviation over the last axis, matching
    ``tidmad.backbone.model.normalise_input_sequence`` exactly."""
    x_mean = torch.nanmean(x, dim=-1, keepdim=True)
    x_std = torch.std(x, dim=-1, keepdim=True)
    return x_mean, x_std


def unstandardise(out_std: Tensor, mean: Tensor, std: Tensor, eps: float = 1e-6) -> Tensor:
    """Map a standardised output back to measurement STFT coordinates."""
    return out_std * (std + eps) + mean


def reconstruction_losses(
    out_std: Tensor,
    target_meas: Tensor,
    input_meas: Tensor,
    j_per_band: Tensor,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Both objectives on the measurement-coordinate residual.

    Parameters
    ----------
    out_std:
        ``(B, C, M)`` model output in standardised coordinates.
    target_meas:
        ``(B, C, M)`` STFT of the reference channel in measurement coordinates.
    input_meas:
        ``(B, C, M)`` STFT of the readout channel in measurement coordinates;
        its per-band normalisation statistics are the scale factors that
        ``normalise_input_sequence`` applied to the model input.
    j_per_band:
        ``(C,)`` positive per-band noise power (``J(f)`` at each band centre,
        tiled over the real/imaginary stacking).
    """
    mean_in, std_in = per_row_stats(input_meas, eps=eps)
    out_meas = unstandardise(out_std, mean_in, std_in, eps=eps)
    resid = out_meas - target_meas
    j = j_per_band.to(resid.dtype).reshape(1, -1, 1)
    if torch.any(j <= 0):
        raise ValueError("J(f) must be strictly positive for the chi2 objective")
    return {
        "mse": resid.pow(2).mean(),
        "chi2": (resid.pow(2) / j).mean(),
    }
