"""BLOCKING gate (PLAN_04 §2, §3.6 item 1): the two objectives must be genuinely
different.

If the loss were computed on the standardised representation (or the input were
whitened), plain MSE would already be the chi2-weighted residual and the two
arms would optimise the same objective. This test constructs residuals whose
energy is concentrated in low- versus high-noise bands and asserts that the
chi2 objective ranks them opposite to raw MSE — something proportional losses
cannot do.
"""

from __future__ import annotations

import numpy as np
import torch

from tidmad.loss import reconstruction_losses


def _make_nonflat_j(bands: int, dynamic_range: float = 1e3) -> torch.Tensor:
    """Exponentially increasing per-band noise power (low noise at low bands)."""
    return torch.as_tensor(np.logspace(0, np.log10(dynamic_range), bands), dtype=torch.float32)


def _resid_concentrated(bands: int, frames: int, band_slice: slice, batch: int = 2) -> torch.Tensor:
    out = torch.zeros(batch, bands, frames)
    out[:, band_slice, :] = 1.0
    return out


def test_losses_not_proportional_nonflat_j():
    bands, frames, batch = 64, 32, 2
    j = _make_nonflat_j(bands)
    input_meas = torch.randn(batch, bands, frames)
    # Low-noise bands: 0..7 ; high-noise bands: 56..63
    low, high = slice(0, 8), slice(56, 64)
    resid_low = _resid_concentrated(bands, frames, low, batch)
    resid_high = _resid_concentrated(bands, frames, high, batch)
    # out_std such that the measurement-coordinate residual is concentrated:
    # out_meas - target_meas = resid, so set out_std and target accordingly.
    target_low = torch.zeros_like(input_meas)
    target_high = torch.zeros_like(input_meas)
    # Choose out_std so that unstandardise(out_std) == resid (concentrated).
    mean_in = torch.nanmean(input_meas, dim=-1, keepdim=True)
    std_in = torch.std(input_meas, dim=-1, keepdim=True) + 1e-6
    out_std_low = (resid_low - mean_in) / std_in
    out_std_high = (resid_high - mean_in) / std_in

    m_low, c_low = _losses_for(out_std_low, target_low, input_meas, j)
    m_high, c_high = _losses_for(out_std_high, target_high, input_meas, j)

    # Raw MSE: same total energy in both cases -> comparable.
    assert torch.allclose(m_low, m_high, rtol=1e-3), "MSE must rank equal-power residuals equally"
    # chi2 penalises a residual in LOW-noise bands far more than one in
    # HIGH-noise bands (dividing by the small J of a quiet band).
    assert c_low > c_high * 10, f"chi2 must distinguish noise bands (low={c_low:.4f} high={c_high:.4f})"
    # The two objectives rank these residuals differently -> not proportional.
    assert (c_low / m_low) > (c_high / m_high), "losses must not be proportional"


def _losses_for(out_std, target, input_meas, j):
    out = reconstruction_losses(out_std, target, input_meas, j)
    return out["mse"], out["chi2"]


def test_losses_identical_under_flat_j():
    """Under a flat J the two objectives coincide (sanity check of the maths)."""
    bands, frames, batch = 16, 8, 2
    j = torch.ones(bands)
    input_meas = torch.randn(batch, bands, frames)
    target = torch.randn_like(input_meas)
    out_std = torch.randn_like(input_meas)
    losses = reconstruction_losses(out_std, target, input_meas, j)
    assert torch.allclose(losses["mse"], losses["chi2"], rtol=1e-4)
