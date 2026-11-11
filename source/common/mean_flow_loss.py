"""Fixed-scale Cauchy loss for MF rows, retaining the original FM objective."""

import math

import torch


def validate_mf_loss(mf_loss_type="adaptive_l2", mf_loss_c=1e-3):
    if mf_loss_type not in ("adaptive_l2", "cauchy"):
        raise ValueError("mf_loss_type must be adaptive_l2 or cauchy")
    if (type(mf_loss_c) not in (int, float)
            or not math.isfinite(mf_loss_c) or mf_loss_c <= 0):
        raise ValueError("mf_loss_c must be finite and positive")


def cauchy_loss(mse, c):
    """Per-trajectory log(1 + MSE/c); c is fixed, never a learned scale."""
    return torch.log1p(mse / c)


def cauchy_mf_loss(error, fm_mask, c):
    """Reduce over the full batch, with original adaptive L2 on FM rows.

    Select rows before computing the losses so large half-precision MF errors
    cannot overflow an unused adaptive-L2 branch. FM keeps its original dtype
    and operations; MF squared-error reduction uses at least float32.
    """
    validate_mf_loss("cauchy", c)
    if (error.ndim != 3 or fm_mask.shape != (error.shape[0],)
            or fm_mask.dtype != torch.bool or fm_mask.device != error.device):
        raise ValueError("Expected error [B,H,D] and boolean fm_mask [B] on the same device")

    fm_error = error[fm_mask]
    fm_mse = torch.mean(fm_error ** 2, dim=(1, 2))
    fm_weight = 1.0 / (fm_mse + 1e-3).pow(0.5)
    fm_loss = fm_weight.detach() * fm_mse

    mf_error = error[~fm_mask]
    if mf_error.dtype in (torch.float16, torch.bfloat16):
        mf_error = mf_error.float()
    mf_mse = mf_error.square().mean(dim=(1, 2))
    losses = torch.empty(error.shape[0], dtype=mf_mse.dtype, device=error.device)
    losses[fm_mask] = fm_loss.to(losses.dtype)
    losses[~fm_mask] = cauchy_loss(mf_mse, c)
    return losses.mean()
