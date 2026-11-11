"""Trajectory-level MF curriculum, applied outside the chosen per-row loss.

Based on the progressive interval weighting in arXiv:2511.19065v2, adapted
to AMF vector times by averaging the gap across the action horizon.
"""

from contextlib import contextmanager
import copy
import math
import random

import numpy as np
import torch

from source.common.mean_flow_schedule import fm_only_epoch_count
from source.common.mean_flow_loss import cauchy_loss, validate_mf_loss


def gap_config(config=None):
    result = dict(enabled=False, mode="progressive", start_at="joint_finetune",
                  aggregation="trajectory_mean", k=1.0, lambda_value=None,
                  calibration_batches=64, calibration_seed=0, validation_seed=0)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown progressive_gap_weighting keys: {sorted(unknown)}")
        result.update(config)
    if type(result["enabled"]) is not bool:
        raise ValueError("gap weighting enabled must be boolean")
    if result["mode"] not in ("progressive", "constant_small_gap"):
        raise ValueError("gap mode must be progressive or constant_small_gap")
    if result["start_at"] != "joint_finetune" or result["aggregation"] != "trajectory_mean":
        raise ValueError("Only joint_finetune / trajectory_mean is supported")
    for key in ("k", "lambda_value"):
        value = result[key]
        if value is None and key == "lambda_value":
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("calibration_batches", "calibration_seed", "validation_seed"):
        value = result[key]
        if type(value) is not int or value < (1 if key == "calibration_batches" else 0):
            raise ValueError(f"Invalid {key}")
    if max(result["calibration_seed"], result["validation_seed"]) >= 2**32:
        raise ValueError("Calibration and validation seeds must be below 2**32")
    return result


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def isolated_rng(seed):
    """Calibration and validation must not consume the training random stream."""
    state = capture_rng()
    try:
        random.seed(seed)
        np.random.seed(seed)
        # Do not queue a CUDA seed change before CUDA has been initialized.
        # Such a lazy seed cannot be undone by restoring only the CPU RNG.
        torch.random.default_generator.manual_seed(seed)
        if state['cuda'] is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(state)


def epoch_batch_sizes(dataset_size, batch_size, drop_last=False, max_train_steps=None):
    """Actual batches consumed per epoch, including a partial final batch."""
    if dataset_size <= 0 or batch_size <= 0:
        raise ValueError("Training dataset and batch size must be positive")
    sizes = [batch_size] * (dataset_size // batch_size)
    if not drop_last and dataset_size % batch_size:
        sizes.append(dataset_size % batch_size)
    if max_train_steps is not None:
        if type(max_train_steps) is not int or max_train_steps <= 0:
            raise ValueError("max_train_steps must be a positive integer or null")
        sizes = sizes[:max_train_steps]
    if not sizes:
        raise ValueError("Training dataloader contains no batches")
    return sizes


def calibrate_gap_lambda(policy, batch_sizes, flow_ratio, batches=64, seed=0):
    """Estimate E[1-gap | MF] with the real sampler, excluding diagonal rows."""
    original_ratio = policy.flow_ratio
    total, count = 0.0, 0
    try:
        policy.flow_ratio = flow_ratio
        with isolated_rng(seed), torch.no_grad():
            for _ in range(batches):
                size = batch_sizes[np.random.randint(len(batch_sizes))]
                t, r, fm_mask = policy.sample_t_r(size, "cpu", return_mask=True)
                values = (1 - (t.float() - r.float()).mean(dim=1))[~fm_mask]
                total += values.double().sum().item()
                count += values.numel()
    finally:
        policy.flow_ratio = original_ratio
    if count == 0:
        return None
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Cannot calibrate lambda for this MF interval distribution")
    return count / total


class GapWeightingSchedule:
    """Count training batches only; the workspace checkpoints the mutable state."""

    def __init__(self, config, policy, num_epochs, batch_sizes, flow_ratio,
                 flow_schedule=None, next_epoch=0, saved_state=None):
        cfg = gap_config(config)
        if not 0 <= flow_ratio <= 1:
            raise ValueError("flow_ratio must lie in [0, 1]")
        start = fm_only_epoch_count(num_epochs, flow_schedule)
        active = cfg["enabled"] and flow_ratio < 1 and start < num_epochs
        total = (num_epochs - start) * len(batch_sizes) if active else 0
        seen = max(0, next_epoch - start) * len(batch_sizes) if active else 0
        if not 0 <= next_epoch <= num_epochs:
            raise ValueError("Resume epoch is outside the training budget")
        plan = dict(num_epochs=num_epochs, joint_start_epoch=start,
                    batch_sizes=list(batch_sizes), flow_ratio=flow_ratio,
                    horizon=policy.horizon, n_obs_steps=policy.n_obs_steps,
                    monotone_schedule_ratio=policy.monotone_schedule_ratio,
                    time_dist=list(policy.time_dist))
        if saved_state is not None:
            if (saved_state.get('version') != 1 or saved_state['active'] != active
                    or saved_state["plan"] != plan or saved_state["config"] != cfg
                    or saved_state["joint_train_batches_seen"] != seen
                    or saved_state["planned_joint_batches"] != total):
                raise ValueError("Gap checkpoint/config/epoch mismatch; do not replay epochs or change the curriculum on resume")
            self.state = copy.deepcopy(saved_state)
        else:
            value = cfg["lambda_value"]
            if active and value is None:
                value = calibrate_gap_lambda(policy, batch_sizes, flow_ratio,
                                             cfg["calibration_batches"], cfg["calibration_seed"])
            self.state = dict(version=1, config=cfg, active=active, plan=plan,
                              joint_train_batches_seen=seen, planned_joint_batches=total,
                              lambda_value=value if active else None)
        if active:
            gap_config(dict(lambda_value=self.state['lambda_value']))
            if self.state['lambda_value'] is None:
                raise ValueError("An active MF curriculum requires a calibrated lambda")

    def context(self):
        s = self.state
        if not s["active"]:
            return None
        total = s["planned_joint_batches"]
        q = min(1.0, s["joint_train_batches_seen"] / (total - 1)) if total > 1 else 1.0
        if s["config"]["mode"] == "constant_small_gap":
            q = 0.0
        return dict(progress=q, gap_lambda=s["lambda_value"], k=s["config"]["k"])

    def consume(self, epoch):
        s = self.state
        if s["active"] and epoch >= s["plan"]["joint_start_epoch"]:
            if s["joint_train_batches_seen"] >= s["planned_joint_batches"]:
                raise ValueError("Consumed more joint training batches than planned")
            s["joint_train_batches_seen"] += 1


def gap_weights(t, r, fm_mask, progress, gap_lambda, k=1.0):
    if t.ndim != 2 or r.shape != t.shape or fm_mask.shape != (t.shape[0],):
        raise ValueError("Expected t/r [B,H] and fm_mask [B]")
    if fm_mask.dtype != torch.bool or fm_mask.device != t.device:
        raise ValueError("FM mask must be boolean and on the time tensor device")
    if not math.isfinite(progress) or not math.isfinite(k) or k <= 0:
        raise ValueError("Progress must be finite and k must be positive")
    if not math.isfinite(gap_lambda) or gap_lambda <= 0:
        raise ValueError("lambda must be finite and positive")
    with torch.no_grad():
        if not bool(torch.isfinite(t).all() & torch.isfinite(r).all()
                    & (r >= 0).all() & (r <= t).all() & (t <= 1).all()):
            raise ValueError("Expected finite times 0 <= r <= t <= 1")
        gap = (t.float() - r.float()).mean(dim=1)
        qk = min(1.0, max(0.0, progress)) ** k
        beta = qk + (1 - qk) * gap_lambda * (1 - gap)
        return torch.where(fm_mask, torch.ones_like(beta), beta)


def gap_loss(error, t, r, fm_mask, context=None, diagnostics=False,
             mf_loss_type="adaptive_l2", mf_loss_c=1e-3):
    """Weight after adaptive L2 or MF-only Cauchy, before the full batch mean."""
    validate_mf_loss(mf_loss_type, mf_loss_c)
    if error.ndim != 3 or error.shape[:2] != t.shape:
        raise ValueError("Expected error [B,H,D] matching t/r")
    if fm_mask.shape != (error.shape[0],) or fm_mask.dtype != torch.bool or fm_mask.device != error.device:
        raise ValueError("Expected boolean fm_mask [B] on the error device")
    mse = error.float().square().mean(dim=(1, 2))
    adaptive = mse * (mse.detach() + 1e-3).pow(-0.5)
    objective = adaptive
    if mf_loss_type == "cauchy":
        objective = torch.where(fm_mask, adaptive, cauchy_loss(mse, mf_loss_c))
    weights = (torch.ones_like(mse) if context is None
               else gap_weights(t, r, fm_mask, **context))
    weighted = weights * objective
    loss = weighted.mean()
    if not diagnostics:
        return loss
    with torch.no_grad():
        gap = (t.float() - r.float()).mean(dim=1)
        details = {"fm_ratio_actual": fm_mask.float().mean().item()}
        if context is not None:
            details.update(gap_progress=context["progress"], gap_lambda=context["gap_lambda"])
        for label, mask in (("fm", fm_mask), ("mf", ~fm_mask)):
            count = int(mask.sum())
            details[f"{label}_count"] = count
            if count:
                details[f"{label}_mse"] = mse[mask].mean().item()
                details[f"{label}_adaptive_loss"] = adaptive[mask].mean().item()
                details[f"{label}_objective_loss"] = objective[mask].mean().item()
        mf = ~fm_mask
        if mf.any():
            details.update(mf_gap_mean=gap[mf].mean().item(),
                           mf_weight_mean=weights[mf].mean().item(),
                           mf_weight_min=weights[mf].min().item(),
                           mf_weight_max=weights[mf].max().item(),
                           mf_weighted_loss=weighted[mf].mean().item())
        # These are unweighted bootstrap residuals, NOT ground-truth velocity errors.
        for i, (lo, hi) in enumerate(((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))):
            mask = mf & (gap >= lo) & (gap < hi)
            count = int(mask.sum())
            details[f"mf_gap_bin_{i}_count"] = count
            if count:
                details[f"mf_gap_bin_{i}_mse"] = mse[mask].mean().item()
    return loss, details


def aggregate_diagnostics(rows):
    """Sample-count-weighted validation means; absent groups stay absent."""
    result = {}
    for key in set().union(*(row.keys() for row in rows)):
        if key.endswith("_count"):
            result[key] = sum(row.get(key, 0) for row in rows)
        elif key.endswith(("_mse", "_adaptive_loss", "_objective_loss")):
            group = key.removesuffix("_mse").removesuffix("_adaptive_loss").removesuffix("_objective_loss")
            count_key = group + "_count"
            count = sum(row.get(count_key, 0) for row in rows)
            if count:
                result[key] = sum(row.get(key, 0) * row.get(count_key, 0) for row in rows) / count
    return result
