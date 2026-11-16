"""Kinematic MeanFlow targets and learned splitting (Fan et al., Eq. 7–8).

The policy call convention here is model(z, t, r, cond). Times are either
trajectory scalars [B] for MF or per-action vectors [B, H] for AMF.
"""

import math

import torch
from torch import nn


class KinematicSplitGenerator(nn.Module):
    """Time-only split network, shared across action positions in AMF."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        # Start at the previously tested midpoint; the paper leaves init open.
        nn.init.zeros_(self.network[-2].weight)
        nn.init.zeros_(self.network[-2].bias)
        self.register_buffer("progress", torch.tensor(0.0))

    def forward(self, t, r):
        # Keep the small generator and its squared weights stable under AMP.
        with torch.autocast(device_type=t.device.type, enabled=False):
            times = torch.stack((t, r), dim=-1).to(self.network[0].weight.dtype)
            return self.network(times).squeeze(-1)


def configure_kmf(policy, mf_target_type, kmf_alpha=0.1, kmf_gamma_max=0.5):
    if mf_target_type not in ("meanflow", "kmf_midpoint", "kmf_learnable"):
        raise ValueError("mf_target_type must be meanflow, kmf_midpoint or kmf_learnable")
    if not math.isfinite(kmf_alpha) or kmf_alpha < 0:
        raise ValueError("kmf_alpha must be finite and nonnegative")
    if not math.isfinite(kmf_gamma_max) or not 0 <= kmf_gamma_max <= 1:
        raise ValueError("kmf_gamma_max must be finite and in [0, 1]")
    policy.mf_target_type = mf_target_type
    policy.kmf_alpha = float(kmf_alpha)
    policy.kmf_gamma_max = float(kmf_gamma_max)
    policy.kmf_generator = KinematicSplitGenerator() if mf_target_type == "kmf_learnable" else None
    policy.kmf_diagnostics = {}


def add_kmf_optimizer_group(policy, param_groups):
    generator = getattr(policy, "kmf_generator", None)
    if generator is not None:
        # Same learning rate as the policy, no decay for the split network.
        param_groups.append({"params": list(generator.parameters()), "weight_decay": 0.0})


def finish_kmf_loss(policy, policy_loss, generator_loss):
    if generator_loss is None:
        return policy_loss
    policy.kmf_diagnostics["kmf_policy_loss"] = policy_loss.detach().item()
    return policy_loss + policy.kmf_alpha * generator_loss


def learnable_target(model, actions, noise, t, r, cond, generator, gamma):
    """Eq. 7 target and Eq. 8 auxiliary objective with isolated gradients.

    Only the explicit split weights (including those in omega) train G. The
    locations c/z_c and all JVPs are constants in its objective, per Algorithm 1.
    The caller must detach the returned target in the policy objective.
    """
    shape = (t.shape[0], -1, 1)
    tb = t.reshape(shape)
    velocity = noise - actions
    z_t = (1 - tb) * actions + tb * noise
    # Algorithm 1 uses the same stochastic forward for u and its full-interval
    # derivative. In particular, they must share the attention dropout mask.
    prediction, d_tr = torch.autograd.functional.jvp(
        lambda state, upper, lower: model(state, upper, lower, cond=cond),
        (z_t, t, r), (velocity, torch.ones_like(t), torch.zeros_like(r)),
        create_graph=torch.is_grad_enabled(),
    )
    d_tr = d_tr.detach()
    split = generator(t, r)
    with torch.no_grad():
        c = r + split.detach() * (t - r)
        cb = c.reshape(shape)
        z_c = (1 - cb) * actions + cb * noise
        # Each subinterval derivative holds its lower time fixed.
        states = torch.cat((z_t, z_c), dim=0)
        upper = torch.cat((t, c), dim=0)
        lower = torch.cat((c, r), dim=0)
        conditions = torch.cat((cond.detach(),) * 2, dim=0)
        _, derivatives = torch.autograd.functional.jvp(
            lambda state, upper, lower: model(state, upper, lower, cond=conditions),
            (states, upper, lower),
            (torch.cat((velocity,) * 2, dim=0), torch.ones_like(upper), torch.zeros_like(lower)),
            create_graph=False,
        )
        dtype = torch.promote_types(velocity.dtype, torch.float32)
        d_tc, d_cr = derivatives.to(dtype).chunk(2, dim=0)
        d_tr = d_tr.to(dtype)

    # Form the target outside no_grad: G receives gradients through its weights,
    # while the policy/observation encoder cannot receive auxiliary gradients.
    lam = split.to(dtype).reshape(shape)
    decoupled = (1 - lam).square() * d_tc + lam.square() * d_cr
    target = velocity.detach().to(dtype) - (t - r).detach().reshape(shape) * decoupled
    delta_norm = torch.linalg.vector_norm((decoupled - d_tr).flatten(1), dim=1)
    omega = 1 + gamma * torch.exp(-1 / (delta_norm + 1e-6))
    residual_sq = (prediction.detach().to(dtype) - target).square().flatten(1).mean(dim=1)
    generator_loss = (omega * residual_sq).mean()
    diagnostics = {
        "kmf_lambda_mean": split.detach().mean().item(),
        "kmf_lambda_min": split.detach().min().item(),
        "kmf_lambda_max": split.detach().max().item(),
        "kmf_gamma": float(gamma),
        "kmf_generator_loss": generator_loss.detach().item(),
    }
    return prediction, target, generator_loss, diagnostics


def midpoint_target(model, actions, noise, t, r, cond):
    """Build a detached target using two independently evaluated subintervals.

    Both states lie on the conditional path through the same actions and noise.
    Each JVP varies the state and upper time, holding the lower time fixed.
    The 1/4 weights combine the interval weight and its relative duration.
    """
    with torch.no_grad():
        midpoint = (t + r) * 0.5
        time_shape = (t.shape[0], -1, 1)
        t_broadcast = t.reshape(time_shape)
        midpoint_broadcast = midpoint.reshape(time_shape)
        velocity = noise - actions
        z_t = (1 - t_broadcast) * actions + t_broadcast * noise
        z_midpoint = (1 - midpoint_broadcast) * actions + midpoint_broadcast * noise

        states = torch.cat((z_t, z_midpoint), dim=0)
        upper_times = torch.cat((t, midpoint), dim=0)
        lower_times = torch.cat((midpoint, r), dim=0)
        velocities = torch.cat((velocity, velocity), dim=0)
        conditions = torch.cat((cond, cond), dim=0)

        # Reuse the repository's JVP backend. The target needs no backward graph.
        _, derivatives = torch.autograd.functional.jvp(
            lambda state, upper, lower: model(state, upper, lower, cond=conditions),
            (states, upper_times, lower_times),
            (velocities, torch.ones_like(upper_times), torch.zeros_like(lower_times)),
            create_graph=False,
        )
        # AMP can return half-precision JVPs even when actions are float32.
        # Promote before summing, so two finite derivatives cannot overflow here.
        target_dtype = torch.promote_types(velocity.dtype, torch.float32)
        first_derivative, second_derivative = derivatives.to(target_dtype).chunk(2, dim=0)
        gap = (t - r).reshape(time_shape)
        target = velocity - 0.25 * gap * (first_derivative + second_derivative)
    return target
